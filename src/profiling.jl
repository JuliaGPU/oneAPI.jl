# Integrated profiler, built on Intel's Profiling Tools Interfaces (PTI).
#
# The display code is adapted from Metal.jl's integrated profiler.

module Profiling

using ..oneAPI
using ..oneAPI: PTI, global_stream

using Printf: @sprintf
using Statistics: mean, std, quantile

using PrettyTables: pretty_table, TextHighlighter
using Crayons: @crayon_str


#
# data collection
#

const views = (PTI.PTI_VIEW_DEVICE_GPU_KERNEL, PTI.PTI_VIEW_DEVICE_GPU_MEM_COPY,
               PTI.PTI_VIEW_DEVICE_GPU_MEM_FILL, PTI.PTI_VIEW_DRIVER_API)

Base.@kwdef struct ProfileResults
    # device-side operations, and the Level Zero API calls made by the host, with times in
    # seconds since the start of the profile and PTI's correlation IDs linking them
    device::NamedTuple
    host::NamedTuple

    wall::Float64
    trace::Bool
    raw::Bool
end

function profile_internally(@nospecialize(f); trace::Bool=false, raw::Bool=false)
    PTI.is_available() ||
        error("Integrated profiling is not supported on this platform; use `oneAPI.@profile external=true` with an external profiler instead")

    stream = global_stream(oneAPI.context(), oneAPI.device())
    synchronize(stream)

    PTI.set_callbacks()

    # discard what a failed session left behind, in PTI or in the queue of buffers
    PTI.ptiFlushAllViews()
    PTI.foreach_record(Returns(nothing))
    lost = PTI.lost_buffers()

    enabled = PTI.pti_view_kind[]
    local t_start, t_stop, records
    try
        for kind in views
            PTI.ptiViewEnable(kind)
            push!(enabled, kind)
        end
        t_start = PTI.ptiViewGetTimestamp()
        try
            f()
        finally
            synchronize(stream)
        end
        t_stop = PTI.ptiViewGetTimestamp()
    finally
        # attempt all of the cleanup, so that no view is left enabled
        errors = Any[]
        for kind in enabled
            try
                PTI.ptiViewDisable(kind)
            catch err
                push!(errors, err)
            end
        end
        try
            PTI.ptiFlushAllViews()
            records = decode_records()
        catch err
            push!(errors, err)
        end
        isempty(errors) || throw(first(errors))
    end

    if PTI.lost_buffers() > lost
        @warn "The profiler could not allocate memory for its records; some activity is missing from the results"
    end

    seconds(t) = relative_seconds(t, t_start)
    device = (; records.device.kind, records.device.name, records.device.id,
              records.device.bytes, start=seconds.(records.device.start),
              stop=seconds.(records.device.stop))
    host = (; records.host.name, records.host.id, records.host.tid, records.host.unfinished,
            start=seconds.(records.host.start), stop=seconds.(records.host.stop))
    return ProfileResults(; device, host, wall=seconds(t_stop), trace, raw)
end

# Other tasks can submit work before the start of the profile, so subtract in a signed type.
# Integer arithmetic retains the precision of the timestamps.
relative_seconds(t, t_start) = (Int128(t) - Int128(t_start)) / 1e9

function decode_records()
    device = (kind=Symbol[], name=String[], id=Int[], bytes=Union{Missing,Int}[],
              start=UInt64[], stop=UInt64[])
    host = (name=String[], id=Int[], tid=Int[], start=UInt64[], stop=UInt64[],
            unfinished=Bool[])

    function push_device!(kind, r, bytes)
        push!(device.kind, kind)
        push!(device.name, unsafe_string(r._name))
        push!(device.id, r._correlation_id)
        push!(device.bytes, bytes)
        push!(device.start, r._start_timestamp)
        push!(device.stop, r._end_timestamp)
    end

    PTI.foreach_record() do ptr
        kind = unsafe_load(ptr)._view_kind
        if kind == PTI.PTI_VIEW_DEVICE_GPU_KERNEL
            r = unsafe_load(convert(Ptr{PTI.pti_view_record_kernel}, ptr))
            push_device!(:kernel, r, missing)
        elseif kind == PTI.PTI_VIEW_DEVICE_GPU_MEM_COPY
            r = unsafe_load(convert(Ptr{PTI.pti_view_record_memory_copy}, ptr))
            push_device!(:copy, r, Int(r._bytes))
        elseif kind == PTI.PTI_VIEW_DEVICE_GPU_MEM_FILL
            r = unsafe_load(convert(Ptr{PTI.pti_view_record_memory_fill}, ptr))
            push_device!(:fill, r, Int(r._bytes))
        elseif kind == PTI.PTI_VIEW_DRIVER_API
            r = unsafe_load(convert(Ptr{PTI.pti_view_record_api}, ptr))
            push!(host.name, PTI.api_name(r._api_group, r._api_id))
            push!(host.id, r._correlation_id)
            push!(host.tid, r._thread_id)
            push!(host.start, r._start_timestamp)
            push!(host.stop, r._end_timestamp)
            push!(host.unfinished, r._return_code == UInt32(oneL0.RESULT_NOT_READY))
        end
    end

    return (; device, host)
end


#
# display
#

format_percentage(x::Number) = @sprintf("%.2f%%", x * 100)

function format_time(ts::Number...)
    t = ts[1]
    range, unit = if abs(t) < 1e-6
        1e9, "ns"
    elseif abs(t) < 1e-3
        1e6, "µs"
    elseif abs(t) < 1
        1e3, "ms"
    else
        1, "s"
    end

    strs = String[]
    let io = IOBuffer()
        Base.print(io, round(t * range, digits=2), " ", unit)
        push!(strs, String(take!(io)))
    end
    for t in ts[2:end]
        let io = IOBuffer()
            Base.print(io, round(t * range, digits=2))
            push!(strs, String(take!(io)))
        end
    end

    length(strs) == 1 ? strs[1] : strs
end

# time during which at least one operation was executing; operations can overlap
function busy_time(start, stop)
    isempty(start) && return 0.0
    perm = sortperm(start)
    total = 0.0
    cur_start = start[perm[1]]
    cur_stop = stop[perm[1]]
    for k in 2:length(perm)
        s, e = start[perm[k]], stop[perm[k]]
        if s > cur_stop
            total += cur_stop - cur_start
            cur_start, cur_stop = s, e
        else
            cur_stop = max(cur_stop, e)
        end
    end
    total + (cur_stop - cur_start)
end

function summarize_trace(names, times, span)
    groups = Dict{String,Vector{Float64}}()
    for (name, t) in zip(names, times)
        push!(get!(Vector{Float64}, groups, name), t)
    end

    n = length(groups)
    out_name = Vector{String}(undef, n)
    out_time = Vector{Float64}(undef, n)
    out_calls = Vector{Int}(undef, n)
    out_dist = Vector{Union{Missing,@NamedTuple{std::Float64, mean::Float64, min::Float64, max::Float64}}}(undef, n)
    for (i, (name, ts)) in enumerate(groups)
        out_name[i] = name
        out_time[i] = sum(ts)
        out_calls[i] = length(ts)
        out_dist[i] = length(ts) == 1 ? missing :
            (; std=std(ts), mean=mean(ts), min=minimum(ts), max=maximum(ts))
    end
    out_ratio = out_time ./ span

    perm = sortperm(out_ratio; rev=true)
    return (name=out_name[perm], time=out_time[perm], calls=out_calls[perm],
            time_dist=out_dist[perm], time_ratio=out_ratio[perm])
end

function time_highlighters(df)
    relevant = Float64[t for t in df.time if t !== missing && t >= 1e-8]
    isempty(relevant) && return TextHighlighter[]
    p75 = quantile(relevant, 0.75)
    p95 = quantile(relevant, 0.95)

    atleast(v, p) = v !== missing && v >= p
    hl_p95 = TextHighlighter((data, i, j) -> keys(data)[j] == :time && atleast(data[j][i], p95),
                             crayon"red")
    hl_p75 = TextHighlighter((data, i, j) -> keys(data)[j] == :time && atleast(data[j][i], p75),
                             crayon"yellow")
    hl_bold = TextHighlighter((data, i, j) -> keys(data)[j] == :name && atleast(data.time[i], p75),
                              crayon"bold")
    TextHighlighter[hl_p95, hl_p75, hl_bold]
end

const trace_column_names = Dict(:id => "ID", :start => "Start", :time => "Time",
                                :tid => "Thread", :size => "Size",
                                :throughput => "Throughput", :name => "Name")
const summary_column_names = Dict(:time_ratio => "Time (%)", :time => "Total time",
                                  :calls => "Calls", :time_dist => "Time distribution",
                                  :name => "Name")

# Calls that poll for the completion of work, which oneAPI.jl and oneMKL make while
# synchronizing. Functions that wait for completion can also poll, with a zero timeout,
# so those are only hidden when the work was not done yet.
const status_queries = Set(["zeEventQueryStatus", "zeFenceQueryStatus"])

show_host_call(name::String, unfinished::Bool, raw::Bool) =
    raw || !(unfinished || name in status_queries)

function device_trace(device)
    perm = sortperm(device.start)
    time = (device.stop .- device.start)[perm]
    size = device.bytes[perm]
    df = (id=device.id[perm], start=device.start[perm], time, size,
          throughput=[b === missing || t <= 0 ? missing : b / t for (b, t) in zip(size, time)],
          name=device.name[perm])
    if all(ismissing, df.size)
        df = (; df.id, df.start, df.time, df.name)
    end
    return df
end

function trace_formatter(df)
    return function(v, _, j)
        v === missing && return "-"
        col = keys(df)[j]
        if col in (:start, :time)
            format_time(v)
        elseif col === :size
            Base.format_bytes(v)
        elseif col === :throughput
            "$(Base.format_bytes(round(Int, v)))/s"
        else
            v
        end
    end
end

function print_trace(io, df, crop)
    header = [trace_column_names[k] for k in keys(df)]
    alignment = [k === :name ? :l : :r for k in keys(df)]
    pretty_table(io, df; column_labels=header, alignment, formatters=[trace_formatter(df)],
                 highlighters=time_highlighters(df),
                 fit_table_in_display_horizontally=(crop == :horizontal),
                 fit_table_in_display_vertically=false)
end

function summary_formatter(df)
    return function(v, _, j)
        col = keys(df)[j]
        if col === :time_ratio
            format_percentage(v)
        elseif col === :time
            format_time(v)
        elseif col === :time_dist
            v === missing && return ""
            m, s, lo, hi = format_time(v.mean, v.std, v.min, v.max)
            @sprintf("%9s ± %-6s (%6s ‥ %s)", m, s, lo, hi)
        else
            v
        end
    end
end

function print_summary(io, df, crop)
    header = [summary_column_names[k] for k in keys(df)]
    alignment = [k in (:name, :time_dist) ? :l : :r for k in keys(df)]
    pretty_table(io, df; column_labels=header, alignment, formatters=[summary_formatter(df)],
                 highlighters=time_highlighters(df),
                 fit_table_in_display_horizontally=(crop == :horizontal),
                 fit_table_in_display_vertically=false)
end

function summary_table(names, times, den)
    s = summarize_trace(names, times, den)
    columns = [:time_ratio, :time, :calls]
    any(!ismissing, s.time_dist) && push!(columns, :time_dist)
    push!(columns, :name)
    NamedTuple{Tuple(columns)}(Tuple(s[c] for c in columns))
end

function table_crop(io)
    if get(io, :is_pluto, false) || get(io, :jupyter, false)
        :none
    elseif io isa Base.TTY || get(io, :limit, false)::Bool
        :horizontal
    else
        :none
    end
end

function Base.show(io::IO, r::ProfileResults)
    # render into a buffer, so that the code below (which takes long to compile) does not
    # need to be specialized on, and precompiled for, every kind of output stream
    buf = IOContext(IOContext(IOBuffer(), io), :color => get(io, :color, false)::Bool,
                    :displaysize => displaysize(io))
    show_results(buf, r, table_crop(io))
    write(io, take!(buf.io))
    return
end

function show_results(io::IOContext{IOBuffer}, r::ProfileResults, crop::Symbol)
    device = r.device
    shown = [show_host_call(name, unfinished, r.raw)
             for (name, unfinished) in zip(r.host.name, r.host.unfinished)]
    host = map(col -> col[shown], r.host)
    if isempty(device.name) && isempty(host.name)
        print(io, "No GPU operations or Level Zero API calls were profiled.")
        return
    end

    den = r.wall == 0 ? 1.0 : r.wall
    println(io, "Profiled over $(format_time(r.wall)).")

    # host-side activity: the Level Zero API calls, made by oneAPI.jl and by oneMKL
    if isempty(host.name)
        println(io, "\nNo host-side activity was recorded.")
    else
        times = host.stop .- host.start
        htotal = sum(times)
        println(io, "\nHost-side activity: $(length(host.name)) Level Zero API calls taking $(format_time(htotal)) ",
                    "($(format_percentage(htotal / den)) of wall-clock)")
        if r.trace
            perm = sortperm(host.start)
            columns = [:id, :start, :time]
            length(unique(host.tid)) > 1 && push!(columns, :tid)
            push!(columns, :name)
            all_columns = (id=host.id[perm], start=host.start[perm], time=times[perm],
                           tid=host.tid[perm], name=host.name[perm])
            df = NamedTuple{Tuple(columns)}(Tuple(all_columns[c] for c in columns))
            print_trace(io, df, crop)
        else
            print_summary(io, summary_table(host.name, times, den), crop)
        end
    end

    # device-side activity: kernels, copies and fills
    if isempty(device.name)
        println(io, "\nNo device-side activity was recorded.")
    else
        busy = busy_time(device.start, device.stop)
        println(io, "\nDevice-side activity: GPU was busy for $(format_time(busy)) ",
                    "($(format_percentage(busy / den)) of wall-clock)")
        if r.trace
            print_trace(io, device_trace(device), crop)
        else
            print_summary(io, summary_table(device.name, device.stop .- device.start, den), crop)
        end
    end
end


#
# benchmarking
#

function benchmark_and_profile(@nospecialize(f); time::Real=1.0, kwargs...)
    # warm up (compilation etc.) outside of the profiled region
    f()
    synchronize()

    function harness()
        t0 = Base.time_ns()
        while (Base.time_ns() - t0) / 1e9 < time
            f()
            synchronize()
        end
    end

    profile_internally(harness; kwargs...)
end


#
# precompilation
#

using PrecompileTools: @compile_workload

# Displaying results takes seconds to compile. Each table's columns depend on the data
# (e.g. a distribution column for repeated calls), so cover the variants that are common.
@compile_workload begin
    host = (name=["zeCommandListAppendLaunchKernel", "zeEventQueryStatus",
                  "zeCommandListAppendLaunchKernel"],
            id=[1, 0, 2], tid=[1, 1, 2], unfinished=[false, true, false],
            start=[0.0, 2e-6, 4e-6], stop=[1e-6, 3e-6, 5e-6])
    device = (kind=[:kernel, :copy], name=["kernel", "copy"], id=[1, 2],
              bytes=Union{Missing,Int}[missing, 64], start=[1e-6, 5e-6], stop=[2e-6, 6e-6])
    single_threaded_host = map(col -> col[1:2], host)
    kernels_only = (kind=[:kernel, :kernel], name=["kernel", "kernel"], id=[1, 2],
                    bytes=Union{Missing,Int}[missing, missing], start=[1e-6, 5e-6],
                    stop=[2e-6, 6e-6])
    for (host, device) in ((host, device), (single_threaded_host, kernels_only)),
        trace in (false, true)
        r = ProfileResults(; device, host, wall=1e-5, trace, raw=false)
        show(IOContext(IOBuffer(), :limit => true), MIME"text/plain"(), r)
    end
end

end
