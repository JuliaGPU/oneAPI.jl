# Profiler control

"""
    oneAPI.@profile [trace=false] [raw=false] code...
    oneAPI.@profile external=true code...

Profile the GPU execution of `code`.

There are two modes of operation, depending on whether `external` is `true` or `false`.
The default value depends on whether Julia is being run under an external profiler.

## Integrated profiler (`external=false`, the default)

oneAPI.jl collects the activity of `code` using Intel's Profiling Tools Interfaces for GPU
(PTI), and displays two tables:

  - **Host-side activity**: the Level Zero API calls made while running `code`, by
    oneAPI.jl and by libraries such as oneMKL, grouped by function and sorted by time.
  - **Device-side activity**: the kernels, memory copies and fills that executed on the
    GPU, grouped by name.

To display a chronological trace instead, set `trace` to `true`. The ID column then links
API calls to the GPU operations they submitted. Calls that poll for the completion of GPU
work are hidden when the work was not done yet, unless `raw` is set to `true`.

Slow entries are highlighted: yellow is among the slowest 25%, red among the slowest 5%.

Work submitted to the calling task's stream is synchronized before and after `code`, so
the profile covers its execution. The profile can also contain work from other tasks.
GPU operations may overlap in time, so the device-side percentages are relative to the
wall-clock time of `code`, not to the sum of the durations.

## External profilers (`external=true`, when an external profiler is detected)

For more advanced profiling, use an external profiler such as Intel's
[unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) or VTune. When
started in paused mode (e.g. `unitrace --start-paused`), wrapping code in
`oneAPI.@profile external=true` restricts collection to that code, using the
Instrumentation and Tracing Technology (ITT) API. Work submitted to the calling task's
stream is synchronized before resuming the profiler and after `code`, so that it is
attributed to the right side of the boundary; work on other streams or devices is not
waited for. When no profiler is attached, `code` still runs, and a warning is shown.

See the profiling page of the documentation for how to launch Julia under an external
profiler.
"""
macro profile(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    external = quote
        if $itt_collector()
            @info "This Julia session is being profiled by an external profiler; defaulting to `oneAPI.@profile external=true`." maxlog=1 _id=:oneapi_profile_external
            true
        else
            false
        end
    end
    explicit_external = false
    remaining = Expr[]
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) ||
            throw(ArgumentError("Invalid keyword argument to oneAPI.@profile: $kwarg"))
        key, value = kwarg.args
        if key === :external
            isa(value, Bool) || throw(ArgumentError("Invalid value for keyword argument `external`: got `$value`, expected literal boolean value"))
            external = value
            explicit_external = value
        elseif key in (:trace, :raw)
            push!(remaining, Expr(:kw, key, esc(value)))
        else
            throw(ArgumentError("Invalid keyword argument to oneAPI.@profile: $kwarg"))
        end
    end
    if explicit_external && !isempty(remaining)
        throw(ArgumentError("oneAPI.@profile external=true does not support other keyword arguments"))
    end

    quote
        profiled_code() = $(esc(code))
        if $external
            $profile_externally(profiled_code)
        else
            $profile_internally(profiled_code; $(remaining...))
        end
    end
end

"""
    oneAPI.@bprofile [time=1.0] [trace=false] [raw=false] code...

Benchmark the GPU execution of `code` by running it repeatedly for `time` seconds, and
report the aggregated results using the integrated profiler ([`oneAPI.@profile`](@ref)).
`code` runs once before profiling, to exclude compilation, and the task's stream is
synchronized after every iteration.

The `time` keyword argument is optional and defaults to `1.0` seconds; the remaining
keyword arguments are forwarded to `oneAPI.@profile`.
"""
macro bprofile(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    remaining = Expr[]
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) ||
            throw(ArgumentError("Invalid keyword argument to oneAPI.@bprofile: $kwarg"))
        key, value = kwarg.args
        key === :external &&
            throw(ArgumentError("The `external` keyword argument is not supported by oneAPI.@bprofile"))
        key in (:time, :trace, :raw) ||
            throw(ArgumentError("Invalid keyword argument to oneAPI.@bprofile: $kwarg"))
        push!(remaining, Expr(:kw, key, esc(value)))
    end

    quote
        profiled_code() = $(esc(code))
        $with_session() do
            $(Profiling.benchmark_and_profile)(profiled_code; $(remaining...))
        end
    end
end

# Profiling state is process-wide: an inner session pausing an ITT collector, or disabling
# PTI's views, would truncate an outer one.
const profiling = Threads.Atomic{Bool}(false)

function with_session(f)
    Threads.atomic_cas!(profiling, false, true) &&
        error("oneAPI.@profile does not support nested or concurrent profiling sessions")
    try
        f()
    finally
        profiling[] = false
    end
end

profile_internally(f; kwargs...) = with_session(() -> Profiling.profile_internally(f; kwargs...))

itt_collector() = IntelITT.ittapi_jll.is_available() && IntelITT.isactive()

# A profiler that initializes Level Zero before oneAPI.jl does (e.g. unitrace, which is
# preloaded) may pick up another copy of the loader or driver than the ones oneAPI.jl uses,
# and then traces that copy instead.
function check_level_zero_libraries()
    Sys.islinux() || return
    copies = Dict{String,Set{String}}()
    try
        for line in eachline("/proc/self/maps")
            path = last(split(line))
            m = match(r"/(libze_loader|libze_intel_gpu)\.so[^/]*$", path)
            m === nothing && continue
            push!(get!(Set{String}, copies, m[1]), path)
        end
    catch
        return
    end
    duplicates = filter(kv -> length(kv[2]) > 1, copies)
    isempty(duplicates) && return
    paths = join(("- " * path for (_, paths) in duplicates for path in sort!(collect(paths))), "\n")
    @warn """Multiple copies of the Level Zero loader or driver are loaded in this process, so the external profiler may not see the work oneAPI.jl submits:
             $paths
             See the profiling page of the oneAPI.jl documentation for how to launch the profiler.""" maxlog=1 _id=:oneapi_profile_libraries
end

function profile_externally(@nospecialize(f))
    with_session() do
        stream = global_stream(context(), device())
        synchronize(stream)

        # IntelITT's wrappers throw when the collector's function pointers are missing
        collector = itt_collector()
        if collector
            check_level_zero_libraries()
        else
            @warn """No ITT collector is attached to this process; oneAPI.@profile will not restrict profiling to the given code.
                     Run Julia under an external profiler, e.g., `unitrace --start-paused`.""" maxlog=1 _id=:oneapi_profile_collector
        end

        collector && IntelITT.resume()
        try
            f()
        finally
            try
                synchronize(stream)
            finally
                collector && IntelITT.pause()
            end
        end
    end
end
