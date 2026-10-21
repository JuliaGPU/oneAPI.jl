# Profiler control

"""
    oneAPI.@profile external=true code...

Profile the GPU execution of `code` with an external profiler, such as Intel's
[unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) or VTune.

Collection is restricted to `code` using the Instrumentation and Tracing Technology (ITT)
API: the profiler is resumed before `code` runs and paused afterwards, so it should be
started in paused mode (e.g. `unitrace --start-paused`). Work submitted to the calling
task's stream is synchronized before resuming and after `code`, so that it is attributed
to the right side of the boundary; work on other streams or devices is not waited for.

When no profiler is attached, `code` still runs, and a warning is shown.

See the profiling page of the documentation for how to launch Julia under an external
profiler.

!!! note
    Integrated profiling, where oneAPI.jl collects and displays the results itself, is
    not supported yet, so `external=true` is required.
"""
macro profile(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    external = false
    for kwarg in kwargs
        if Meta.isexpr(kwarg, :(=)) && kwarg.args[1] === :external
            value = kwarg.args[2]
            isa(value, Bool) || throw(ArgumentError("Invalid value for keyword argument `external`: got `$value`, expected literal boolean value"))
            external = value
        else
            throw(ArgumentError("Invalid keyword argument to oneAPI.@profile: $kwarg"))
        end
    end
    external || throw(ArgumentError("oneAPI.@profile does not support integrated profiling yet; use `oneAPI.@profile external=true` under an external profiler"))

    quote
        profiled_code() = $(esc(code))
        $profile_externally(profiled_code)
    end
end

# ITT collection control is process-wide, so an inner session pausing the collector would
# truncate an outer one.
const profiling = Threads.Atomic{Bool}(false)

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
    Threads.atomic_cas!(profiling, false, true) &&
        error("oneAPI.@profile does not support nested or concurrent profiling sessions")
    try
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
    finally
        profiling[] = false
    end
end
