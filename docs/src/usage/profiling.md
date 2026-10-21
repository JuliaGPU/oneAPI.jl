# Profiling

To find out where a GPU application spends its time, run it under an external profiler,
such as Intel's [unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace)
or VTune. These record the Level Zero API calls the application makes and the time each
kernel and memory copy takes to execute on the device.

For simple timing measurements, see [Benchmarking](@ref) in the Performance Guide.

## Restricting collection

Profiling a whole Julia session also captures compilation and other warm-up work. To only
collect a region of interest, start the profiler paused and wrap that region in
`oneAPI.@profile external=true`:

```julia
using oneAPI

a = oneArray(rand(Float32, 1024, 1024))
a * a                           # warm up, not profiled

oneAPI.@profile external=true begin
    for i in 1:10
        a * a
    end
end
```

```@docs
oneAPI.@profile
```

## Launching the profiler

oneAPI.jl uses the Level Zero loader and the Intel GPU driver from its own artifacts. An
external profiler that loads into the process before Julia starts may pick up another copy
of these libraries, e.g., one installed on the system, and then traces that copy instead
of the one oneAPI.jl uses. In that case the profile lacks the work oneAPI.jl submits, and
oneMKL operations may fail, while `oneAPI.@profile` warns about multiple loaded copies.

To avoid this, point the loader at oneAPI.jl's copies before launching the profiler. Run
the following in the shell you start the profiler from, using the same project as the
application. Setting these variables from within the profiled Julia process is too late.

```sh
eval "$(julia --project -e '
    using oneAPI
    L0 = oneAPI.oneL0
    loader_dir = dirname(L0.oneAPI_Level_Zero_Loader_jll.libze_loader)
    ld_path = join([loader_dir; split(get(ENV, "LD_LIBRARY_PATH", ""), ":"; keepempty=false)], ":")
    println("export LD_LIBRARY_PATH=", Base.shell_escape_posixly(ld_path))
    println("export ZE_ENABLE_ALT_DRIVERS=", Base.shell_escape_posixly(L0.NEO_jll.libze_intel_gpu))')"
```

This is not needed when oneAPI.jl is configured to use the system libraries (see
[Using System Libraries (Advanced)](@ref)).

### unitrace

unitrace is part of Intel's [PTI-GPU](https://github.com/intel/pti-gpu) tools. Start it
with `--start-paused` so that only the code in `oneAPI.@profile external=true` is
collected:

```sh
unitrace --start-paused --device-timing --chrome-kernel-logging \
    julia --project workload.jl
```

`--device-timing` prints a summary of kernel and memory copy times when the application
exits; the Chrome trace can be opened in [Perfetto](https://ui.perfetto.dev).

### VTune

VTune finds the ITT collector through the `INTEL_LIBITTNOTIFY64` environment variable,
which it sets for applications it launches. Start the collection paused:

```sh
vtune -collect gpu-offload -start-paused -- julia --project workload.jl
```

When attaching VTune to a running Julia process instead, `INTEL_LIBITTNOTIFY64` must have
been set to VTune's `libittnotify_collector.so` before Julia started.
