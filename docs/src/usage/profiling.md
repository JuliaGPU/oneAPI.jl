# Profiling

To find out where a GPU application spends its time, use oneAPI.jl's integrated profiler,
or an external profiler such as Intel's
[unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) or VTune.

For simple timing measurements, see [Benchmarking](@ref) in the Performance Guide.

## Integrated profiler

`oneAPI.@profile` runs code and reports the Level Zero API calls the host made, and the
kernels, memory copies and fills that executed on the GPU. It uses Intel's Profiling Tools
Interfaces for GPU (PTI), so it also covers work submitted by libraries such as oneMKL.
Run the code once beforehand, so that compilation does not end up in the profile:

```julia
julia> a = oneArray(rand(Float32, 1024, 1024)); b = similar(a);

julia> c = a * a; copyto!(b, c);   # warm up

julia> oneAPI.@profile begin
           c = a * a
           copyto!(b, c)
       end
Profiled over 5.1 ms.

Host-side activity: 33 Level Zero API calls taking 1.22 ms (24.02% of wall-clock)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬───────────────────────────────────┐
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                              │
├──────────┼────────────┼───────┼──────────────────────────────────────┼───────────────────────────────────┤
│   13.38% │  681.99 µs │     4 │  170.5 µs ± 340.57 (  0.11 ‥ 681.35) │ zeCommandQueueSynchronize         │
│    5.61% │  286.14 µs │     1 │                                      │ zeContextMakeMemoryResident       │
│    2.50% │  127.69 µs │     1 │                                      │ zeCommandQueueExecuteCommandLists │
│    0.85% │   43.27 µs │     1 │                                      │ zeMemAllocDevice                  │
│    0.81% │   41.14 µs │     1 │                                      │ zeCommandListAppendMemoryCopy     │
│    0.40% │   20.49 µs │     1 │                                      │ zeCommandListReset                │
│    0.27% │   13.74 µs │     1 │                                      │ zeCommandListAppendLaunchKernel   │
│    0.08% │    3.88 µs │     1 │                                      │ zeDeviceGetProperties             │
│    0.06% │    2.91 µs │    15 │  194.0 ns ± 410.03 (  30.0 ‥ 1580.0) │ zeKernelSetArgumentValue          │
│    0.03% │    1.72 µs │     1 │                                      │ zeEventHostReset                  │
│    0.01% │   740.0 ns │     1 │                                      │ zeCommandListClose                │
│    0.01% │   420.0 ns │     3 │  140.0 ns ± 140.0  (  40.0 ‥ 300.0)  │ zeMemGetAllocProperties           │
│    0.00% │   200.0 ns │     1 │                                      │ zeKernelSetGroupSize              │
│    0.00% │   120.0 ns │     1 │                                      │ zeFenceReset                      │
└──────────┴────────────┴───────┴──────────────────────────────────────┴───────────────────────────────────┘

Device-side activity: GPU was busy for 551.25 µs (10.81% of wall-clock)
┌──────────┬────────────┬───────┬────────────────────────────────────┐
│ Time (%) │ Total time │ Calls │ Name                               │
├──────────┼────────────┼───────┼────────────────────────────────────┤
│    9.80% │  499.48 µs │     1 │ sgemm_nocopy_nn_16x16_8x4          │
│    1.02% │   51.77 µs │     1 │ zeCommandListAppendMemoryCopy(D2D) │
└──────────┴────────────┴───────┴────────────────────────────────────┘
```

With `trace=true`, the profiler lists every API call and GPU operation in chronological
order instead. Their ID column links each GPU operation to the API call that submitted it.
To profile code that runs quickly, `oneAPI.@bprofile` runs it repeatedly for a second (or
`time` seconds) and reports the aggregated results.

The integrated profiler is only available on x86_64 Linux.

```@docs
oneAPI.@profile
oneAPI.@bprofile
```

## External profilers

External profilers offer more detail, such as hardware metrics. Profiling a whole Julia
session also captures compilation and other warm-up work, so to only collect a region of
interest, start the profiler paused and wrap that region in `oneAPI.@profile external=true`:

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

When Julia runs under an external profiler, `oneAPI.@profile` defaults to `external=true`.

### Launching the profiler

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

#### unitrace

unitrace is part of Intel's [PTI-GPU](https://github.com/intel/pti-gpu) tools. Start it
with `--start-paused` so that only the code in `oneAPI.@profile external=true` is
collected:

```sh
unitrace --start-paused --device-timing --chrome-kernel-logging \
    julia --project workload.jl
```

`--device-timing` prints a summary of kernel and memory copy times when the application
exits; the Chrome trace can be opened in [Perfetto](https://ui.perfetto.dev).

#### VTune

VTune finds the ITT collector through the `INTEL_LIBITTNOTIFY64` environment variable,
which it sets for applications it launches. Start the collection paused:

```sh
vtune -collect gpu-offload -start-paused -- julia --project workload.jl
```

When attaching VTune to a running Julia process instead, `INTEL_LIBITTNOTIFY64` must have
been set to VTune's `libittnotify_collector.so` before Julia started.
