# Intel LTS driver stack

Intel ships the Compute Runtime in two lines: frequent *rolling* releases, and a
long-term-servicing (LTS) branch that large deployments stay on for years — for example
the NEO/IGC LTS stack on [Aurora](https://www.alcf.anl.gov/aurora). oneAPI.jl targets the
rolling stack by default.

The LTS branch predates a number of driver and compiler fixes that landed in rolling. Left
alone, some of those defects are not merely inconvenient: they silently corrupt results, or
get the Level Zero context banned so that every later submission fails. oneAPI.jl therefore
carries a set of workarounds, kept behind a single opt-in switch so that the default
(rolling) path is completely unaffected.

## Enabling

Set the environment variable before loading oneAPI.jl:

```bash
export ONEAPI_LTS=1
```

`1`, `true`, `yes` and `on` enable it; `0`, `false`, `no` and `off` disable it. An
unrecognized value warns and falls back to the default (disabled), rather than silently
reading as off.

You can check which mode is active from Julia:

```julia
julia> using oneAPI

julia> oneAPI.oneL0.LTS[]
true
```

!!! note
    This flag does not install or select an LTS driver. oneAPI.jl ships pinned rolling
    NEO artifacts; on an LTS system you point the package at the system libraries instead
    (see the "Using System Libraries" section of [Installation](installation.md)).
    `ONEAPI_LTS=1` tells oneAPI.jl to *compile and behave* for that driver.

## What changes

### SPIR-V code generation

Julia kernels are compiled to SPIR-V. By default oneAPI.jl uses [LLVM's SPIR-V
back-end](https://llvm.org/docs/SPIRVUsage.html) (`SPIRV_LLVM_Backend_jll`). The LTS
NEO/IGC runtime does not accept that output, so with `ONEAPI_LTS=1` the package switches to
the [Khronos SPIR-V translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
(`SPIRV_LLVM_Translator_jll`).

Both tools are dependencies of oneAPI.jl and are resolved lazily by GPUCompiler, so the
choice costs nothing on either path and needs no reconfiguration of your environment.
`oneAPI.versioninfo()` reports both, and `@device_code_spirv` will show which one produced a
given module in its `Generator:` line.

### BFloat16 is unavailable

The LTS SPIR-V stack cannot translate the LLVM `bfloat` type in generic kernels: a kernel
that merely keeps a `BFloat16` value (`clamp!`, say) fails with an `InvalidIRError`, and
declaring the `SPV_KHR_bfloat16` extension crashes the LTS runtime outright. On LTS the
compiler is therefore configured with `supports_bfloat16 = false`, regardless of what the
device reports — `oneAPI._device_supports_bfloat16()` is a *hardware* capability check and
does not capture this limitation.

Consequently `BFloat16` is dropped from the element types exercised by the test suite, and
`examples/bfloat16.jl` exits early with a message. Other floating-point types are unaffected.

### Reductions over strided inputs

The LTS IGC miscompiles non-coalesced (strided) global reads inside the reduction kernel,
silently producing wrong results — no error, just bad numbers. Reading an array along a
non-contiguous axis is enough to trigger it, so `sum(transpose(x))`, `a == transpose(b)` and
`ishermitian(x)` are all affected. Elementwise copies are not.

Two mitigations apply on LTS:

- An input that is not densely laid out (a transposed, permuted or otherwise strided view,
  or a broadcast containing one) is materialized into a dense `oneArray` before the
  reduction runs. This costs an extra allocation and copy.
- A reduction that keeps the contiguous leading dimension (`sum(A; dims=2)` and friends) is
  routed to a coalesced kernel that assigns one work-item per output slice, so neighbouring
  lanes read neighbouring memory. Reductions with few output slices get less parallelism
  than the default workgroup-per-slice kernel, but stay correct.

Reductions that include dimension 1 (`dims=(1,3)`, or a full reduction) keep a contiguous
innermost axis and use the normal kernel.

### Frees are synchronized against in-flight work

NEO LTS advertises `ZE_extension_memory_free_policies` but does not honor its
`BLOCKING_FREE` policy: it unmaps an allocation immediately, even with work in flight that
references it. A garbage-collected free of a dead array whose last kernel has not retired
then faults on the GPU, which bans the kernel context — after which *every* later submission
fails with `ZE_RESULT_ERROR_UNKNOWN`.

On LTS, oneAPI.jl keeps a registry of the streams in use — each task's immediate command
list plus its companion command queue for oneMKL work — and drains those that could
reference a buffer before freeing it. Lists and queues are likewise drained before being
destroyed; one still busy after 10 s is deliberately leaked, since destroying it would
trigger the very fault the drain prevents. The visible cost is that a GC-driven free can
block until outstanding work completes.

### Optional: synchronize after every submission

Under heavy multi-process oversubscription of a single tile, a whole-queue
`zeCommandQueueSynchronize` on the LTS stack does not reliably retire the tail of an
earlier, separately submitted command list. The result is a silent *dropped tail*: the last
work-items of a kernel, or the last elements of a copy, never land.

Synchronizing after every submission eliminates it, at roughly a 3× throughput cost. It is
off by default and enabled with the setting below. With the current immediate-command-list
submission path it host-synchronizes the stream after every append; the dropped-tail
failure was only ever observed on the earlier queue-submission path, so this workaround may
no longer be needed — it is kept until that is re-established under oversubscription.

```bash
export ONEAPI_SYNC_EACH_SUBMISSION=1
```

It can also be controlled at runtime:

```julia
oneL0.sync_each_submission()            # query
oneL0.sync_each_submission!(true)       # set, returns the previous value

oneL0.sync_each_submission(false) do    # scoped, restores afterwards
    # ...
end
```

!!! warning
    A *submit-then-signal* pattern — work submitted with a wait event that is only signaled
    after submission returns — deadlocks with this enabled, because `execute!` blocks in the
    synchronize before the gating event can be signaled. No high-level oneAPI.jl code path
    submits event-gated work, but hand-written Level Zero code may; wrap those regions in
    `oneL0.sync_each_submission(false) do ... end`.

This workaround is independent of `ONEAPI_LTS` and can be enabled on its own.

## Caveats

- **Precompilation.** The precompilation workload warms up whichever SPIR-V tool
  `ONEAPI_LTS` selects *at precompile time*, and Julia does not invalidate the cache when an
  environment variable changes. After flipping the variable, run `Pkg.precompile()` to
  refresh it. This only affects first-call latency, never correctness.
- **No effect on the rolling stack.** Every workaround above is gated on the switch, so with
  `ONEAPI_LTS` unset oneAPI.jl behaves exactly as it does upstream. The workarounds trade
  performance for correctness and should not be enabled on a driver that does not need them.

## Running the test suite

The test suite reads the same switch and skips the cases the LTS stack cannot support (the
`BFloat16` element type, the BFloat16 example). On a multi-tile node, pinning each test
worker to its own tile avoids the oversubscription that provokes the dropped-tail
corruption:

```bash
ONEAPI_LTS=1 ONEAPI_TEST_SPREAD_GPUS=1 julia --project=. -e 'import Pkg; Pkg.test()'
```
