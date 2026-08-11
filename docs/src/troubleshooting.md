# Troubleshooting

## Common Issues

### No devices detected

**Symptom**: `oneAPI.devices()` returns an empty list.

**Solution**:
1. Ensure you are running on Linux (recommended) or WSL2.
2. Check if the Intel Compute Runtime is installed and accessible.
3. Verify your user has permissions to access the GPU render device (usually `render` group).
4. Run `oneAPI.versioninfo()` to see detailed diagnostic information.

### "Double type is not supported"

**Symptom**: Kernel compilation fails with an error about `Float64` or `Double` support.

**Solution**:
Some Intel GPUs (especially integrated graphics) lack native hardware support for 64-bit floating point operations.
- Use `Float32` instead of `Float64`.
- Check support with:
  ```julia
  using oneAPI.oneL0
  oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 != 0
  ```

### "Out of memory" errors

**Symptom**: Memory allocation fails.

**Solution**:
- Trigger garbage collection: `GC.gc()`.
- Manually free unused arrays: `oneAPI.unsafe_free!(array)`.
- Check if you are exceeding the device's memory capacity.
- When several processes share one GPU, give each a budget with `ONEAPI_MEMORY_LIMIT`; see
  [Memory Management](memory.md).

### Silently wrong results, or every submission failing

**Symptom**: on a system running Intel's long-term-servicing (LTS) Compute Runtime — such
as Aurora — reductions over transposed or otherwise strided arrays return wrong values
without raising an error, kernels drop their last work-items, or every submission starts
failing with `ZE_RESULT_ERROR_UNKNOWN` after a garbage collection.

**Solution**:
Enable the LTS workarounds and restart Julia:

```bash
export ONEAPI_LTS=1
```

See [Intel LTS Driver Stack](lts.md) for the individual defects, the additional
`ONEAPI_SYNC_EACH_SUBMISSION=1` switch for oversubscribed tiles, and the performance
trade-offs involved. Do not enable these on a rolling-release driver, which does not need
them.

### "InvalidIRError" for a BFloat16 kernel

**Symptom**: a kernel using `BFloat16` fails to compile even though the device reports
BFloat16 support.

**Solution**:
The LTS SPIR-V stack cannot translate the LLVM `bfloat` type in generic kernels; the
device-level check (`oneAPI._device_supports_bfloat16()`) reports hardware capability and
does not capture this. Use `Float16` or `Float32` on that stack — see
[Intel LTS Driver Stack](lts.md).

## Debugging

### Validation Layer

Enable the Level Zero validation layer to catch API misuse:

```bash
export ZE_ENABLE_VALIDATION_LAYER=1
export ZE_ENABLE_PARAMETER_VALIDATION=1
export EnableDebugBreak=0  # works around intel/compute-runtime#639
```

### Debug Mode

Enable debug mode in oneAPI.jl to use debug builds of underlying toolchains (if available):

```julia
oneAPI.set_debug!(true)
```


### Scratch hedge

Before the first submission of a kernel whose register spill exceeds the stream's
high-water mark, oneAPI.jl drains the stream and runs finalizers: the driver performs its
scratch-buffer allocation at that moment, and (at least through NEO 26.x) aborts the
process instead of erroring when it fails. Making the allocation happen at a clean moment
removes that failure mode for workloads under memory pressure. The drain costs one
synchronization per (stream, spill tier) — once per workload in practice. Disable it with:

```bash
export ONEAPI_SCRATCH_HEDGE=0
```
