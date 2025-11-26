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

## Debugging

### Validation Layer

Enable the Level Zero validation layer to catch API misuse:

```bash
export ZE_ENABLE_VALIDATION_LAYER=1
export ZE_ENABLE_PARAMETER_VALIDATION=1
```

### Debug Mode

Enable debug mode in oneAPI.jl to use debug builds of underlying toolchains (if available):

```julia
oneAPI.set_debug!(true)
```

