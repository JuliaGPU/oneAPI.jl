# Installation

## Requirements

oneAPI.jl requires:
- **Julia**: 1.10 or higher
- **OS**: Linux (recommended) or Windows (experimental via WSL2)
- **Hardware**: Intel Gen9 graphics or newer. For Intel Arc GPUs (A580, A750, A770, etc), **Linux 6.2+** is required.

## Installing oneAPI.jl

You can install oneAPI.jl using the Julia package manager:

```julia
pkg> add oneAPI
```

This will automatically download the necessary binary dependencies, including:
- `oneAPI loader`
- `SPIR-V tools`
- `Intel Compute Runtime` (if compatible hardware is found)

## Verifying Installation

After installation, you can verify that oneAPI.jl is working correctly and detecting your hardware:

```julia
julia> using oneAPI
julia> oneAPI.versioninfo()
```

The output should list the binary dependencies, toolchain versions, available drivers, and devices.

## Troubleshooting Drivers

If no drivers or devices are detected, ensure that you have the correct Intel graphics drivers installed for your system.
- On Linux, check if `libze_intel_gpu.so` or similar libraries are available.
- On Windows (WSL2), ensure you have the latest Intel graphics drivers installed on the host Windows system and that WSL2 is configured to access the GPU.

You can explicitly select drivers and devices if multiple are available:

```julia
julia> drivers()
julia> devices()
julia> device!(1) # Select the first available device
```

## Using System Libraries (Advanced)

!!! warning
    Using system libraries instead of the provided artifacts is **not recommended** for most users. Only use this approach if you have specialized requirements or custom Intel binaries.

By default, oneAPI.jl uses pre-built binary artifacts (JLLs) for the Intel Compute Runtime, oneAPI loader, and related libraries. However, you may need to use system-installed libraries in certain situations:

- Custom or newer Intel graphics drivers
- Specialized hardware configurations
- Development or debugging of the runtime stack
- Systems where the artifacts are incompatible

### Configuration Script

oneAPI.jl provides a helper script to discover and configure system libraries. From the Julia REPL:

```julia
julia> include(joinpath(pkgdir(oneAPI), "res", "local.jl"))
```

This script will:
1. Search for Intel libraries on your system:
   - Intel Graphics Compiler (IGC): `libigc`, `libiga64`, `libigdfcl`, `libopencl-clang`
   - Graphics Memory Management Library: `libigdgmm`
   - Intel Compute Runtime (NEO): `libze_intel_gpu`, `libigdrcl`
   - oneAPI Level Zero Loader: `libze_loader`, `libze_validation_layer`

2. Generate preferences in `LocalPreferences.toml` that override the artifact paths

### Manual Configuration

You can also manually set preferences to use specific library paths. Create or edit `LocalPreferences.toml` in your project or global environment:

```toml
[NEO_jll]
libze_intel_gpu_path = "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1"
libigdrcl_path = "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"

[libigc_jll]
libigc_path = "/usr/lib/x86_64-linux-gnu/libigc.so"
libigdfcl_path = "/usr/lib/x86_64-linux-gnu/libigdfcl.so"

[gmmlib_jll]
libigdgmm_path = "/usr/lib/x86_64-linux-gnu/libigdgmm.so"

[oneAPI_Level_Zero_Loader_jll]
libze_loader_path = "/usr/lib/x86_64-linux-gnu/libze_loader.so"
```

### Reverting to Artifacts

To revert to the default artifact binaries, simply delete the oneAPI-related entries from `LocalPreferences.toml` (or delete the entire file if it only contains these preferences).

### Common Locations

System libraries are typically installed in:

**Ubuntu/Debian:**
- `/usr/lib/x86_64-linux-gnu/`
- `/usr/lib/x86_64-linux-gnu/intel-opencl/`

**Fedora/RHEL:**
- `/usr/lib64/`
- `/usr/lib64/intel-opencl/`

**Custom Intel oneAPI installation:**
- `/opt/intel/oneapi/compiler/latest/linux/lib/`
- `/opt/intel/oneapi/compiler/latest/linux/lib/x64/`

### Verifying System Library Configuration

After configuring system libraries, restart Julia and verify the configuration:

```julia
julia> using oneAPI
julia> oneAPI.versioninfo()
```

Check that the reported library paths match your system libraries. If issues arise, examine the `LocalPreferences.toml` file and ensure all paths are correct and the libraries are compatible with each other.

