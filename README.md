# oneAPI.jl

*Julia support for the oneAPI programming toolkit.*

[![][doi-img]][doi-url] [![][buildkite-img]][buildkite-url] [![][codecov-img]][codecov-url]

[doi-img]: https://zenodo.org/badge/252466420.svg
[doi-url]: https://zenodo.org/badge/latestdoi/252466420

[buildkite-img]: https://badge.buildkite.com/00fff01fd4d6cdd905e61e2ce7ed0f7203ba227df9b575426c.svg
[buildkite-url]: https://buildkite.com/julialang/oneapi-dot-jl

[codecov-img]: https://codecov.io/gh/JuliaGPU/oneAPI.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/oneAPI.jl

oneAPI.jl provides support for working with the [oneAPI unified programming
model](https://software.intel.com/en-us/oneapi). The package is verified to work with the
(currently) only implementation of this interface [that is part of the Intel Compute
Runtime](https://github.com/intel/compute-runtime), only available on Linux.
Windows support is experimental.


## Status

**oneAPI.jl is looking for contributors and/or a maintainer. Reach out if you can help!**

The current version of oneAPI.jl supports most of the oneAPI Level Zero interface, has
good kernel programming capabilties, and as a demonstration of that it fully implements
the GPUArrays.jl array interfaces. This results in a full-featured GPU array type.

However, the package has not been extensively tested, and performance issues might be
present. The integration with vendor libraries like oneMKL or oneDNN is still in
development, and as result certain array operations may be unavailable or slow.


## Quick start

You need to use Julia 1.8 or higher, and it is strongly advised to use [the official
binaries](https://julialang.org/downloads/). For now, only Linux is supported.
On Windows, you need to use the second generation Windows Subsystem for Linux (WSL2).
**If you're using Intel Arc GPUs (A580, A750, A770, etc), you need to use at least
Linux 6.2.** For other hardware, any recent Linux distribution should work.

Once you have installed Julia, proceed by entering the package manager REPL mode by pressing
`]` and adding theoneAPI package:

```
pkg> add oneAPI
```

This installation will take a couple of minutes to download necessary binaries, such as the
oneAPI loader, several SPIR-V tools, etc. For now, the oneAPI.jl package also depends on
[the Intel implementation](https://github.com/intel/compute-runtime) of the oneAPI spec.
That means you need compatible hardware; refer to the Intel documentation for more details.

Once you have oneAPI.jl installed, perform a smoke test by calling the `versioninfo()` function:

```julia
julia> using oneAPI

julia> oneAPI.versioninfo()
Binary dependencies:
- NEO: 24.26.30049+0
- libigc: 1.0.17193+0
- gmmlib: 22.3.20+0
- SPIRV_LLVM_Translator: 20.1.0+1
- SPIRV_Tools: 2025.1.0+1

Toolchain:
- Julia: 1.11.5
- LLVM: 16.0.6

1 driver:
- 00000000-0000-0000-173d-d94201036013 (v1.3.24595, API v1.3.0)

2 devices:
- Intel(R) Graphics [0x56a0]
- Intel(R) HD Graphics P630 [0x591d]
```

If you have multiple compatible drivers or devices, use the `driver!` and `device!`
functions to configure which one to use in the current task:

```julia
julia> devices()
ZeDevice iterator for 2 devices:
1. Intel(R) Graphics [0x56a0]
2. Intel(R) HD Graphics P630 [0x591d]

julia> device()
ZeDevice(GPU, vendor 0x8086, device 0x56a0): Intel(R) Graphics [0x56a0]

julia> device!(2)
ZeDevice(GPU, vendor 0x8086, device 0x591d): Intel(R) HD Graphics P630 [0x591d]
```

To ensure other functionality works as expected, you can run the test suite from the package
manager REPL mode. Note that this will pull and run the test suite for
[GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl), which takes quite some time:

```
pkg> test oneAPI
...
Testing finished in 16 minutes, 27 seconds, 506 milliseconds

Test Summary: | Pass  Total  Time
  Overall     | 4945   4945
    SUCCESS
     Testing oneAPI tests passed
```


## Usage

The functionality of oneAPI.jl is organized as follows:

- low-level wrappers for the Level Zero library
- kernel programming capabilities
- abstractions for high-level array programming

The level zero wrappers are available in the `oneL0` submodule, and expose all flexibility
of the underlying APIs with user-friendly wrappers:

```julia
julia> using oneAPI, oneAPI.oneL0

julia> drv = first(drivers());

julia> ctx = ZeContext(drv);

julia> dev = first(devices(drv))
ZeDevice(GPU, vendor 0x8086, device 0x1912): Intel(R) Gen9

julia> compute_properties(dev)
(maxTotalGroupSize = 256, maxGroupSizeX = 256, maxGroupSizeY = 256, maxGroupSizeZ = 256, maxGroupCountX = 4294967295, maxGroupCountY = 4294967295, maxGroupCountZ = 4294967295, maxSharedLocalMemory = 65536, subGroupSizes = (8, 16, 32))

julia> queue = ZeCommandQueue(ctx, dev);

julia> execute!(queue) do list
         append_barrier!(list)
       end
```

Built on top of that, are kernel programming capabilities for executing Julia code on oneAPI
accelerators. For now, we reuse OpenCL intrinsics, and compile to SPIR-V using [Khronos'
translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator):

```julia
julia> function kernel()
         barrier(0)
         return
       end

julia> @oneapi items=1 kernel()
```

Code reflection macros are available to see the generated code:

```julia
julia> @device_code_llvm @oneapi items=1 kernel()
```

```llvm
;  @ REPL[18]:1 within `kernel'
define dso_local spir_kernel void @_Z17julia_kernel_3053() local_unnamed_addr {
top:
;  @ REPL[18]:2 within `kernel'
; ┌ @ oneAPI.jl/src/device/opencl/synchronization.jl:9 within `barrier' @ oneAPI.jl/src/device/opencl/synchronization.jl:9
; │┌ @ oneAPI.jl/src/device/opencl/utils.jl:34 within `macro expansion'
    call void @_Z7barrierj(i32 0)
; └└
;  @ REPL[18]:3 within `kernel'
  ret void
}
```

```julia
julia> @device_code_spirv @oneapi items=1 kernel()
```

```spirv
; SPIR-V
; Version: 1.0
; Generator: Khronos LLVM/SPIR-V Translator; 14
; Bound: 9
; Schema: 0
               OpCapability Addresses
               OpCapability Kernel
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %4 "_Z17julia_kernel_3067"
               OpSource OpenCL_C 200000
               OpName %top "top"
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
     %uint_0 = OpConstant %uint 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
          %4 = OpFunction %void None %3
        %top = OpLabel
               OpControlBarrier %uint_2 %uint_2 %uint_0
               OpReturn
               OpFunctionEnd

```

Finally, the `oneArray` type makes it possible to use your oneAPI accelerator without the
need to write custom kernels, thanks to Julia's high-level array abstractions:

```julia
julia> a = oneArray(rand(Float32, 2,2))
2×2 oneArray{Float32,2}:
 0.592979  0.996154
 0.874364  0.232854

julia> a .+ 1
2×2 oneArray{Float32,2}:
 1.59298  1.99615
 1.87436  1.23285
```

### `Float64` support

Not all oneAPI GPUs support Float64 datatypes. You can test if your GPU does using
the following code:

```julia
julia> using oneAPI
julia> oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
false
```

If your GPU doesn't, executing code that relies on Float64 values will result in an error:

```julia
julia> oneArray([1.]) .+ 1
┌ Error: Module compilation failed:
│
│ error: Double type is not supported on this platform.
```



## Development

To work on oneAPI.jl, you just need to `dev` the package. In addition, you may need to
**build the binary support library** that's used to interface with oneMKL and other C++
vendor libraries. This library is normally provided by the oneAPI_Support_jll.jl package,
however, we only guarantee to update this package when releasing oneAPI.jl. You can build
this library yourself by simply executing `deps/build_local.jl`.

To facilitate development, there are other things you may want to configure:

### Enabling the oneAPI validation layer

The oneAPI Level Zero libraries feature a so-called validation layer, which
validates the arguments to API calls. This can be useful to spot potential
isssues, and can be enabled by setting the following environment variables:

- `ZE_ENABLE_VALIDATION_LAYER=1`
- `ZE_ENABLE_PARAMETER_VALIDATION=1`
- `EnableDebugBreak=0` (this is needed to work around intel/compute-runtime#639)

### Using a debug toolchain

If you're experiencing an issue with the underlying toolchain (NEO, IGC, etc), you may
want to use a debug build of these components, which also perform additional
validation. This can be done simply by calling `oneAPI.set_debug!(true)` and restarting
your Julia session. This sets a preference used by the respective JLL packages.

### Using a local toolchain

To further debug the toolchain, you may need a custom build and point oneAPI.jl towards it.
This can also be done using preferences, overriding the paths to resources provided by the
various JLLs that oneAPI.jl uses. A helpful script to automate this is provided in the
`res` folder of this repository:

```
$ julia res/local.jl

Trying to find local IGC...
- found libigc at /usr/local/lib/libigc.so
- found libiga64 at /usr/local/lib/libiga64.so
- found libigdfcl at /usr/local/lib/libigdfcl.so
- found libopencl-clang at /usr/local/lib/libopencl-clang.so.11

Trying to find local gmmlib...
- found libigdgmm at /usr/local/lib/libigdgmm.so

Trying to find local NEO...
- found libze_intel_gpu.so.1 at /usr/local/lib/libze_intel_gpu.so.1
- found libigdrcl at /usr/local/lib/intel-opencl/libigdrcl.so

Trying to find local oneAPI loader...
- found libze_loader at /lib/x86_64-linux-gnu/libze_loader.so
- found libze_validation_layer at /lib/x86_64-linux-gnu/libze_validation_layer.so

Writing preferences...
```

The discovered paths will be written to a global file with preferences, typically
`$HOME/.julia/environments/vX.Y/LocalPreferences.toml` (where `vX.Y` refers to the Julia
version you are using). You can modify this file, or remove it when you want to revert to
default set of binaries.
