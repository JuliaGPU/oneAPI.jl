# oneAPI.jl

*Julia support for the oneAPI programming toolkit.*

| **Build Status**                    | **Coverage**                    |
|:-----------------------------------:|:-------------------------------:|
| [![][buildkite-img]][buildkite-url] | [![][codecov-img]][codecov-url] |

[buildkite-img]: https://badge.buildkite.com/00fff01fd4d6cdd905e61e2ce7ed0f7203ba227df9b575426c.svg
[buildkite-url]: https://buildkite.com/julialang/oneapi-dot-jl

[codecov-img]: https://codecov.io/gh/JuliaGPU/oneAPI.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/oneAPI.jl

oneAPI.jl provides support for working with the [oneAPI unified programming
model](https://software.intel.com/en-us/oneapi). The package is verified to work with the
(currently) only implementation of this interface [that is part of the Intel Compute
Runtime](https://github.com/intel/compute-runtime), only available on Linux.

This package is still under significant development, so expect bugs and missing features.


## Installation

You need to use Julia 1.6 or higher, and it is strongly advised to use [the official
binaries](https://julialang.org/downloads/). For now, only Linux is supported.

Once you have installed Julia, proceed by entering the package manager REPL mode by pressing
`]` and adding theoneAPI package:

```
pkg> add oneAPI
```

This installation will take a couple of minutes to download necessary binaries, such as the
oneAPI loader, several SPIR-V tools, etc. For now, the oneAPI.jl package also depends on
[the Intel implementation](https://github.com/intel/compute-runtime) of the oneAPI spec.
That means you need compatible hardware; refer to the Intel documentation for more details.

Once you have oneAPI.jl installed, you can perform a smoke-test using the low-level wrappers
for the Level Zero library:

```julia
julia> using oneAPI

julia> using oneAPI.oneL0

julia> drv = first(drivers());

julia> dev = first(devices(drv))
ZeDevice(GPU, vendor 0x8086, device 0x1912): Intel(R) Gen9
```

To ensure other functionality works as expected, you can run the test suite from the package
manager REPL mode, note that it will pull and run the test suite for [GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl)
which takes quite some time:

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
         barrier()
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


## Status

The current version of oneAPI.jl supports most of oneAPI Level Zero interface, has good
kernel programming capabilties, and as a demonstration of that it fully implements the
GPUArrays.jl array interfaces. This results in a full-featured GPU array type.

However, the package has not been extensively tested, and performance issues might be
present. There is no integration with vendor libraries like oneMKL or oneDNN, and as a
result certain operations (like matrix multiplication) will be unavailable or slow.


## Using a local toolchain

For debugging issues with the underlying toolchain (NEO, IGC, etc), you may want the
package to use your local installation of these components instead of downloading the
prebuilt Julia binaries from Yggdrasil. This can be done using Preferences.jl, overriding
the paths to resources provided by the various JLLs that oneAPI.jl uses. A helpful script
to automate this is provided in the `res` folder of this repository:

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
