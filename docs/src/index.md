# oneAPI.jl

*Julia support for the oneAPI programming toolkit.*

oneAPI.jl provides support for working with the [oneAPI unified programming model](https://software.intel.com/en-us/oneapi). The package is currently verified to work with the implementation provided by the [Intel Compute Runtime](https://github.com/intel/compute-runtime), primarily on Linux.

## Writing Portable Code

While oneAPI.jl provides specific functionality for Intel GPUs, it is highly recommended to write **backend-agnostic code** whenever possible. This allows your code to run on various hardware backends (NVIDIA, AMD, Intel, Apple) without modification.

- **[GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl)**: Use high-level array abstractions that work across different GPU backends.
- **[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)**: Use this package for writing kernels that can be compiled for CPU, CUDA, ROCm, and oneAPI devices.

Direct use of `oneAPI`-specific macros (like `@oneapi`) and types (like `oneArray`) should be reserved for cases where you need specific optimizations or features not covered by the generic abstractions.

## Features

- **High-level Array Abstractions**: `oneArray` type fully implementing the `GPUArrays.jl` interface.
- **Kernel Programming**: Execute custom kernels written in Julia on Intel GPUs.
- **Level Zero Integration**: Low-level access to the Level Zero API via the `oneL0` submodule.
- **oneMKL Support**: Integration with Intel oneMKL for BLAS, LAPACK, and sparse operations.
- **SYCL Integration**: Interoperability with SYCL (on Linux).

## Requirements

- **Julia**: 1.10 or higher
- **OS**: Linux
- **Hardware**: Intel Gen9 graphics or newer (including Intel Arc A-Series)

