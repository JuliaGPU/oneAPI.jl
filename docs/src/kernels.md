# Kernel Programming

For maximum performance or custom operations not covered by high-level array abstractions, you can write custom kernels in Julia that execute on the GPU.

## The `@oneapi` Macro

The `@oneapi` macro is used to launch a kernel on the device. It takes configuration arguments like the number of items (threads) and groups (blocks).

```julia
using oneAPI

function kernel(a, b)
    i = get_global_id()
    if i <= length(a)
        @inbounds a[i] += b[i]
    end
    return
end

a = oneArray(rand(Float32, 100))
b = oneArray(rand(Float32, 100))

# Launch configuration
items = 100
groups = 1

@oneapi items=items groups=groups kernel(a, b)
```

## KernelAbstractions.jl

For portable kernel programming, it is highly recommended to use [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). This allows you to write kernels that work on CPU, CUDA, ROCm, and oneAPI.

```julia
using KernelAbstractions, oneAPI

@kernel function my_kernel!(a, b)
    i = @index(Global, Linear)
    @inbounds a[i] += b[i]
end

# Get the backend
backend = get_backend(a)

# Instantiate the kernel
k = my_kernel!(backend)

# Launch with configuration
k(a, b; ndrange=length(a))
```

## Device Intrinsics

Inside a kernel, you can use various intrinsics to interact with the hardware:
- `get_global_id()`: Get the global thread ID.
- `get_local_id()`: Get the local thread ID within a workgroup.
- `get_group_id()`: Get the workgroup ID.
- `barrier()`: Synchronize threads within a workgroup.

These correspond to standard OpenCL/Level Zero intrinsics.

