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
- `barrier(flags)`: Synchronize threads within a workgroup.

These are provided by [SPIRVIntrinsics.jl](https://github.com/JuliaGPU/SPIRVIntrinsics.jl)
and correspond to the standard OpenCL built-in functions. Note that the indices they return
are 1-based, so they can be used to index Julia arrays directly. See
[Device Intrinsics](device.md) for the full list.


## Dynamic Memory Allocation

Kernels can allocate Julia objects, such as a `Ref` passed to a `@noinline` function or a
boxed value in an `Any` field. Allocations that survive optimization use a 1 KiB heap
private to each work-item. Each allocation is rounded up to 16 bytes, and memory is only
reclaimed when the work-item exits. Allocated objects must not be shared with other
work-items or retained across kernel launches.

When the heap is exhausted, the work-item prints an error and exits without completing
its work. This does not raise a host-side exception, and kernel output may be incomplete:

```
ERROR: Out of dynamic GPU memory (trying to allocate 4 bytes)
```

Kernels without remaining allocations do not reserve an arena. The device compiler may
optimize away some heap storage, but allocations can increase private-memory use and
reduce performance. Avoid repeated allocations in loops: even short-lived objects consume
heap space until the work-item exits.
