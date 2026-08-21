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


## Exceptions and Dynamic Allocation

Kernels can throw. An exception on the device aborts the work-item that threw it and is
reported on the host as a `KernelException` at the next synchronization — `synchronize()`,
`oneAPI.@sync`, or copying data back with `Array` — after the reason was printed by the
device:

```julia
julia> function kernel(a)
           a[2] = 1f0     # bounds-checked
           return
       end;

julia> @oneapi kernel(oneArray(Float32[0]));

julia> synchronize()
ERROR: Out-of-bounds array access.
ERROR: KernelException: exception thrown during kernel execution on device Intel(R) Data Center GPU Max 1550
```

Exception objects that survive optimization, and any other Julia object that has to be
heap-allocated inside a kernel (for example a `Ref` passed to a `@noinline` function), are
allocated from a small per-work-item heap in private memory; objects never outlive the
work-item that created them and are never freed. The heap is limited to
`oneAPI.PRIVATE_HEAP_SIZE` bytes per work-item, and exhausting it is reported as a
`KernelException` as well, rather than failing silently. Code on a hot path should not
allocate.
