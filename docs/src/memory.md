# Memory Management

Efficient memory management is crucial for GPU programming. oneAPI.jl provides tools to manage device memory allocation and data transfer.

## Unified Shared Memory (USM)

oneAPI uses Unified Shared Memory, which allows for pointers that can be accessible from both the host and the device, or specific to one.

- **Device Memory**: Accessible only by the device. Fastest access for kernels.
- **Host Memory**: Accessible by the host and device.
- **Shared Memory**: Automatically migrated between host and device.

`oneArray` typically uses device memory for performance.

## Allocation

You can perform low-level memory allocation using the `oneL0` submodule if needed, though `oneArray` handles this automatically.

```julia
using oneAPI.oneL0

# Allocate device memory
ptr = oneL0.zeMemAllocDevice(context(), device(), 1024, 1)

# Free memory
oneL0.zeMemFree(context(), ptr)
```

## Garbage Collection

Julia's garbage collector automatically manages `oneArray` objects. However, GPU memory is a limited resource. If you are running into out-of-memory errors, you might need to manually trigger garbage collection or free arrays.

```julia
a = oneArray(rand(Float32, 1024*1024*100))
a = nothing
GC.gc() # Reclaim memory
```

## Explicit Freeing

For immediate memory release, you can use `unsafe_free!`:

```julia
using oneAPI

a = oneArray(rand(1024))
oneAPI.unsafe_free!(a)
```

**Warning**: Only use `unsafe_free!` if you are sure the array is no longer used, including by any pending GPU operations.

