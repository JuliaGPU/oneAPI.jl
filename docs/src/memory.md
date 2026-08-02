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
using oneAPI, oneAPI.oneL0

drv = first(drivers())
ctx = ZeContext(drv)
dev = first(devices(drv))

# Allocate 1024 bytes of device memory, aligned to 8 bytes
buf = device_alloc(ctx, dev, 1024, 8)

# Free memory
free(buf)
```

`host_alloc` and `shared_alloc` allocate host and shared USM memory respectively, using the
same signature.

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

## Memory Limit

When multiple processes share one GPU, each process only tracks its own allocations,
and garbage collection in one process cannot free memory held by another. To keep the
processes from collectively exhausting the device, give each a budget with the
`ONEAPI_MEMORY_LIMIT` environment variable, either as a byte count or as a percentage
of device memory:

```
ONEAPI_MEMORY_LIMIT=50%          # this process budgets half the device
ONEAPI_MEMORY_LIMIT=4000000000   # ... or an absolute byte count
```

This is a soft limit: it scales the thresholds at which oneAPI.jl proactively runs the
garbage collector to free stale GPU buffers, before falling back to a full collection
when an allocation actually fails. Allocations larger than the limit are still allowed
to succeed. The test suite sets this automatically for its parallel workers.

