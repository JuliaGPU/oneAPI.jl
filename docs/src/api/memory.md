# Memory Management

This page documents memory management in oneAPI.jl.

## Memory Operations

### `Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dst, src, N)`

Low-level memory copy operation on the GPU. Copies `N` elements from `src` to `dst` using
the specified context and device. Both `src` and `dst` can be either host pointers (`Ptr`)
or device pointers (`ZePtr`).

!!! warning
    This is a low-level function. No bounds checking is performed. For safe array copying,
    use `copyto!` on `oneArray` objects instead.

### `unsafe_fill!(ctx::ZeContext, dev::ZeDevice, ptr, pattern, N)`

Low-level memory fill operation on the GPU. Fills `N` elements at `ptr` with the given pattern
using the specified context and device.

!!! warning
    This is a low-level function. For safe array operations, use `fill!` on `oneArray`
    objects instead.


## Memory Types

oneAPI supports three types of memory through Unified Shared Memory (USM):

### Device Memory (Default)

Fastest GPU access, not directly accessible from CPU.

```julia
# Create array in device memory (default)
a = oneArray{Float32}(undef, 1000)
@assert is_device(a)

# Or explicitly specify
b = oneArray{Float32,1,oneL0.DeviceBuffer}(undef, 1000)
```

**Advantages:**
- Fastest GPU access
- Best for compute-intensive operations

**Disadvantages:**
- Cannot directly access from CPU
- Requires explicit copy to/from CPU

**Use when:** Data stays on GPU for multiple operations

### Shared Memory

Accessible from both CPU and GPU with automatic migration.

```julia
# Create array in shared memory
a = oneArray{Float32,1,oneL0.SharedBuffer}(undef, 1000)
@assert is_shared(a)

# Can access from CPU
a[1] = 42.0f0  # Automatic migration to CPU
println(a[1])  # Read from CPU

# Can use in GPU kernels
@oneapi groups=1 items=1000 kernel(a)  # Automatic migration to GPU
```

**Advantages:**
- Accessible from both CPU and GPU
- Unified virtual addressing
- Automatic migration

**Disadvantages:**
- Migration overhead
- Slower than device memory for pure GPU work

**Use when:** Frequent CPU-GPU data exchange needed

### Host Memory

CPU memory that's pinned and visible to GPU.

```julia
# Create array in host memory
a = oneArray{Float32,1,oneL0.HostBuffer}(undef, 1000)
@assert is_host(a)

# Direct CPU access
a[1] = 42.0f0

# Can be used by GPU (but slower than device memory)
@oneapi groups=1 items=1000 kernel(a)
```

**Advantages:**
- Direct CPU access
- Pinned memory (faster PCIe transfers)
- Good for staging

**Disadvantages:**
- Slower GPU access than device memory
- Uses pinned system memory (limited resource)

**Use when:** Staging data for transfer, or CPU needs to write while GPU reads

## Memory Type Comparison

| Feature | Device | Shared | Host |
|---------|--------|--------|------|
| CPU Access | ❌ No | ✅ Yes | ✅ Yes |
| GPU Performance | ⭐⭐⭐ Fastest | ⭐⭐ Good | ⭐ Slower |
| Migration | Manual | Automatic | Manual |
| Use Case | Pure GPU | Mixed CPU/GPU | Staging |

## Memory Allocation and Deallocation

### Automatic Management

Julia's garbage collector automatically manages `oneArray` memory:

```julia
function allocate_and_compute()
    a = oneArray(rand(Float32, 1000))
    b = oneArray(rand(Float32, 1000))
    c = a .+ b
    return Array(c)  # Only c is copied back
    # a and b will be garbage collected
end

result = allocate_and_compute()
# GPU memory for a and b is freed eventually
```

### Manual Garbage Collection

Force garbage collection to free GPU memory:

```julia
# Allocate large arrays
a = oneArray(rand(Float32, 10_000_000))
b = oneArray(rand(Float32, 10_000_000))

# Clear references
a = nothing
b = nothing

# Force GC to reclaim GPU memory
GC.gc()
```

### Explicit Freeing

Immediately free GPU memory (use with caution):

```julia
a = oneArray(rand(Float32, 1000))
# ... use a ...

# Explicitly free (dangerous if still in use!)
unsafe_free!(a)

# a is now invalid - do not use!
```

!!! warning
    Only use `unsafe_free!` when you're certain the array is no longer needed, including
    by any pending GPU operations. Prefer letting the GC handle cleanup.

### Do-Block Pattern

Use do-blocks for automatic cleanup:

```julia
result = oneArray{Float32}(1000) do temp
    # temp is automatically freed when block exits
    temp .= 1.0f0
    sum(temp)  # Result is returned
end
```

## Memory Pooling

oneAPI.jl uses memory pooling to reduce allocation overhead:

```julia
using oneAPI

# Allocations are pooled
for i in 1:100
    a = oneArray(rand(Float32, 1000))
    # ... use a ...
    # Memory is returned to pool, not freed
end
```

The pool automatically manages memory reuse, reducing allocation costs.

## Checking Memory Usage

Query GPU memory info:

```julia
using oneAPI.oneL0

dev = device()
props = memory_properties(dev)

for prop in props
    println("Memory size: ", prop.totalSize ÷ (1024^3), " GB")
end
```

## Out of Memory Errors

If you encounter out-of-memory errors:

### 1. Reduce Batch Size

```julia
# Instead of processing all at once
result = process(oneArray(huge_data))

# Process in smaller batches
for batch in batches(huge_data, size=1000)
    result = process(oneArray(batch))
    # Process result...
end
```

### 2. Free Unused Arrays

```julia
a = oneArray(rand(Float32, 1_000_000))
b = compute(a)

# If 'a' is no longer needed
unsafe_free!(a)

# Continue with 'b'
result = process(b)
```

### 3. Use Shared or Host Memory

```julia
# Instead of device memory
a = oneArray{Float32}(undef, huge_size)

# Use shared memory (can swap to system RAM)
a = oneArray{Float32,1,oneL0.SharedBuffer}(undef, huge_size)
```

### 4. Force Garbage Collection

```julia
# After freeing references
large_array = nothing
GC.gc()  # Immediately reclaim GPU memory
```

### 5. Use Multiple Devices

```julia
# Distribute work across devices
for (i, dev_id) in enumerate(1:length(devices()))
    Threads.@spawn begin
        device!(dev_id)
        partition = data_partitions[i]
        a = oneArray(partition)
        result = compute(a)
        # ...
    end
end
```

## Low-Level Memory Operations

For advanced users, oneL0 provides direct memory management:

```julia
using oneAPI.oneL0

ctx = context()
dev = device()

# Allocate device memory
ptr = device_alloc(ctx, dev, 1024, 8)  # 1024 bytes, 8-byte aligned

# Copy data
data = rand(Float32, 256)
GC.@preserve data begin
    unsafe_copyto!(ctx, dev, ptr, pointer(data), 256)
end

# Free memory
free(ctx, ptr)
```

## Memory Advise and Prefetch

Hint to the runtime about memory usage (shared memory only):

```julia
using oneAPI.oneL0

a = oneArray{Float32,1,oneL0.SharedBuffer}(undef, 1000)

# Advise that this will be read-only on the device
# (Implementation depends on Level Zero driver support)

# Prefetch to device
ctx = context()
dev = device()
queue = global_queue(ctx, dev)

execute!(queue) do list
    append_prefetch!(list, pointer(a), sizeof(a))
end
```

## Best Practices

1. **Use device memory by default** for best GPU performance
2. **Use shared memory** when you need CPU access without explicit copies
3. **Use host memory** for staging data or when CPU writes frequently
4. **Let GC handle cleanup** unless you have specific memory pressure
5. **Reuse allocations** within loops when possible
6. **Profile memory usage** to identify bottlenecks
7. **Be cautious with `unsafe_free!`** - use only when you're certain it's safe

## Example: Efficient Memory Usage

```julia
using oneAPI

function efficient_pipeline(data_batches)
    # Allocate output buffer once
    result = oneArray{Float32}(undef, 1000)
    results = Float32[]

    for batch in data_batches
        # Reuse input buffer by copying
        input = oneArray(batch)

        # Compute in-place when possible
        @oneapi groups=4 items=250 process_kernel!(result, input)

        # Copy result back
        push!(results, Array(result)...)

        # Input is freed when loop continues
    end

    return results
end
```
