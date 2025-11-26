# Performance Guide

This guide provides tips and techniques for optimizing oneAPI.jl applications.

## Quick Wins

### 1. Use Device Memory

Device memory is fastest for GPU operations:

```julia
# ✅ Good: Device memory (default)
a = oneArray{Float32}(undef, 1000)

# ❌ Slower: Shared memory (unless CPU access is needed)
a = oneArray{Float32,1,oneL0.SharedBuffer}(undef, 1000)
```

### 2. Minimize Data Transfers

Keep data on GPU between operations:

```julia
# ❌ Bad: Unnecessary transfers
for i in 1:100
    cpu_data = Array(gpu_array)  # GPU → CPU
    cpu_data .+= 1
    gpu_array = oneArray(cpu_data)  # CPU → GPU
end

# ✅ Good: Keep data on GPU
for i in 1:100
    gpu_array .+= 1  # All on GPU
end
```

### 3. Use Fused Operations

Broadcasting automatically fuses operations:

```julia
# ❌ Slower: Multiple kernel launches
a = oneArray(rand(Float32, 1000))
b = sin.(a)
c = b .+ 1.0f0
d = c .* 2.0f0

# ✅ Faster: Single fused kernel
d = 2.0f0 .* (sin.(a) .+ 1.0f0)
```

### 4. Specify Float32

GPUs are typically optimized for single precision:

```julia
# ❌ Slower: Float64 (if not needed)
a = oneArray(rand(Float64, 1000))

# ✅ Faster: Float32
a = oneArray(rand(Float32, 1000))
```

## Kernel Optimization

### Launch Configuration

Choose appropriate workgroup sizes:

```julia
# Typical good workgroup sizes
items = 256   # Common choice, adjust based on hardware
items = 128   # Try smaller if using lots of local memory
items = 512   # Try larger for simple kernels

# Calculate groups
N = length(array)
groups = cld(N, items)  # Ceiling division

@oneapi groups=groups items=items kernel(array)
```

### Memory Access Patterns

Coalesced memory access is crucial for performance:

```julia
# ✅ Good: Coalesced access (consecutive threads access consecutive memory)
function good_kernel!(output, input)
    i = get_global_id()
    @inbounds output[i] = input[i] + 1.0f0
    return
end

# ❌ Bad: Strided access (cache inefficient)
function bad_kernel!(output, input, stride)
    i = get_global_id()
    @inbounds output[i] = input[i * stride] + 1.0f0
    return
end
```

### Use Local Memory

Local memory is faster than global memory for data reuse:

```julia
function optimized_reduction!(result, input)
    local_id = get_local_id()
    local_size = get_local_size()
    group_id = get_group_id()

    # Allocate local memory
    local_mem = oneLocalArray(Float32, 256)

    # Load global → local (coalesced)
    global_id = get_global_id()
    @inbounds local_mem[local_id] = input[global_id]
    barrier()

    # Reduce in local memory (much faster)
    stride = local_size ÷ 2
    while stride > 0
        if local_id <= stride
            @inbounds local_mem[local_id] += local_mem[local_id + stride]
        end
        barrier()
        stride ÷= 2
    end

    # Write result
    if local_id == 1
        @inbounds result[group_id] = local_mem[1]
    end
    return
end
```

### Minimize Barriers

Barriers have overhead:

```julia
# ❌ Bad: Unnecessary barriers
function wasteful_kernel!(a)
    i = get_local_id()
    a[i] += 1
    barrier()  # Not needed if no data sharing
    a[i] *= 2
    barrier()  # Not needed
    return
end

# ✅ Good: Barriers only when needed
function efficient_kernel!(a, shared)
    i = get_local_id()

    # Load to shared memory
    shared[i] = a[i]
    barrier()  # Needed: ensure all loads complete

    # Use shared data
    result = shared[i] + shared[i+1]
    a[i] = result
    return
end
```

### Avoid Divergence

Minimize thread divergence (different execution paths):

```julia
# ❌ Bad: High divergence
function divergent_kernel!(a)
    i = get_global_id()
    if i % 32 == 0
        # Only 1 in 32 threads executes this
        @inbounds a[i] = expensive_computation(a[i])
    else
        @inbounds a[i] += 1.0f0
    end
    return
end

# ✅ Better: Separate into different kernels
function uniform_kernel!(a)
    i = get_global_id()
    @inbounds a[i] += 1.0f0
    return
end

function sparse_kernel!(a, indices)
    i = get_global_id()
    if i <= length(indices)
        idx = indices[i]
        @inbounds a[idx] = expensive_computation(a[idx])
    end
    return
end
```

## Type Stability

Type instability severely hurts performance:

```julia
# ❌ Bad: Type unstable
function unstable_kernel!(output, input, flag)
    i = get_global_id()
    if flag
        value = input[i]  # Float32
    else
        value = 0         # Int
    end
    output[i] = value * 2  # Type uncertain!
    return
end

# ✅ Good: Type stable
function stable_kernel!(output, input, flag)
    i = get_global_id()
    if flag
        value = input[i]  # Float32
    else
        value = 0.0f0     # Float32
    end
    output[i] = value * 2.0f0  # All Float32!
    return
end

# Check type stability
@device_code_warntype @oneapi groups=1 items=10 stable_kernel!(output, input, true)
```

## Algorithmic Optimization

### Use Library Functions

Leverage optimized library implementations:

```julia
using oneAPI, LinearAlgebra

# ✅ Good: Use oneMKL through LinearAlgebra
A = oneArray(rand(Float32, 1000, 1000))
B = oneArray(rand(Float32, 1000, 1000))
C = A * B  # Uses optimized oneMKL

# ❌ Bad: Write your own matrix multiplication
# (unless you have a very specific use case)
```

### Choose Right Algorithm

Some algorithms parallelize better than others:

```julia
# ❌ Sequential algorithm
function sequential_sum(arr)
    sum = 0.0f0
    for x in arr
        sum += x
    end
    return sum
end

# ✅ Parallel reduction
result = sum(oneArray(data))  # Optimized parallel reduction
```

## Benchmarking

### Basic Timing

```julia
using BenchmarkTools, oneAPI

a = oneArray(rand(Float32, 1000))
b = oneArray(rand(Float32, 1000))

# Warmup
c = a .+ b
synchronize()

# Benchmark
@benchmark begin
    c = $a .+ $b
    synchronize()
end
```

### Accurate GPU Timing

Always synchronize before timing:

```julia
using oneAPI

a = oneArray(rand(Float32, 1_000_000))

# ❌ Wrong: Doesn't wait for GPU
@time a .+= 1  # Only measures kernel launch overhead

# ✅ Correct: Wait for GPU to finish
@time begin
    a .+= 1
    synchronize()
end
```

### Profiling with Time

```julia
function profile_operation(a, b)
    # Warmup
    c = a .+ b
    synchronize()

    # Time kernel launch
    t1 = time()
    c = a .+ b
    t2 = time()
    launch_time = t2 - t1

    # Time including synchronization
    synchronize()
    t3 = time()
    total_time = t3 - t1

    println("Launch: ", launch_time * 1000, " ms")
    println("Total:  ", total_time * 1000, " ms")
    println("Actual: ", (total_time - launch_time) * 1000, " ms")
end

a = oneArray(rand(Float32, 10_000_000))
b = oneArray(rand(Float32, 10_000_000))
profile_operation(a, b)
```

## Memory Bandwidth

### Theoretical Peak

Calculate theoretical bandwidth:

```julia
# Example: Intel Iris Xe Graphics
# 96 execution units, 1.35 GHz
# Memory bandwidth: ~68 GB/s

# Your kernel processes N Float32 values
N = 10_000_000
bytes_transferred = N * sizeof(Float32) * 2  # Read + Write

# Measure time
t = @elapsed begin
    a .+= b
    synchronize()
end

bandwidth_achieved = bytes_transferred / t / 1e9  # GB/s
println("Bandwidth: ", bandwidth_achieved, " GB/s")
```

### Improving Bandwidth Utilization

```julia
# ✅ Good: Single pass with fusion
result = @. a + b * c - d / e  # One pass over data

# ❌ Bad: Multiple passes
result = a .+ b
result = result .* c
result = result .- d
result = result ./ e
# Four separate passes over data!
```

## Common Performance Issues

### Issue 1: Too Many Small Kernels

```julia
# ❌ Bad: Many small kernel launches
for i in 1:100
    a .+= 1  # 100 kernel launches!
end

# ✅ Good: Single kernel or batching
a .+= 100  # Single operation
```

### Issue 2: Unnecessary Allocations

```julia
# ❌ Bad: Allocates temporary
c = a .+ b  # Allocates new array

# ✅ Good: In-place operation
c = similar(a)
c .= a .+ b  # Uses pre-allocated array
```

### Issue 3: Wrong Number Type

```julia
# ❌ Bad: Mixed types
a = oneArray(rand(Float32, 1000))
b = a .+ 1.0  # Float64 constant!

# ✅ Good: Matching types
b = a .+ 1.0f0  # Float32 constant
```

## Performance Checklist

- [ ] Using device memory (not shared unless necessary)
- [ ] Minimizing CPU-GPU transfers
- [ ] Using Float32 (unless Float64 required)
- [ ] Fusing operations with broadcasting
- [ ] Type-stable kernels (`@device_code_warntype`)
- [ ] Appropriate workgroup sizes
- [ ] Coalesced memory access
- [ ] Minimal thread divergence
- [ ] Leveraging local memory for reuse
- [ ] Using library functions when available
- [ ] Synchronizing before timing
- [ ] Avoiding unnecessary allocations

## Hardware-Specific Tuning

Different Intel GPUs have different characteristics:

```julia
using oneAPI.oneL0

dev = device()
props = properties(dev)
compute_props = compute_properties(dev)

println("Device: ", props.name)
println("EU count: ", compute_props.numEUsPerSubslice *
                       compute_props.numSubslicesPerSlice *
                       compute_props.numSlices)
println("Max workgroup size: ", compute_props.maxTotalGroupSize)
println("Max local memory: ", compute_props.maxSharedLocalMemory, " bytes")

# Adjust your code based on these properties
```

## Advanced: Async Operations

For overlapping compute and transfers (advanced users):

```julia
using oneAPI.oneL0

ctx = context()
dev = device()

# Create multiple queues for async operations
queue1 = ZeCommandQueue(ctx, dev)
queue2 = ZeCommandQueue(ctx, dev)

# Launch kernel on queue1
execute!(queue1) do list
    # ... kernel launch ...
end

# Overlap with transfer on queue2
execute!(queue2) do list
    append_copy!(list, dst, src, size)
end

# Synchronize both
synchronize(queue1)
synchronize(queue2)
```

## Further Resources

- [Intel GPU Architecture](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-gpu-architecture.html)
- [oneAPI Programming Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/programming-guide.html)
- [Level Zero Specification](https://spec.oneapi.io/level-zero/latest/index.html)
