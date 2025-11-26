# Kernel Programming

This page documents the kernel programming API for writing custom GPU kernels in oneAPI.jl.

## Kernel Launch

### `@oneapi [kwargs...] kernel(args...)`

High-level interface for launching Julia kernels on Intel GPUs using oneAPI.

This macro compiles a Julia function to SPIR-V, prepares the arguments, and optionally
launches the kernel on the GPU.

**Keyword Arguments:**

**Macro Keywords (compile-time):**
- `launch::Bool=true`: Whether to launch the kernel immediately

**Compiler Keywords:**
- `kernel::Bool=false`: Whether to compile as a kernel or device function
- `name::Union{String,Nothing}=nothing`: Explicit name for the kernel
- `always_inline::Bool=false`: Whether to always inline device functions

**Launch Keywords (runtime):**
- `groups`: Number of workgroups (required). Can be an integer or tuple.
- `items`: Number of work-items per workgroup (required). Can be an integer or tuple.
- `queue::ZeCommandQueue`: Command queue to submit to (defaults to global queue).

### `zefunction(f, tt; kwargs...)`

Compile a Julia function to a Level Zero kernel function. This is the lower-level interface
used by `@oneapi`. Returns a callable kernel object.

### `kernel_convert(x)`

Convert arguments for kernel execution. This function is called for every argument passed to
a kernel, allowing customization of argument conversion. By default, it converts `oneArray`
to `oneDeviceArray`.


## Basic Kernel Example

```julia
using oneAPI

function vadd_kernel!(a, b, c)
    i = get_global_id()
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

N = 1024
a = oneArray(rand(Float32, N))
b = oneArray(rand(Float32, N))
c = similar(a)

# Launch with 4 workgroups of 256 work-items each
@oneapi groups=4 items=256 vadd_kernel!(a, b, c)
```

## Launch Configuration

### Workgroups and Work-Items

The oneAPI execution model is based on:
- **Work-items**: Individual threads of execution (analogous to CUDA threads)
- **Workgroups**: Groups of work-items that can synchronize and share local memory (analogous to CUDA blocks)

```julia
# 1D configuration
@oneapi groups=10 items=64 kernel(args...)        # 640 work-items total

# 2D configuration
@oneapi groups=(10, 10) items=(8, 8) kernel(args...)  # 6400 work-items total

# 3D configuration
@oneapi groups=(4, 4, 4) items=(4, 4, 4) kernel(args...)  # 4096 work-items total
```

### Determining Launch Configuration

```julia
# For simple element-wise operations
N = length(array)
items = 256  # Typical workgroup size
groups = cld(N, items)  # Ceiling division

@oneapi groups=groups items=items kernel(array)
```

### Compile Without Launch

You can compile a kernel without launching it:

```julia
# Compile the kernel
kernel = @oneapi launch=false vadd_kernel!(a, b, c)

# Launch later with different configurations
kernel(a, b, c; groups=4, items=256)
kernel(a, b, c; groups=8, items=128)
```

## Device Intrinsics

Inside GPU kernels, you can use various intrinsics to query execution context and synchronize work-items.

### Thread Indexing

```julia
# Global ID (unique across all work-items)
i = get_global_id()      # 1D linear index
i = get_global_id(0)     # X dimension
j = get_global_id(1)     # Y dimension
k = get_global_id(2)     # Z dimension

# Local ID (within workgroup)
local_i = get_local_id()   # 1D linear index
local_i = get_local_id(0)  # X dimension
local_j = get_local_id(1)  # Y dimension
local_k = get_local_id(2)  # Z dimension

# Workgroup ID
group_i = get_group_id(0)  # X dimension
group_j = get_group_id(1)  # Y dimension
group_k = get_group_id(2)  # Z dimension

# Workgroup size
local_size = get_local_size()   # Total work-items in workgroup
local_size_x = get_local_size(0)
local_size_y = get_local_size(1)

# Global size
global_size = get_global_size()   # Total work-items
global_size_x = get_global_size(0)
```

### 2D Matrix Example

```julia
function matmul_kernel!(C, A, B)
    # Get 2D indices
    row = get_global_id(0)
    col = get_global_id(1)

    if row <= size(C, 1) && col <= size(C, 2)
        sum = 0.0f0
        for k in 1:size(A, 2)
            @inbounds sum += A[row, k] * B[k, col]
        end
        @inbounds C[row, col] = sum
    end
    return
end

M, N, K = 256, 256, 256
A = oneArray(rand(Float32, M, K))
B = oneArray(rand(Float32, K, N))
C = oneArray{Float32}(undef, M, N)

# Launch with 2D configuration
items = (16, 16)  # 16x16 work-items per workgroup
groups = (cld(M, items[1]), cld(N, items[2]))

@oneapi groups=groups items=items matmul_kernel!(C, A, B)
```

### Synchronization

```julia
# Barrier: synchronize all work-items in a workgroup
barrier()

# Memory fences (ensure memory operations are visible)
mem_fence()     # Both local and global memory
local_mem_fence()   # Local memory only
global_mem_fence()  # Global memory only
```

### Local Memory

Local memory (workgroup-shared memory) enables cooperation between work-items:

```julia
function optimized_reduction!(result, input)
    local_id = get_local_id()
    local_size = get_local_size()

    # Allocate local memory (shared within workgroup)
    local_data = oneLocalArray(Float32, 256)

    # Load into local memory
    @inbounds local_data[local_id] = input[get_global_id()]
    barrier()

    # Tree reduction in local memory
    stride = local_size ÷ 2
    while stride > 0
        if local_id <= stride
            @inbounds local_data[local_id] += local_data[local_id + stride]
        end
        barrier()
        stride ÷= 2
    end

    # First work-item writes result
    if local_id == 1
        @inbounds result[get_group_id()] = local_data[1]
    end
    return
end
```

### Atomic Operations

For thread-safe operations on shared data:

```julia
# Atomic add
oneAPI.atomic_add!(ptr, value)

# Atomic exchange
old_value = oneAPI.atomic_xchg!(ptr, new_value)

# Atomic compare-and-swap
old_value = oneAPI.atomic_cas!(ptr, compare, new_value)

# Atomic min/max
oneAPI.atomic_min!(ptr, value)
oneAPI.atomic_max!(ptr, value)
```

Example histogram kernel:

```julia
function histogram_kernel!(hist, data, bins)
    i = get_global_id()
    if i <= length(data)
        @inbounds val = data[i]
        bin = clamp(floor(Int, val * bins) + 1, 1, bins)
        oneAPI.atomic_add!(pointer(hist, bin), 1)
    end
    return
end
```

## Kernel Restrictions

GPU kernels have certain restrictions:

1. **Must return `nothing`**: Kernels cannot return values directly. Use output arrays instead.
2. **No dynamic memory allocation**: Cannot allocate arrays inside kernels
3. **No I/O operations**: Cannot print or write to files (use printf-style debugging with care)
4. **Limited recursion**: Avoid or minimize recursive calls
5. **Type stability**: Ensure type-stable code for best performance

```julia
# ❌ Bad: Returns a value
function bad_kernel(a)
    return a[1] + 1
end

# ✅ Good: Returns nothing, uses output parameter
function good_kernel!(result, a)
    @inbounds result[1] = a[1] + 1
    return
end
```

## KernelAbstractions.jl

For portable GPU programming across CUDA, AMD, and Intel GPUs, use KernelAbstractions.jl:

```julia
using KernelAbstractions
using oneAPI

@kernel function generic_kernel!(a, b)
    i = @index(Global)
    @inbounds a[i] = a[i] + b[i]
end

a = oneArray(rand(Float32, 100))
b = oneArray(rand(Float32, 100))

backend = get_backend(a)  # oneAPIBackend()
kernel! = generic_kernel!(backend)
kernel!(a, b, ndrange=length(a))
```

See the [KernelAbstractions.jl documentation](https://juliagpu.github.io/KernelAbstractions.jl/stable/) for more details.

## Debugging Kernels

See the [Compiler and Reflection](@ref) page for tools to inspect generated code and debug kernels.
