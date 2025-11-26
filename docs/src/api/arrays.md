# Array Operations

This page documents the array types and operations provided by oneAPI.jl.

## Array Types

### Host-Side Arrays

#### `oneArray{T,N,B}`

N-dimensional dense array type for Intel GPU programming using oneAPI and Level Zero.

**Type Parameters:**
- `T`: Element type (must be stored inline, no isbits-unions)
- `N`: Number of dimensions
- `B`: Buffer type, one of:
  - `oneL0.DeviceBuffer`: GPU device memory (default, not CPU-accessible)
  - `oneL0.SharedBuffer`: Unified shared memory (CPU and GPU accessible)
  - `oneL0.HostBuffer`: Pinned host memory (CPU-accessible, GPU-visible)

**Type Aliases:**
- `oneVector{T}` = `oneArray{T,1}` - 1D array
- `oneMatrix{T}` = `oneArray{T,2}` - 2D array
- `oneVecOrMat{T}` = `Union{oneVector{T}, oneMatrix{T}}` - 1D or 2D array

### Device-Side Arrays

#### `oneDeviceArray{T,N,A}`

Device-side array type for use within GPU kernels. This type represents a view of GPU memory
accessible within kernel code. Unlike `oneArray` which is used on the host, `oneDeviceArray`
is designed for device-side operations and cannot be directly constructed on the host.

**Type Parameters:**
- `T`: Element type
- `N`: Number of dimensions
- `A`: Address space (typically `AS.CrossWorkgroup` for global memory)

**Type Aliases:**
- `oneDeviceVector` = `oneDeviceArray{T,1}` - 1D device array
- `oneDeviceMatrix` = `oneDeviceArray{T,2}` - 2D device array

#### `oneLocalArray(::Type{T}, dims)`

Allocate local (workgroup-shared) memory within a GPU kernel. Local memory is shared among
all work-items in a workgroup and provides faster access than global memory.

## Memory Type Queries

### `is_device(a::oneArray) -> Bool`

Check if the array is stored in device memory (not directly CPU-accessible).

### `is_shared(a::oneArray) -> Bool`

Check if the array is stored in shared (unified) memory, accessible from both CPU and GPU.

### `is_host(a::oneArray) -> Bool`

Check if the array is stored in pinned host memory, which resides on the CPU but is visible to the GPU.


## Array Construction

`oneArray` supports multiple construction patterns similar to standard Julia arrays:

```julia
using oneAPI

# Uninitialized arrays
a = oneArray{Float32}(undef, 100)
b = oneArray{Float32,2}(undef, 10, 10)

# Specify memory type
c = oneArray{Float32,1,oneL0.SharedBuffer}(undef, 100)  # Shared memory
d = oneArray{Float32,1,oneL0.HostBuffer}(undef, 100)    # Host memory

# From existing arrays
e = oneArray(rand(Float32, 100))
f = oneArray([1, 2, 3, 4])

# Using zeros/ones/rand
g = oneAPI.zeros(Float32, 100)
h = oneAPI.ones(Float32, 100)
i = oneAPI.rand(Float32, 100)

# Do-block for automatic cleanup
result = oneArray{Float32}(100) do arr
    arr .= 1.0f0
    sum(arr)  # Returns result, arr is freed automatically
end
```

## Array Operations

`oneArray` implements the full `AbstractArray` interface and supports:

### Broadcasting

```julia
a = oneArray(rand(Float32, 100))
b = oneArray(rand(Float32, 100))

c = a .+ b          # Element-wise addition
d = a .* 2.0f0      # Scalar multiplication
e = sin.(a)         # Unary operations
f = a .+ b .* c     # Fused operations
```

### Reductions

```julia
a = oneArray(rand(Float32, 100))

s = sum(a)          # Sum
p = prod(a)         # Product
m = maximum(a)      # Maximum
n = minimum(a)      # Minimum
μ = mean(a)         # Mean (requires Statistics)
```

### Mapping

```julia
a = oneArray(rand(Float32, 100))

b = map(x -> x^2, a)        # Apply function
c = map(+, a, b)            # Binary operation
```

### Accumulation

```julia
a = oneArray([1, 2, 3, 4])

b = cumsum(a)       # Cumulative sum: [1, 3, 6, 10]
c = cumprod(a)      # Cumulative product: [1, 2, 6, 24]
```

### Finding Elements

```julia
a = oneArray([1.0f0, -2.0f0, 3.0f0, -4.0f0])

indices = findall(x -> x > 0, a)  # Indices of positive elements
```

### Random Number Generation

```julia
using oneAPI, Random

# Uniform distribution
a = oneAPI.rand(Float32, 100)
b = oneAPI.rand(Float32, 10, 10)

# Normal distribution
c = oneAPI.randn(Float32, 100)

# With seed
Random.seed!(1234)
d = oneAPI.rand(Float32, 100)
```

## Data Transfer

### CPU to GPU

```julia
# Using constructor
h_array = rand(Float32, 100)
d_array = oneArray(h_array)

# Using copyto!
d_array = oneArray{Float32}(undef, 100)
copyto!(d_array, h_array)
```

### GPU to CPU

```julia
# Using Array constructor
h_array = Array(d_array)

# Using copyto!
h_array = Vector{Float32}(undef, 100)
copyto!(h_array, d_array)
```

### GPU to GPU

```julia
d_array1 = oneArray(rand(Float32, 100))
d_array2 = similar(d_array1)
copyto!(d_array2, d_array1)
```

## Memory Types Comparison

| Memory Type | CPU Access | GPU Access | Performance | Use Case |
|-------------|-----------|------------|-------------|----------|
| Device (default) | ❌ No | ✅ Fast | Fastest | GPU computations |
| Shared | ✅ Yes | ✅ Good | Good | CPU-GPU data sharing |
| Host | ✅ Yes | ✅ Slower | Moderate | Staging, pinned buffers |

```julia
# Device memory (default, fastest for GPU)
a = oneArray{Float32}(undef, 100)

# Shared memory (CPU and GPU accessible)
b = oneArray{Float32,1,oneL0.SharedBuffer}(undef, 100)

# Host memory (CPU memory visible to GPU)
c = oneArray{Float32,1,oneL0.HostBuffer}(undef, 100)

# Query memory type
is_device(a)  # true
is_shared(b)  # true
is_host(c)    # true
```

## Views and Slicing

`oneArray` supports array views for efficient sub-array operations without copying:

```julia
a = oneArray(rand(Float32, 100))

# Create a view
v = view(a, 1:50)
v .= 0.0f0  # Modifies first 50 elements of a

# Slicing returns a view
s = a[1:50]  # This is a view, not a copy
```

## Reshaping

```julia
a = oneArray(rand(Float32, 100))

# Reshape to 2D
b = reshape(a, 10, 10)

# Flatten
c = vec(b)  # Returns 1D view
```

## Advanced: Custom Array Wrappers

For advanced use cases, oneAPI.jl provides type aliases for array wrappers:

- `oneDenseArray`: Dense contiguous arrays
- `oneStridedArray`: Arrays with arbitrary strides (including views)
- `oneWrappedArray`: Any array backed by a oneArray

These are useful for writing functions that accept various array types:

```julia
function my_kernel!(a::oneStridedArray{Float32})
    # Accepts oneArray and views
    a .+= 1.0f0
end
```
