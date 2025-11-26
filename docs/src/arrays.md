# Array Programming

oneAPI.jl provides an array type, `oneArray`, which lives on the GPU. It implements the interface defined by `GPUArrays.jl`, allowing for high-level array operations.

## The `oneArray` Type

The `oneArray{T,N}` type represents an N-dimensional array with elements of type `T` stored on the GPU.

```julia
using oneAPI

# Allocate an uninitialized array
a = oneArray{Float32}(undef, 1024)

# Initialize from a CPU array
b = oneArray([1, 2, 3, 4])

# Initialize with zeros/ones
z = oneAPI.zeros(Float32, 100)
o = oneAPI.ones(Float32, 100)
```

## Array Operations

Since `oneArray` implements the AbstractArray interface, you can use standard Julia array operations.

```julia
a = oneArray(rand(Float32, 10))
b = oneArray(rand(Float32, 10))

c = a .+ b        # Element-wise addition
d = sum(a)        # Reduction
e = map(sin, a)   # Map
```

## Data Transfer

To move data between the host (CPU) and the device (GPU), use the constructors or `copyto!`.

```julia
# CPU to GPU
d_a = oneArray(h_a)

# GPU to CPU
h_a = Array(d_a)
```

## Backend Agnostic Programming

To write code that works on both CPU and GPU (and other backends like CUDA), use the generic array interfaces provided by `GPUArrays.jl`. Avoid hardcoding `oneArray` in your functions; instead, accept `AbstractArray` and let the dispatch system handle the specific implementation.

```julia
function generic_add!(a::AbstractArray, b::AbstractArray)
    a .+= b
    return a
end

# Works on CPU
generic_add!(rand(10), rand(10))

# Works on Intel GPU
generic_add!(oneArray(rand(10)), oneArray(rand(10)))
```

