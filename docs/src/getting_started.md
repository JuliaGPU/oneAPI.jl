# Getting Started

## Basic Usage

The most basic usage involves moving data to the GPU using `oneArray` and performing operations on it.

```julia
using oneAPI

# Create an array on the CPU
a = rand(Float32, 1024)

# Move it to the GPU
d_a = oneArray(a)

# Perform operations on the GPU
d_b = d_a .+ 1.0f0

# Move the result back to the CPU
b = Array(d_b)
```

## Matrix Multiplication

Matrix multiplication is accelerated using the oneMKL library when available.

```julia
using oneAPI

A = oneArray(rand(Float32, 128, 128))
B = oneArray(rand(Float32, 128, 128))

# This operation runs on the GPU
C = A * B
```

## Writing Kernels

For custom operations, you can write kernels using the `@oneapi` macro.

```julia
using oneAPI

function my_kernel(a, b)
    i = get_global_id()
    @inbounds a[i] += b[i]
    return
end

a = oneArray(ones(Float32, 1024))
b = oneArray(ones(Float32, 1024))

# Launch the kernel with 1024 items
@oneapi items=1024 my_kernel(a, b)
```

See the [Kernel Programming](kernels.md) section for more details.

