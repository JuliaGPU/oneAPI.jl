# oneMKL Integration

oneAPI.jl provides bindings to the Intel oneMKL library, enabling high-performance linear algebra operations on Intel GPUs.

## Dense Linear Algebra (BLAS/LAPACK)

Standard BLAS and LAPACK operations are automatically accelerated when using `oneArray`.

```julia
using oneAPI, LinearAlgebra

A = oneArray(rand(Float32, 100, 100))
B = oneArray(rand(Float32, 100, 100))

# Matrix multiplication (GEMM)
C = A * B

# Linear solve (AX = B)
X = A \ B
```

## Sparse Linear Algebra

oneAPI.jl supports sparse matrix operations via oneMKL's sparse BLAS functionality. These integrate with Julia's `SparseArrays` standard library.

```julia
using oneAPI, oneAPI.oneMKL, SparseArrays, LinearAlgebra

# Create a sparse matrix on CPU
A = sprand(100, 100, 0.1)

# Move to GPU (converts to oneMKL format)
dA = oneMKL.oneSparseMatrixCSC(A)

# Create a dense vector
x = oneArray(rand(100))

# Sparse matrix-vector multiplication
y = dA * x
```

Note that `oneSparseMatrixCSC` is available for Compressed Sparse Column format, which is the standard in Julia.

## FFTs

Fast Fourier Transforms are supported through `AbstractFFTs.jl` interface integration with oneMKL DFTs.

```julia
using oneAPI, FFTW

a = oneArray(rand(ComplexF32, 1024))

# Forward FFT
b = fft(a)

# Inverse FFT
c = ifft(b)
```

