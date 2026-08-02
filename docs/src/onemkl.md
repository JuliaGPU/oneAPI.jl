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
A = sprand(Float32, 100, 100, 0.1)

# Move to GPU (converts to oneMKL format)
dA = oneMKL.oneSparseMatrixCSR(A)

# Create a dense vector
x = oneArray(rand(Float32, 100))

# Sparse matrix-vector multiplication
y = dA * x
```

Three storage formats are available: `oneSparseMatrixCSR`, `oneSparseMatrixCSC` and
`oneSparseMatrixCOO`. oneMKL's sparse back-end is CSR-based, and a `oneSparseMatrixCSC` is
therefore stored as the CSR representation of its transpose. As a consequence the triangular
operations (`sparse_trmv!`, `sparse_trsv!`, `sparse_trsm!`) cannot be expressed for CSC
matrices and throw an `ArgumentError`. Prefer CSR when you have the choice.

## FFTs

Fast Fourier Transforms are supported through `AbstractFFTs.jl` interface integration with oneMKL DFTs. oneAPI.jl depends on AbstractFFTs.jl, so no separate FFT package is required.

```julia
using oneAPI, AbstractFFTs

a = oneArray(rand(ComplexF32, 1024))

# Forward FFT
b = fft(a)

# Inverse FFT
c = ifft(b)
```

