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
`oneSparseMatrixCOO`. They are subtypes of the corresponding GPUArrays.jl abstract types
(`AbstractGPUSparseMatrixCSR`, `AbstractGPUSparseMatrixCSC`, `AbstractGPUSparseMatrixCOO`), so the
generic sparse functionality of GPUArrays.jl is available: broadcasting (zero-preserving functions
return a sparse matrix, others a dense `oneArray`), `sum`/`mapreduce` (also along a dimension),
`norm`/`opnorm`, `findnz`, `triu`/`tril`/`kron`, `iszero`, and scalar indexing under
`GPUArrays.@allowscalar`. Matrices can be converted between the three formats, transposed and
added on the device, and `adapt(oneArray, A)` of a `SparseMatrixCSC` yields a `oneSparseMatrixCSC`.

```julia
dA = oneSparseMatrixCSR(sprand(Float32, 100, 100, 0.1))
dB = dA .* 2f0                      # oneSparseMatrixCSR
dC = dA .+ 1f0                      # dense oneMatrix
sum(dA; dims=1)                     # row vector
dAt = oneSparseMatrixCSC(dA)        # format conversion, on the device
dS = dA + transpose(dA)             # sparse addition
```

Any element type can be stored in these matrices, but the oneMKL operations (`*`, `mul!`, the
triangular solves, and the `sparse_*!` wrappers) require `Float32`, `Float64`, `ComplexF32` or
`ComplexF64` values with `Int32` or `Int64` indices. The oneMKL matrix handle is created lazily
when such an operation is first invoked; it refers to the storage vectors of the matrix, which
therefore must not be modified in place afterwards (use `copyto!` or create a new matrix instead).

oneMKL's sparse back-end is CSR-based, and a `oneSparseMatrixCSC` is therefore handed to oneMKL as
the CSR representation of its transpose (this requires oneMKL 2025.3 or later). As a consequence
the triangular operations (`sparse_trmv!`, `sparse_trsv!`, `sparse_trsm!`) cannot be expressed
for CSC matrices and throw an `ArgumentError`. Prefer CSR when you have the choice.

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

