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
`oneSparseMatrixCOO`, plus `oneSparseVector`. They are aliases of GPUArrays.jl's sparse types
with `oneVector` storage (e.g. `oneSparseMatrixCSR{Tv,Ti}` is a `GPUSparseMatrixCSR` whose
buffers are `oneVector`s), so the generic sparse functionality of GPUArrays.jl is available:
broadcasting (zero-preserving functions return a sparse matrix, others a dense `oneArray`),
`sum`/`mapreduce` (also along a dimension), `norm`/`opnorm`, `findnz`, `triu`/`tril`/`kron`,
indexing with ranges, sparse × sparse products, and scalar indexing under
`GPUArrays.@allowscalar`. Matrices can be converted between the formats, transposed and added
on the device, and `adapt(oneArray, A)` of a `SparseMatrixCSC` yields a `oneSparseMatrixCSC`
with the same element and index types. Constructing a matrix from its buffers
(`oneSparseMatrixCSR(rowPtr, colVal, nzVal, dims)`) borrows them; see the GPUArrays.jl
documentation for the rules on sharing and mutating buffers.

```julia
dA = oneSparseMatrixCSR(sprand(Float32, 100, 100, 0.1))
dB = dA .* 2f0                      # oneSparseMatrixCSR
dC = dA .+ 1f0                      # dense oneMatrix
sum(dA; dims=1)                     # row vector
dAt = oneSparseMatrixCSC(dA)        # format conversion, on the device
dS = dA + transpose(dA)             # sparse addition
```

Any element type can be stored in these matrices. Products (`*`, `mul!`) use oneMKL for
`Float32`, `Float64`, `ComplexF32` or `ComplexF64` CSR and CSC matrices with `Int32` or `Int64`
indices, and GPUArrays' generic implementation otherwise (e.g. for COO matrices, other element
types, or transposed dense operands). The triangular solves and the `oneMKL.sparse_*!` wrappers
require these types.

The arrays themselves hold no oneMKL state: every operation describes the matrix to oneMKL for
the duration of the call. For repeated operations with the same matrix, prepare a
`oneMKL.oneSparseMatrixHandle` instead, which `sparse_optimize_*!` returns, and pass it to the
`sparse_*!` wrappers in place of the matrix. The handle refers to the buffers of the matrix,
which must not be modified, resized or freed until the handle is closed:

```julia
P = oneMKL.sparse_optimize_gemv!('N', dA)
try
    oneMKL.sparse_gemv!('N', 1f0, P, x, 0f0, y)
finally
    close(P)
end
```

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

