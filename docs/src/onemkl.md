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
`oneSparseMatrixCOO`. They are aliases of GPUArrays' immutable, storage-parametric
canonical formats with `oneVector` components. Compressed pointers and coordinates
may have independent integer widths. Raw buffer constructors borrow their storage;
structural mutation requires exclusive, resizable components. The
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

Any element type can be stored in these matrices. GPUArrays centrally selects portable
or oneMKL multiplication; `GPUArrays.Portable()` and `GPUArrays.Vendor()` explicitly
select a path through `GPUArrays.spmv!`/`spmm!`. oneMKL requires matching `Float32`,
`Float64`, `ComplexF32` or `ComplexF64` values and uniform `Int32` or `Int64` indices.
Mixed-width inputs remain usable through portable operations. Arrays contain no native
handle. Ordinary vendor calls acquire and dispose a scoped resource.

Repeated optimized operations use the object returned by `sparse_optimize_*`:

```julia
P = oneMKL.sparse_optimize_gemv!('N', dA)
y = similar(x)
try
    oneMKL.sparse_gemv!('N', 1f0, P, x, 0f0, y)
finally
    close(P)
end
```

Discarding `P` and subsequently multiplying `dA` does not reuse its analysis. A prepared
operation is restricted to its originating task, stream, context, device and recorded
parameters. It borrows fixed pattern storage: do not edit indices, resize, rebind or
explicitly free any source component until disposal succeeds. In-place value updates
between completed calls are supported. The native handle binds an independent numerical
snapshot; every call compares current values bitwise on-device, requiring O(nnz) work and
a host-visible decision. Unchanged values reuse native analysis. Changes rebuild it using
a new snapshot, briefly retaining both analyses and snapshots. Do not modify the source
concurrently. Failed disposal retains borrowing restrictions; an uncertain native release
quarantines the complete resource rather than retrying destruction.

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

