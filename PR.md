The BLAS-1 methods (`norm`, `dot`, `BLAS.asum`, `axpy!`, `axpby!`, `rmul!`, `rotate!`, `reflect!`) and the storage-level `mul!` and triangular methods pass any strided view to oneMKL with `incx = stride(x, 1)` and `lda = max(1, stride(A, 2))`, without checking that oneMKL can address that layout. Reversed views get a negative increment, which BLAS walks backwards from the *lowest* address while the view's pointer is at its *highest* element, so the result is silently wrong. Views that skip rows get passed as `length(x)` elements with a single increment, which reads and writes outside the view. Reversed columns give a negative `lda`, and oneMKL's `invalid_argument` escapes the support library's C interface and aborts the process:

```julia
x = oneArray(Float32[3, 4]); r = view(x, 2:-1:1)
norm(r)                    # 4.0, expected 5.0
rmul!(r, 2f0); x           # [3, 8], expected [6, 8]
mul!(C, 'N', 'N', view(A, :, 4:-1:1), B, true, false)
# terminate called after throwing an instance of 'oneapi::mkl::invalid_argument'
```

The triangular case is reachable through public API, e.g. `lmul!(UpperTriangular(A), view(x, n:-1:1))`.

This ports JuliaGPU/CUDA.jl#3411 and JuliaGPU/CUDA.jl#3409. The level-1 wrappers now take strided vectors or dense arrays only, and pass a pointer to the lowest address together with the signed increment, like LinearAlgebra's `vec_pointer_stride`; nrm2, asum, scal, iamax and iamin use the absolute increment since the order doesn't matter (oneMKL rejects a negative one for iamax/iamin, so their index is flipped back). The LinearAlgebra BLAS-1 methods fall back to broadcasts for strided arrays that aren't vectors or dense. Storage-level `mul!` only calls oneMKL when every matrix has a unit row stride and a leading dimension of at least its row count, and every vector a positive stride, falling back to `GPUArrays.generic_matmatmul!` otherwise as it already did for unsupported element types. The triangular methods run on contiguous copies and copy the result back.

Not covered: the level-2 wrappers (`gemv!`, `symv!`, ...) still take raw strides when called directly, as in CUDA.jl, and the support library should turn `oneapi::mkl::exception` into an error code instead of aborting, which needs a JLL rebuild.

Tested on an Iris Xe (Julia 1.12): the reproducers from both issues fail on main and pass here, and the new tests cover reversed and row-skipping views for every BLAS-1 method, the storage-level `mul!` paths (gemm, gemv, symm, hemm, symv, hemv) and the triangular products and solves.

Fixes #712
Fixes #700
