# oneMKL sparse operations
#
# oneMKL describes a sparse matrix with a handle that refers to its buffers. The `sparse_*!`
# operations below create a handle for the duration of the call, or take an explicitly
# prepared `oneSparseMatrixHandle` in place of the matrix, which `sparse_optimize_*!` can
# optimize for repeated operations. The support library completes every native call before
# returning, so a handle can be released as soon as the operation returns.

check_sparse_status(status, operation) =
    iszero(status) || error("oneMKL sparse operation $operation failed (status $status)")

# The CSC setters are exported by the support library unconditionally, but compile to
# unsupported stubs when the library was built against an oneMKL without the wide sparse
# API (2025.2 and older); turn that into a clear error instead of a silent no-op.
csc_supported() = version() >= v"2025.3"
function _check_csc_support()
    csc_supported() && return
    error(
        "oneMKL sparse operations on a oneSparseMatrixCSC require a support library built " *
        "against oneMKL 2025.3 or later; the loaded one was built against oneMKL $(version())."
    )
end

# oneMKL operations are only defined for these element and index types; matrices with other
# types can still be created and used with the generic GPUArrays functionality.
const onemklSparseFloat = onemklFloat
const onemklSparseInt = Union{Int32, Int64}


## matrix handles

"""
    oneSparseMatrixHandle(A::oneAbstractSparseMatrix)

A oneMKL handle describing the sparse matrix `A`, for repeated operations. The `sparse_*!`
operations accept it in place of `A`, and the `sparse_optimize_*!` functions return one
(or optimize the one they are given) so that oneMKL can analyze `A` once:

```julia
P = oneMKL.sparse_optimize_gemv!('N', A)
try
    for x in xs
        oneMKL.sparse_gemv!('N', 1f0, P, x, 0f0, y)
    end
finally
    close(P)
end
```

The handle refers to the buffers of `A`, which must not be modified, resized or freed
until the handle is closed with `close`. Handles that are not closed are released after
they are garbage collected, the next time a handle is created.
"""
mutable struct oneSparseMatrixHandle{M <: oneAbstractSparseMatrix, Q}
    handle::matrix_handle_t    # C_NULL for empty matrices, and once released
    const matrix::M            # keeps the buffers alive
    const queue::Q             # the queue of the buffers, on which the handle is used
end

sparse_matrix(A::oneAbstractSparseMatrix) = A
sparse_matrix(P::oneSparseMatrixHandle) = P.matrix
Base.size(P::oneSparseMatrixHandle, args...) = size(P.matrix, args...)
SparseArrays.nnz(P::oneSparseMatrixHandle) = nnz(P.matrix)

const SparseOperand{M} = Union{M, oneSparseMatrixHandle{<:M}}

# Finalizers run on the GC thread, but releasing a handle submits work to its queue, which
# must not be used concurrently from another thread (#585). Finalized handles are therefore
# queued, keeping their matrix alive, and released by the next task that creates a handle.
const _deferred_sparse_handles = Tuple{matrix_handle_t, Any, Any}[]
const _deferred_sparse_handles_lock = Threads.SpinLock()  # (finalizers must not yield)

function _defer_release(P::oneSparseMatrixHandle)
    P.handle == C_NULL && return
    lock(_deferred_sparse_handles_lock) do
        push!(_deferred_sparse_handles, (P.handle, P.queue, P.matrix))
    end
    return
end

function flush_deferred_sparse_releases()
    pending = lock(_deferred_sparse_handles_lock) do
        isempty(_deferred_sparse_handles) && return nothing
        handles = copy(_deferred_sparse_handles)
        empty!(_deferred_sparse_handles)
        return handles
    end
    pending === nothing && return
    for (handle, queue, _) in pending
        try
            release_matrix_handle(handle, queue)
        catch err
            @error "Error releasing a oneMKL sparse matrix handle" exception = (err, catch_backtrace())
        end
    end
    return
end

release_matrix_handle(handle::matrix_handle_t, queue) =
    check_sparse_status(onemklXsparse_release_matrix_handle(sycl_queue(queue), Ref(handle)),
                        "release_matrix_handle")

# matrices without stored entries never get a handle: the operations short-circuit instead
_mkl_empty(A::oneAbstractSparseMatrix) = nnz(A) == 0 || any(iszero, size(A))
_mkl_empty(P::oneSparseMatrixHandle) = _mkl_empty(P.matrix)

# a new handle for `A`, on the queue of its buffers
function new_matrix_handle(A::oneAbstractSparseMatrix)
    flush_deferred_sparse_releases()
    Support._check_sparse_abi()
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    handle = Ref{matrix_handle_t}(C_NULL)
    check_sparse_status(onemklXsparse_init_matrix_handle(handle), "init_matrix_handle")
    try
        _set_matrix_data!(A, handle[], queue)
    catch
        release_matrix_handle(handle[], queue)
        rethrow()
    end
    return handle[], queue
end

function oneSparseMatrixHandle(A::oneAbstractSparseMatrix)
    handle, queue = if _mkl_empty(A)
        matrix_handle_t(C_NULL), global_queue(context(A.nzVal), device(A.nzVal))
    else
        new_matrix_handle(A)
    end
    P = oneSparseMatrixHandle(handle, A, queue)
    return finalizer(_defer_release, P)
end

# (so that it can be passed to the support library's functions)
Base.unsafe_convert(::Type{matrix_handle_t}, P::oneSparseMatrixHandle) = P.handle

function sparse_matrix_handle(A::oneAbstractSparseMatrix)
    Base.depwarn("`sparse_matrix_handle(A)` is deprecated, use `oneSparseMatrixHandle(A)` " *
                 "and close it when done.", :sparse_matrix_handle)
    return oneSparseMatrixHandle(A)
end

function Base.close(P::oneSparseMatrixHandle)
    P.handle == C_NULL && return
    handle = P.handle
    P.handle = C_NULL
    release_matrix_handle(handle, P.queue)
    return
end

# The handle for an operation on `A`, and the queue to run it on: a new one for a matrix,
# released by `release_handle` after the operation.
acquire_handle(A::oneAbstractSparseMatrix) = new_matrix_handle(A)
function acquire_handle(P::oneSparseMatrixHandle)
    P.handle == C_NULL && throw(ArgumentError("the oneMKL sparse matrix handle has been closed"))
    return P.handle, P.queue
end
release_handle(::oneAbstractSparseMatrix, handle, queue) = release_matrix_handle(handle, queue)
release_handle(::oneSparseMatrixHandle, handle, queue) = nothing

function _set_matrix_data!(A::oneAbstractSparseMatrix, handle::matrix_handle_t, queue)
    throw(
        ArgumentError(
            "oneMKL sparse operations only support Float32, Float64, ComplexF32 and ComplexF64 " *
                "matrices with Int32 or Int64 indices, got $(typeof(A))"
        )
    )
end

for (fname, elty, intty) in ((:onemklSsparse_set_csr_data   , :Float32   , :Int32),
                             (:onemklSsparse_set_csr_data_64, :Float32   , :Int64),
                             (:onemklDsparse_set_csr_data   , :Float64   , :Int32),
                             (:onemklDsparse_set_csr_data_64, :Float64   , :Int64),
                             (:onemklCsparse_set_csr_data   , :ComplexF32, :Int32),
                             (:onemklCsparse_set_csr_data_64, :ComplexF32, :Int64),
                             (:onemklZsparse_set_csr_data   , :ComplexF64, :Int32),
                             (:onemklZsparse_set_csr_data_64, :ComplexF64, :Int64))
    @eval begin
        function _set_matrix_data!(A::oneSparseMatrixCSR{$elty, $intty}, handle::matrix_handle_t, queue)
            m, n = size(A)
            status = $fname(sycl_queue(queue), handle, m, n, nnz(A), 'O', A.rowPtr, A.colVal, A.nzVal)
            check_sparse_status(status, $(string(fname)))
        end

        function _set_matrix_data!(A::oneSparseMatrixCSC{$elty, $intty}, handle::matrix_handle_t, queue)
            _check_csc_support()
            m, n = size(A)
            # CSC of A is CSR of Aᵀ
            status = $fname(sycl_queue(queue), handle, n, m, nnz(A), 'O', A.colPtr, A.rowVal, A.nzVal)
            check_sparse_status(status, $(string(fname)))
        end
    end
end

for (fname, elty, intty) in ((:onemklSsparse_set_coo_data   , :Float32   , :Int32),
                             (:onemklSsparse_set_coo_data_64, :Float32   , :Int64),
                             (:onemklDsparse_set_coo_data   , :Float64   , :Int32),
                             (:onemklDsparse_set_coo_data_64, :Float64   , :Int64),
                             (:onemklCsparse_set_coo_data   , :ComplexF32, :Int32),
                             (:onemklCsparse_set_coo_data_64, :ComplexF32, :Int64),
                             (:onemklZsparse_set_coo_data   , :ComplexF64, :Int32),
                             (:onemklZsparse_set_coo_data_64, :ComplexF64, :Int64))
    @eval begin
        function _set_matrix_data!(A::oneSparseMatrixCOO{$elty, $intty}, handle::matrix_handle_t, queue)
            m, n = size(A)
            status = $fname(sycl_queue(queue), handle, m, n, nnz(A), 'O', A.rowInd, A.colInd, A.nzVal)
            check_sparse_status(status, $(string(fname)))
        end
    end
end


## helpers

# `mul!` ignores the output when beta is zero, and callers pass uninitialized memory (e.g.
# from `similar`), but oneMKL computes beta * y regardless in some cases (gemv with a
# transposed CSR matrix, and gemm on older releases), letting NaNs into the result (#703).
function _clear_ignored_output!(beta::Number, y::AbstractArray)
    iszero(beta) && fill!(y, zero(eltype(y)))
    return y
end

# result of op(A) * x for an empty A
function _scale_output!(beta::Number, y::AbstractArray)
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    else
        y .*= beta
    end
    return y
end

# an empty triangular matrix is the identity if it has a unit diagonal, and singular otherwise
# (unless it has no rows or columns, in which case there is nothing to compute)
_empty_triangular_is_identity(diag::Char, A) = diag == 'U' || any(iszero, size(A))

# result of op(A) * x for an empty triangular A
function _empty_trmv!(diag::Char, alpha::Number, x::AbstractVector, beta::Number, y::AbstractVector)
    _scale_output!(beta, y)
    diag == 'U' && (y .+= alpha .* x)
    return y
end

# result of op(A) \ alpha * op(X) for an empty triangular A
function _empty_trsm!(diag::Char, A, alpha::Number, transX::Char, X::AbstractArray, Y::AbstractArray)
    _empty_triangular_is_identity(diag, A) || throw(LinearAlgebra.SingularException(1))
    opX = transX == 'N' ? X : transX == 'T' ? permutedims(X) : conj.(permutedims(X))
    Y .= alpha .* opX
    return Y
end


## operations

for SparseMatrix in (:oneSparseMatrixCSR, :oneSparseMatrixCOO)
    for (fname, elty) in ((:onemklSsparse_gemv, :Float32),
                          (:onemklDsparse_gemv, :Float64),
                          (:onemklCsparse_gemv, :ComplexF32),
                          (:onemklZsparse_gemv, :ComplexF64))
        @eval begin
            function sparse_gemv!(trans::Char,
                                  alpha::Number,
                                  A::SparseOperand{<:$SparseMatrix{$elty}},
                                  x::oneStridedVector{$elty},
                                  beta::Number,
                                  y::oneStridedVector{$elty})

                _mkl_empty(A) && return _scale_output!(beta, y)
                _clear_ignored_output!(beta, y)
                handle, queue = acquire_handle(A)
                try
                    check_sparse_status($fname(sycl_queue(queue), trans, alpha, handle, x, beta, y), $(string(fname)))
                finally
                    release_handle(A, handle, queue)
                end
                y
            end
        end
    end

    @eval begin
        function sparse_optimize_gemv!(trans::Char, A::oneSparseMatrixHandle{<:$SparseMatrix})
            _mkl_empty(A) && return A
            check_sparse_status(onemklXsparse_optimize_gemv(sycl_queue(A.queue), trans, A.handle), "optimize_gemv")
            return A
        end
    end
end

for SparseMatrix in (:oneSparseMatrixCSC,)
    # CSC(A) is represented by storing CSR(A^T). Map operations accordingly:
    #  - trans = 'N': want A*x -> use op(S)='T' with S=A^T.
    #  - trans = 'T': want A^T*x -> use op(S)='N' with S=A^T.
    #  - trans = 'C': want A^H*x.
    #      * For real eltypes, A^H == A^T -> use op(S)='N'.
    #      * For complex eltypes, we cannot express A^H using a single op(S).
    #        Use identity: conj(y_new) = conj(alpha) * A * conj(x) + conj(beta) * conj(y)
    #        and compute with op(S)='T' (since S^T = A), conjugating x and y around the call.
    for (fname, elty) in ((:onemklSsparse_gemv, :Float32),
                          (:onemklDsparse_gemv, :Float64))
        @eval begin
            function sparse_gemv!(trans::Char,
                                  alpha::Number,
                                  A::SparseOperand{<:$SparseMatrix{$elty}},
                                  x::oneStridedVector{$elty},
                                  beta::Number,
                                  y::oneStridedVector{$elty})

                _mkl_empty(A) && return _scale_output!(beta, y)
                m, n = size(A)
                if m != 0 && n != 0
                    _clear_ignored_output!(beta, y)
                    handle, queue = acquire_handle(A)
                    try
                        check_sparse_status($fname(sycl_queue(queue), flip_trans(trans), alpha, handle, x, beta, y), $(string(fname)))
                    finally
                        release_handle(A, handle, queue)
                    end
                end
                y
            end
        end
    end

    # Special handling for CSC matrices since they are stored as transposed CSR
    for (fname, elty) in (
            (:onemklCsparse_gemv, :ComplexF32),
            (:onemklZsparse_gemv, :ComplexF64),
        )
        @eval begin
            function sparse_gemv!(
                    trans::Char,
                    alpha::Number,
                    A::SparseOperand{<:$SparseMatrix{$elty}},
                    x::oneStridedVector{$elty},
                    beta::Number,
                    y::oneStridedVector{$elty}
                )

                _mkl_empty(A) && return _scale_output!(beta, y)
                # Compute A^H*x via identity:
                #   conj(y_new) = conj(alpha) * (A^T) * conj(x) + conj(beta) * conj(y)
                # Since S=A^T and op='N' computes S*x = A^T*x, we can realize this with one call.

                if trans == 'C'
                    y .= conj.(y)
                    x .= conj.(x)
                    alpha = conj(alpha)
                    beta = conj(beta)
                end

                _clear_ignored_output!(beta, y)
                handle, queue = acquire_handle(A)
                try
                    check_sparse_status($fname(sycl_queue(queue), flip_trans(trans), alpha, handle, x, beta, y), $(string(fname)))
                finally
                    release_handle(A, handle, queue)
                end

                if trans == 'C'
                    y .= conj.(y)
                    # Restore x
                    x .= conj.(x)
                end
                return y
            end
        end
    end
    @eval begin
        function sparse_optimize_gemv!(trans::Char, A::oneSparseMatrixHandle{<:$SparseMatrix})
            _mkl_empty(A) && return A
            # complex 'C' case is implemented using op='N' on S=A^T with conjugation trick
            check_sparse_status(onemklXsparse_optimize_gemv(sycl_queue(A.queue), flip_trans(trans), A.handle), "optimize_gemv")
            return A
        end
    end
end

for (fname, elty) in ((:onemklSsparse_gemm, :Float32),
        (:onemklDsparse_gemm, :Float64),
        (:onemklCsparse_gemm, :ComplexF32),
        (:onemklZsparse_gemm, :ComplexF64),
    )
    @eval begin
        function sparse_gemm!(transa::Char,
                transb::Char,
                alpha::Number,
                A::SparseOperand{<:oneSparseMatrixCSR{$elty}},
                B::oneStridedMatrix{$elty},
                beta::Number,
                C::oneStridedMatrix{$elty}
            )

            transb == 'N' || throw(ArgumentError("oneMKL's sparse gemm requires a non-transposed dense operand (transb = 'N')"))
            _mkl_empty(A) && return _scale_output!(beta, C)
            mB, nB = size(B)
            mC, nC = size(C)
            (nB != nC) && (transb == 'N') && throw(ArgumentError("B and C must have the same number of columns."))
            (mB != nC) && (transb != 'N') && throw(ArgumentError("Bᵀ and C must have the same number of columns."))
            nrhs = size(B, 2)
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            _clear_ignored_output!(beta, C)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), 'C', transa, transb, alpha, handle, B, nrhs, ldb, beta, C, ldc), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            C
        end
    end
end

function sparse_optimize_gemm!(trans::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSR})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_gemm(sycl_queue(A.queue), trans, A.handle), "optimize_gemm")
    return A
end

function sparse_optimize_gemm!(trans::Char, transB::Char, nrhs::Int, A::oneSparseMatrixHandle{<:oneSparseMatrixCSR})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_gemm_advanced(sycl_queue(A.queue), 'C', trans, transB, A.handle, nrhs), "optimize_gemm_advanced")
    return A
end

for (fname, elty) in ((:onemklSsparse_gemm, :Float32),
                      (:onemklDsparse_gemm, :Float64))
    @eval begin
        function sparse_gemm!(transa::Char,
                              transb::Char,
                              alpha::Number,
                              A::SparseOperand{<:oneSparseMatrixCSC{$elty}},
                              B::oneStridedMatrix{$elty},
                              beta::Number,
                              C::oneStridedMatrix{$elty})

            transb == 'N' || throw(ArgumentError("oneMKL's sparse gemm requires a non-transposed dense operand (transb = 'N')"))
            _mkl_empty(A) && return _scale_output!(beta, C)
            mB, nB = size(B)
            mC, nC = size(C)
            (nB != nC) && (transb == 'N') && throw(ArgumentError("B and C must have the same number of columns."))
            (mB != nC) && (transb != 'N') && throw(ArgumentError("Bᵀ and C must have the same number of columns."))
            nrhs = size(B, 2)
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            _clear_ignored_output!(beta, C)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), 'C', flip_trans(transa), transb, alpha, handle, B, nrhs, ldb, beta, C, ldc), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            C
        end
    end
end

# Special handling for CSC matrices since they are stored as transposed CSR (S = A^T)
for (fname, elty) in (
        (:onemklCsparse_gemm, :ComplexF32),
        (:onemklZsparse_gemm, :ComplexF64),
    )
    @eval begin
        function sparse_gemm!(
                transa::Char,
                transb::Char,
                alpha::Number,
                A::SparseOperand{<:oneSparseMatrixCSC{$elty}},
                B::oneStridedMatrix{$elty},
                beta::Number,
                C::oneStridedMatrix{$elty}
            )

            transb == 'N' || throw(ArgumentError("oneMKL's sparse gemm requires a non-transposed dense operand (transb = 'N')"))
            _mkl_empty(A) && return _scale_output!(beta, C)
            # Map op(A) to op(S) where S = A^T stored as CSR in the handle
            # transa: 'N' -> op(S)='T'; 'T' -> op(S)='N'; 'C' ->
            #   real: op(S)='N' (since A^H == A^T)
            #   complex: use conjugation identity on B and C with op(S)='N'

            mB, nB = size(B)
            mC, nC = size(C)
            (nB != nC) && (transb == 'N') && throw(ArgumentError("B and C must have the same number of columns."))
            (mB != nC) && (transb != 'N') && throw(ArgumentError("Bᵀ and C must have the same number of columns."))
            nrhs = size(B, 2)
            ldb = max(1, stride(B, 2))
            ldc = max(1, stride(C, 2))

            # Use identity: conj(C_new) = conj(alpha) * S * conj(opB(B)) + conj(beta) * conj(C)
            # Prepare conj(C) in-place and conj(B) into a temporary if needed

            # Determine how to supply opB under conjugation
            # - transb == 'N': pass transb='N' and use conj(B)
            # - transb == 'T': pass transb='T' and use conj(B)
            # - transb == 'C': since conj(B^H) = B^T, pass transb='T' and use B as-is
            local transb_eff
            local Beff
            if transa == 'C'
                C .= conj.(C)
                alpha = conj(alpha)
                beta = conj(beta)
                if transb == 'N'
                    transb_eff = 'N'
                    # Beff = similar(B)
                    B .= conj.(B)
                elseif transb == 'T'
                    transb_eff = 'T'
                    # Beff = similar(B)
                    B .= conj.(B)
                else
                    # transb == 'C'
                    transb_eff = 'T'
                end
            else
                transb_eff = transb
            end

            _clear_ignored_output!(beta, C)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), 'C', flip_trans(transa), transb_eff, alpha, handle, B, nrhs, ldb, beta, C, ldc), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end

            # Undo conjugation to obtain C_new
            if transa == 'C'
                C .= conj.(C)
                if transb == 'N' || transb == 'T'
                    # Restore B
                    B .= conj.(B)
                end
            end
            return C
        end
    end
end

function sparse_optimize_gemm!(trans::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSC})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_gemm(sycl_queue(A.queue), flip_trans(trans), A.handle), "optimize_gemm")
    return A
end

function sparse_optimize_gemm!(trans::Char, transB::Char, nrhs::Int, A::oneSparseMatrixHandle{<:oneSparseMatrixCSC})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_gemm_advanced(sycl_queue(A.queue), 'C', flip_trans(trans), transB, A.handle, nrhs), "optimize_gemm_advanced")
    return A
end

for (fname, elty) in ((:onemklSsparse_symv, :Float32),
                      (:onemklDsparse_symv, :Float64),
                      (:onemklCsparse_symv, :ComplexF32),
                      (:onemklZsparse_symv, :ComplexF64))
    @eval begin
        function sparse_symv!(uplo::Char,
                              alpha::Number,
                              A::SparseOperand{<:oneSparseMatrixCSR{$elty}},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _scale_output!(beta, y)
            _clear_ignored_output!(beta, y)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), uplo, alpha, handle, x, beta, y), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            y
        end
    end
end

for (fname, elty) in ((:onemklSsparse_symv, :Float32),
        (:onemklDsparse_symv, :Float64),
        (:onemklCsparse_symv, :ComplexF32),
        (:onemklZsparse_symv, :ComplexF64),
    )
    @eval begin
        function sparse_symv!(uplo::Char,
                              alpha::Number,
                              A::SparseOperand{<:oneSparseMatrixCSC{$elty}},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _scale_output!(beta, y)
            _clear_ignored_output!(beta, y)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), flip_uplo(uplo), alpha, handle, x, beta, y), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            y
        end
    end
end

for (fname, elty) in ((:onemklSsparse_trmv, :Float32),
                      (:onemklDsparse_trmv, :Float64),
                      (:onemklCsparse_trmv, :ComplexF32),
                      (:onemklZsparse_trmv, :ComplexF64))
    @eval begin
        function sparse_trmv!(uplo::Char,
                              trans::Char,
                              diag::Char,
                              alpha::Number,
                              A::SparseOperand{<:oneSparseMatrixCSR{$elty}},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _empty_trmv!(diag, alpha, x, beta, y)
            _clear_ignored_output!(beta, y)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), uplo, trans, diag, alpha, handle, x, beta, y), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            y
        end
    end
end

function sparse_optimize_trmv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSR})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_trmv(sycl_queue(A.queue), uplo, trans, diag, A.handle), "optimize_trmv")
    return A
end

# Special handling for CSC matrices since they are stored as transposed CSR
for (fname, elty) in (
        (:onemklSsparse_trmv, :Float32),
        (:onemklDsparse_trmv, :Float64),
        (:onemklCsparse_trmv, :ComplexF32),
        (:onemklZsparse_trmv, :ComplexF64),
    )
    @eval begin
        function sparse_trmv!(
                uplo::Char,
                trans::Char,
                diag::Char,
                alpha::Number,
                A::SparseOperand{<:oneSparseMatrixCSC{$elty}},
                x::oneStridedVector{$elty},
                beta::Number,
                y::oneStridedVector{$elty}
            )

            _mkl_empty(A) && return _empty_trmv!(diag, alpha, x, beta, y)
            # Intel oneAPI sparse trmv only supports nontrans operations.
            # Since CSC(A) is stored as CSR(A^T), we cannot map CSC operations
            # to CSR operations for triangular operations without transpose support.
            throw(
                ArgumentError(
                    "sparse_trmv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                        "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                        "Convert to oneSparseMatrixCSR format instead."
                )
            )
            _clear_ignored_output!(beta, y)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), uplo, flip_trans(trans), diag, alpha, handle, x, beta, y), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            return y
        end
    end
end

function sparse_optimize_trmv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSC})
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trmv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    check_sparse_status(onemklXsparse_optimize_trmv(sycl_queue(A.queue), uplo, flip_trans(trans), diag, A.handle), "optimize_trmv")
    return A
end

for (fname, elty) in ((:onemklSsparse_trsv, :Float32),
                      (:onemklDsparse_trsv, :Float64),
                      (:onemklCsparse_trsv, :ComplexF32),
                      (:onemklZsparse_trsv, :ComplexF64))
    @eval begin
        function sparse_trsv!(uplo::Char,
                              trans::Char,
                              diag::Char,
                              alpha::Number,
                              A::SparseOperand{<:oneSparseMatrixCSR{$elty}},
                              x::oneStridedVector{$elty},
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _empty_trsm!(diag, A, alpha, 'N', x, y)
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), uplo, trans, diag, alpha, handle, x, y), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            y
        end
    end
end

function sparse_optimize_trsv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSR})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_trsv(sycl_queue(A.queue), uplo, trans, diag, A.handle), "optimize_trsv")
    return A
end

for (fname, elty) in (
        (:onemklSsparse_trsv, :Float32),
        (:onemklDsparse_trsv, :Float64),
        (:onemklCsparse_trsv, :ComplexF32),
        (:onemklZsparse_trsv, :ComplexF64),
    )
    @eval begin
        function sparse_trsv!(
                uplo::Char,
                trans::Char,
                diag::Char,
                alpha::Number,
                A::SparseOperand{<:oneSparseMatrixCSC{$elty}},
                x::oneStridedVector{$elty},
                y::oneStridedVector{$elty}
            )

            _mkl_empty(A) && return _empty_trsm!(diag, A, alpha, 'N', x, y)
            throw(
                ArgumentError(
                    "sparse_trsv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                        "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                        "Convert to oneSparseMatrixCSR format instead."
                )
            )
            check_sparse_status(onemklXsparse_optimize_trsv(sycl_queue(A.queue), uplo, flip_trans(trans), diag, A.handle), "optimize_trsv")
            return A
        end
    end
end

function sparse_optimize_trsv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSC})
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trsv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    check_sparse_status(onemklXsparse_optimize_trsv(sycl_queue(A.queue), uplo, flip_trans(trans), diag, A.handle), "optimize_trsv")
    return A
end

for (fname, elty) in ((:onemklSsparse_trsm, :Float32),
                      (:onemklDsparse_trsm, :Float64),
                      (:onemklCsparse_trsm, :ComplexF32),
                      (:onemklZsparse_trsm, :ComplexF64))
    @eval begin
        function sparse_trsm!(uplo::Char,
                              transA::Char,
                              transX::Char,
                              diag::Char,
                              alpha::Number,
                              A::SparseOperand{<:oneSparseMatrixCSR{$elty}},
                              X::oneStridedMatrix{$elty},
                              Y::oneStridedMatrix{$elty})

            _mkl_empty(A) && return _empty_trsm!(diag, A, alpha, transX, X, Y)
            mX, nX = size(X)
            mY, nY = size(Y)
            (mX != mY) && (transX == 'N') && throw(ArgumentError("X and Y must have the same number of rows."))
            (nX != nY) && (transX == 'N') && throw(ArgumentError("X and Y must have the same number of columns."))
            (nX != mY) && (transX != 'N') && throw(ArgumentError("Xᵀ and Y must have the same number of rows."))
            (mX != nY) && (transX != 'N') && throw(ArgumentError("Xᵀ and Y must have the same number of columns."))
            nrhs = size(X, 2)
            ldx = max(1,stride(X,2))
            ldy = max(1,stride(Y,2))
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), 'C', transA, transX, uplo, diag, alpha, handle, X, nrhs, ldx, Y, ldy), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            Y
        end
    end
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSR})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_trsm(sycl_queue(A.queue), uplo, trans, diag, A.handle), "optimize_trsm")
    return A
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, nrhs::Int, A::oneSparseMatrixHandle{<:oneSparseMatrixCSR})
    _mkl_empty(A) && return A
    check_sparse_status(onemklXsparse_optimize_trsm_advanced(sycl_queue(A.queue), 'C', uplo, trans, diag, A.handle, nrhs), "optimize_trsm_advanced")
    return A
end

# Only transA = 'N' is supported with oneSparseMatrixCSR.
# We can't use any trick to support sparse "trsm" for oneSparseMatrixCSC.
for (fname, elty) in (
        (:onemklSsparse_trsm, :Float32),
        (:onemklDsparse_trsm, :Float64),
        (:onemklCsparse_trsm, :ComplexF32),
        (:onemklZsparse_trsm, :ComplexF64),
    )
    @eval begin
        function sparse_trsm!(
                uplo::Char,
                transA::Char,
                transX::Char,
                diag::Char,
                alpha::Number,
                A::SparseOperand{<:oneSparseMatrixCSC{$elty}},
                X::oneStridedMatrix{$elty},
                Y::oneStridedMatrix{$elty}
            )

            _mkl_empty(A) && return _empty_trsm!(diag, A, alpha, transX, X, Y)
            # Intel oneAPI sparse trsm only supports nontrans operations for the matrix A.
            # Since CSC(A) is stored as CSR(A^T), we cannot map CSC operations
            # to CSR operations for triangular solve operations without transpose support.
            throw(
                ArgumentError(
                    "sparse_trsm! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                        "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                        "Convert to oneSparseMatrixCSR format instead."
                )
            )

            mX, nX = size(X)
            mY, nY = size(Y)
            (mX != mY) && (transX == 'N') && throw(ArgumentError("X and Y must have the same number of rows."))
            (nX != nY) && (transX == 'N') && throw(ArgumentError("X and Y must have the same number of columns."))
            (nX != mY) && (transX != 'N') && throw(ArgumentError("Xᵀ and Y must have the same number of rows."))
            (mX != nY) && (transX != 'N') && throw(ArgumentError("Xᵀ and Y must have the same number of columns."))
            nrhs = size(X, 2)
            ldx = max(1, stride(X, 2))
            ldy = max(1, stride(Y, 2))
            handle, queue = acquire_handle(A)
            try
                check_sparse_status($fname(sycl_queue(queue), 'C', flip_trans(transA), transX, uplo, diag, alpha, handle, X, nrhs, ldx, Y, ldy), $(string(fname)))
            finally
                release_handle(A, handle, queue)
            end
            return Y
        end
    end
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixHandle{<:oneSparseMatrixCSC})
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trsm! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    check_sparse_status(onemklXsparse_optimize_trsm(sycl_queue(A.queue), uplo, trans, diag, A.handle), "optimize_trsm")
    return A
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, nrhs::Int, A::oneSparseMatrixHandle{<:oneSparseMatrixCSC})
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trsm! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    check_sparse_status(onemklXsparse_optimize_trsm_advanced(sycl_queue(A.queue), 'C', uplo, trans, diag, A.handle, nrhs), "optimize_trsm_advanced")
    return A
end


# Optimizing a matrix prepares a handle for it, which the operations above accept in place of
# the matrix, and which must be closed (see `oneSparseMatrixHandle`).
function prepare_handle(f, A::oneAbstractSparseMatrix)
    P = oneSparseMatrixHandle(A)
    try
        return f(P)
    catch
        close(P)
        rethrow()
    end
end
sparse_optimize_gemv!(trans::Char, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_gemv!(trans, P), A)
sparse_optimize_gemm!(trans::Char, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_gemm!(trans, P), A)
sparse_optimize_gemm!(trans::Char, transB::Char, nrhs::Int, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_gemm!(trans, transB, nrhs, P), A)
sparse_optimize_trmv!(uplo::Char, trans::Char, diag::Char, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_trmv!(uplo, trans, diag, P), A)
sparse_optimize_trsv!(uplo::Char, trans::Char, diag::Char, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_trsv!(uplo, trans, diag, P), A)
sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_trsm!(uplo, trans, diag, P), A)
sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, nrhs::Int, A::oneAbstractSparseMatrix) =
    prepare_handle(P -> sparse_optimize_trsm!(uplo, trans, diag, nrhs, P), A)
