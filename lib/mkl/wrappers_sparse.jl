# Deferred release queue for sparse matrix handles.
# Finalizers run on the GC thread, but onemklXsparse_release_matrix_handle submits
# work to the SYCL queue. Using the same queue from the GC thread and the main thread
# concurrently is not safe and causes ZE_RESULT_ERROR_DEVICE_LOST / ZE_RESULT_ERROR_UNKNOWN.
# Instead, finalizers push handles here and they are released on the main thread.
const _deferred_sparse_handles = Vector{matrix_handle_t}()
const _deferred_sparse_handles_lock = ReentrantLock()

function _defer_release(handle::matrix_handle_t)
    return lock(_deferred_sparse_handles_lock) do
        push!(_deferred_sparse_handles, handle)
    end
end

function sparse_release_matrix_handle(A::oneAbstractSparseMatrix)
    handle = A.handle
    handle === nothing || _defer_release(handle)
    return
end

function flush_deferred_sparse_releases()
    handles = lock(_deferred_sparse_handles_lock) do
        if isempty(_deferred_sparse_handles)
            return matrix_handle_t[]
        end
        h = copy(_deferred_sparse_handles)
        empty!(_deferred_sparse_handles)
        return h
    end
    isempty(handles) && return
    dev = device()
    ctx = context()
    queue = global_queue(ctx, dev)
    for handle in handles
        try
            handle_ptr = Ref{matrix_handle_t}(handle)
            onemklXsparse_release_matrix_handle(sycl_queue(queue), handle_ptr)
        catch err
            @warn "Error releasing sparse matrix handle" exception = err
        end
    end
    return synchronize(queue)
end

# The CSC setters are exported by the support library unconditionally, but compile to
# unsupported stubs when the library was built against an oneMKL without the wide sparse
# API (2025.2 and older); turn that into a clear error instead of a silent no-op.
csc_supported() = version() >= v"2025.3"
function _check_csc_support()
    csc_supported() && return
    error(
        "oneSparseMatrixCSC requires a support library built against oneMKL 2025.3 or " *
        "later; the loaded one was built against oneMKL $(version())."
    )
end


## lazy oneMKL matrix handles

# oneMKL operations are only defined for these element and index types; matrices with other
# types can still be created and used with the generic GPUArrays functionality.
const onemklSparseFloat = onemklFloat
const onemklSparseInt = Union{Int32, Int64}

# matrices without stored entries never get a handle: the operations short-circuit instead
_mkl_empty(A::oneAbstractSparseMatrix) = nnz(A) == 0 || any(iszero, size(A))

# result of op(A) * x for an empty A
function _scale_output!(beta::Number, y::AbstractArray)
    if iszero(beta)
        fill!(y, zero(eltype(y)))
    else
        y .*= beta
    end
    return y
end

# drop the cached oneMKL handle, e.g. because the storage vectors are about to be replaced
function _invalidate_handle!(A::oneAbstractSparseMatrix)
    handle = A.handle
    handle === nothing && return A
    A.handle = nothing
    _defer_release(handle)
    return A
end

"""
    sparse_matrix_handle(A::oneAbstractSparseMatrix)

Return the oneMKL matrix handle describing `A`, creating it on first use. The handle refers to
the storage vectors of `A`, so those must not be replaced or resized afterwards (use `copyto!`,
which takes care of this, or construct a new matrix).
"""
function sparse_matrix_handle(A::oneAbstractSparseMatrix)
    handle = A.handle
    handle === nothing || return handle
    _mkl_empty(A) && throw(ArgumentError("cannot create a oneMKL handle for an empty sparse matrix"))

    flush_deferred_sparse_releases()
    Support._check_sparse_abi()
    handle_ptr = Ref{matrix_handle_t}()
    onemklXsparse_init_matrix_handle(handle_ptr)
    try
        _set_matrix_data!(A, handle_ptr[])
    catch
        _defer_release(handle_ptr[])
        rethrow()
    end
    A.handle = handle_ptr[]
    return handle_ptr[]
end

function _set_matrix_data!(A::oneAbstractSparseMatrix, handle::matrix_handle_t)
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
        function _set_matrix_data!(A::oneSparseMatrixCSR{$elty, $intty}, handle::matrix_handle_t)
            m, n = size(A)
            queue = global_queue(context(A.nzVal), device(A.nzVal))
            $fname(sycl_queue(queue), handle, m, n, nnz(A), 'O', A.rowPtr, A.colVal, A.nzVal)
            return
        end

        function _set_matrix_data!(A::oneSparseMatrixCSC{$elty, $intty}, handle::matrix_handle_t)
            _check_csc_support()
            m, n = size(A)
            queue = global_queue(context(A.nzVal), device(A.nzVal))
            # CSC of A is CSR of Aᵀ
            $fname(sycl_queue(queue), handle, n, m, nnz(A), 'O', A.colPtr, A.rowVal, A.nzVal)
            return
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
        function _set_matrix_data!(A::oneSparseMatrixCOO{$elty, $intty}, handle::matrix_handle_t)
            m, n = size(A)
            queue = global_queue(context(A.nzVal), device(A.nzVal))
            $fname(sycl_queue(queue), handle, m, n, nnz(A), 'O', A.rowInd, A.colInd, A.nzVal)
            return
        end
    end
end

function oneAPI.unsafe_free!(A::oneAbstractSparseMatrix)
    _invalidate_handle!(A)
    foreach(unsafe_free!, _storage(A))
    return
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
                                  A::$SparseMatrix{$elty},
                                  x::oneStridedVector{$elty},
                                  beta::Number,
                                  y::oneStridedVector{$elty})

                _mkl_empty(A) && return _scale_output!(beta, y)
                queue = global_queue(context(x), device(x))
                $fname(sycl_queue(queue), trans, alpha, sparse_matrix_handle(A), x, beta, y)
                y
            end
        end
    end

    @eval begin
        function sparse_optimize_gemv!(trans::Char, A::$SparseMatrix)
            _mkl_empty(A) && return A
            queue = global_queue(context(A.nzVal), device(A.nzVal))
            onemklXsparse_optimize_gemv(sycl_queue(queue), trans, sparse_matrix_handle(A))
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
                                  A::$SparseMatrix{$elty},
                                  x::oneStridedVector{$elty},
                                  beta::Number,
                                  y::oneStridedVector{$elty})

                _mkl_empty(A) && return _scale_output!(beta, y)
                queue = global_queue(context(x), device(x))
                m, n = size(A)
                if m != 0 && n != 0
                    $fname(sycl_queue(queue), flip_trans(trans), alpha, sparse_matrix_handle(A), x, beta, y)
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
                    A::$SparseMatrix{$elty},
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

                queue = global_queue(context(x), device(x))
                $fname(sycl_queue(queue), flip_trans(trans), alpha, sparse_matrix_handle(A), x, beta, y)

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
        function sparse_optimize_gemv!(trans::Char, A::$SparseMatrix)
            _mkl_empty(A) && return A
            # complex 'C' case is implemented using op='N' on S=A^T with conjugation trick
            queue = global_queue(context(A.nzVal), device(A.nzVal))
            onemklXsparse_optimize_gemv(sycl_queue(queue), flip_trans(trans), sparse_matrix_handle(A))
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
                A::oneSparseMatrixCSR{$elty},
                B::oneStridedMatrix{$elty},
                beta::Number,
                C::oneStridedMatrix{$elty}
            )

            _mkl_empty(A) && return _scale_output!(beta, C)
            mB, nB = size(B)
            mC, nC = size(C)
            (nB != nC) && (transb == 'N') && throw(ArgumentError("B and C must have the same number of columns."))
            (mB != nC) && (transb != 'N') && throw(ArgumentError("Bᵀ and C must have the same number of columns."))
            nrhs = size(B, 2)
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(C), device(C))
            $fname(sycl_queue(queue), 'C', transa, transb, alpha, sparse_matrix_handle(A), B, nrhs, ldb, beta, C, ldc)
            C
        end
    end
end

function sparse_optimize_gemm!(trans::Char, A::oneSparseMatrixCSR)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_gemm(sycl_queue(queue), trans, sparse_matrix_handle(A))
    return A
end

function sparse_optimize_gemm!(trans::Char, transB::Char, nrhs::Int, A::oneSparseMatrixCSR)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_gemm_advanced(sycl_queue(queue), 'C', trans, transB, sparse_matrix_handle(A), nrhs)
    return A
end

for (fname, elty) in ((:onemklSsparse_gemm, :Float32),
                      (:onemklDsparse_gemm, :Float64))
    @eval begin
        function sparse_gemm!(transa::Char,
                              transb::Char,
                              alpha::Number,
                              A::oneSparseMatrixCSC{$elty},
                              B::oneStridedMatrix{$elty},
                              beta::Number,
                              C::oneStridedMatrix{$elty})

            _mkl_empty(A) && return _scale_output!(beta, C)
            mB, nB = size(B)
            mC, nC = size(C)
            (nB != nC) && (transb == 'N') && throw(ArgumentError("B and C must have the same number of columns."))
            (mB != nC) && (transb != 'N') && throw(ArgumentError("Bᵀ and C must have the same number of columns."))
            nrhs = size(B, 2)
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(C), device(C))
            $fname(sycl_queue(queue), 'C', flip_trans(transa), transb, alpha, sparse_matrix_handle(A), B, nrhs, ldb, beta, C, ldc)
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
                A::oneSparseMatrixCSC{$elty},
                B::oneStridedMatrix{$elty},
                beta::Number,
                C::oneStridedMatrix{$elty}
            )

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
            queue = global_queue(context(C), device(C))

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

            $fname(sycl_queue(queue), 'C', flip_trans(transa), transb_eff, alpha, sparse_matrix_handle(A), B, nrhs, ldb, beta, C, ldc)

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

function sparse_optimize_gemm!(trans::Char, A::oneSparseMatrixCSC)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_gemm(sycl_queue(queue), flip_trans(trans), sparse_matrix_handle(A))
    return A
end

function sparse_optimize_gemm!(trans::Char, transB::Char, nrhs::Int, A::oneSparseMatrixCSC)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_gemm_advanced(sycl_queue(queue), 'C', flip_trans(trans), transB, sparse_matrix_handle(A), nrhs)
    return A
end

for (fname, elty) in ((:onemklSsparse_symv, :Float32),
                      (:onemklDsparse_symv, :Float64),
                      (:onemklCsparse_symv, :ComplexF32),
                      (:onemklZsparse_symv, :ComplexF64))
    @eval begin
        function sparse_symv!(uplo::Char,
                              alpha::Number,
                              A::oneSparseMatrixCSR{$elty},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _scale_output!(beta, y)
            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, alpha, sparse_matrix_handle(A), x, beta, y)
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
                              A::oneSparseMatrixCSC{$elty},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _scale_output!(beta, y)
            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), flip_uplo(uplo), alpha, sparse_matrix_handle(A), x, beta, y)
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
                              A::oneSparseMatrixCSR{$elty},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && return _scale_output!(beta, y)
            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, trans, diag, alpha, sparse_matrix_handle(A), x, beta, y)
            y
        end
    end
end

function sparse_optimize_trmv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSR)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trmv(sycl_queue(queue), uplo, trans, diag, sparse_matrix_handle(A))
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
                A::oneSparseMatrixCSC{$elty},
                x::oneStridedVector{$elty},
                beta::Number,
                y::oneStridedVector{$elty}
            )

            _mkl_empty(A) && return _scale_output!(beta, y)
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
            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, flip_trans(trans), diag, alpha, sparse_matrix_handle(A), x, beta, y)
            return y
        end
    end
end

function sparse_optimize_trmv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSC)
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trmv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trmv(sycl_queue(queue), uplo, flip_trans(trans), diag, sparse_matrix_handle(A))
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
                              A::oneSparseMatrixCSR{$elty},
                              x::oneStridedVector{$elty},
                              y::oneStridedVector{$elty})

            _mkl_empty(A) && throw(ArgumentError("cannot perform a triangular solve with an empty sparse matrix"))
            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, trans, diag, alpha, sparse_matrix_handle(A), x, y)
            y
        end
    end
end

function sparse_optimize_trsv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSR)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsv(sycl_queue(queue), uplo, trans, diag, sparse_matrix_handle(A))
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
                A::oneSparseMatrixCSC{$elty},
                x::oneStridedVector{$elty},
                y::oneStridedVector{$elty}
            )

            _mkl_empty(A) && throw(ArgumentError("cannot perform a triangular solve with an empty sparse matrix"))
            throw(
                ArgumentError(
                    "sparse_trsv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                        "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                        "Convert to oneSparseMatrixCSR format instead."
                )
            )
            queue = global_queue(context(y), device(y))
            onemklXsparse_optimize_trsv(sycl_queue(queue), uplo, flip_trans(trans), diag, sparse_matrix_handle(A))
            return A
        end
    end
end

function sparse_optimize_trsv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSC)
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trsv! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsv(sycl_queue(queue), uplo, flip_trans(trans), diag, sparse_matrix_handle(A))
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
                              A::oneSparseMatrixCSR{$elty},
                              X::oneStridedMatrix{$elty},
                              Y::oneStridedMatrix{$elty})

            _mkl_empty(A) && throw(ArgumentError("cannot perform a triangular solve with an empty sparse matrix"))
            mX, nX = size(X)
            mY, nY = size(Y)
            (mX != mY) && (transX == 'N') && throw(ArgumentError("X and Y must have the same number of rows."))
            (nX != nY) && (transX == 'N') && throw(ArgumentError("X and Y must have the same number of columns."))
            (nX != mY) && (transX != 'N') && throw(ArgumentError("Xᵀ and Y must have the same number of rows."))
            (mX != nY) && (transX != 'N') && throw(ArgumentError("Xᵀ and Y must have the same number of columns."))
            nrhs = size(X, 2)
            ldx = max(1,stride(X,2))
            ldy = max(1,stride(Y,2))
            queue = global_queue(context(Y), device(Y))
            $fname(sycl_queue(queue), 'C', transA, transX, uplo, diag, alpha, sparse_matrix_handle(A), X, nrhs, ldx, Y, ldy)
            Y
        end
    end
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSR)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsm(sycl_queue(queue), uplo, trans, diag, sparse_matrix_handle(A))
    return A
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, nrhs::Int, A::oneSparseMatrixCSR)
    _mkl_empty(A) && return A
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsm_advanced(sycl_queue(queue), 'C', uplo, trans, diag, sparse_matrix_handle(A), nrhs)
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
                A::oneSparseMatrixCSC{$elty},
                X::oneStridedMatrix{$elty},
                Y::oneStridedMatrix{$elty}
            )

            _mkl_empty(A) && throw(ArgumentError("cannot perform a triangular solve with an empty sparse matrix"))
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
            queue = global_queue(context(Y), device(Y))
            $fname(sycl_queue(queue), 'C', flip_trans(transA), transX, uplo, diag, alpha, sparse_matrix_handle(A), X, nrhs, ldx, Y, ldy)
            return Y
        end
    end
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSC)
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trsm! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsm(sycl_queue(queue), uplo, trans, diag, sparse_matrix_handle(A))
    return A
end

function sparse_optimize_trsm!(uplo::Char, trans::Char, diag::Char, nrhs::Int, A::oneSparseMatrixCSC)
    _mkl_empty(A) && return A
    throw(
        ArgumentError(
            "sparse_optimize_trsm! is not supported for oneSparseMatrixCSC due to Intel oneAPI limitations. " *
                "Intel sparse library only supports nontrans operations for triangular matrix operations. " *
                "Convert to oneSparseMatrixCSR format instead."
        )
    )
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsm_advanced(sycl_queue(queue), 'C', uplo, trans, diag, sparse_matrix_handle(A), nrhs)
    return A
end
