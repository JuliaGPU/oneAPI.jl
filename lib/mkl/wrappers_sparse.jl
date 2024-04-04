function sparse_release_matrix_handle(A::oneSparseMatrixCSR)
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    handle_ptr = Ref{matrix_handle_t}(A.handle)
    onemklXsparse_release_matrix_handle(sycl_queue(queue), handle_ptr)
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
        function oneSparseMatrixCSR(A::SparseMatrixCSC{$elty, $intty})
            handle_ptr = Ref{matrix_handle_t}()
            onemklXsparse_init_matrix_handle(handle_ptr)
            m, n = size(A)
            At = SparseMatrixCSC(A |> transpose)
            rowPtr = oneVector{$intty}(At.colptr)
            colVal = oneVector{$intty}(At.rowval)
            nzVal = oneVector{$elty}(At.nzval)
            nnzA = length(At.nzval)
            queue = global_queue(context(nzVal), device(nzVal))
            $fname(sycl_queue(queue), handle_ptr[], m, n, 'O', rowPtr, colVal, nzVal)
            dA = oneSparseMatrixCSR{$elty, $intty}(handle_ptr[], rowPtr, colVal, nzVal, (m,n), nnzA)
            finalizer(sparse_release_matrix_handle, dA)
            return dA
        end

        function SparseMatrixCSC(A::oneSparseMatrixCSR{$elty, $intty})
            handle_ptr = Ref{matrix_handle_t}()
            At = SparseMatrixCSC(reverse(A.dims)..., Array(A.rowPtr), Array(A.colVal), Array(A.nzVal))
            A_csc = SparseMatrixCSC(At |> transpose)
            return A_csc
        end
    end
end

for (fname, elty) in ((:onemklSsparse_gemv, :Float32),
                      (:onemklDsparse_gemv, :Float64),
                      (:onemklCsparse_gemv, :ComplexF32),
                      (:onemklZsparse_gemv, :ComplexF64))
    @eval begin
        function sparse_gemv!(trans::Char,
                              alpha::Number,
                              A::oneSparseMatrixCSR{$elty},
                              x::oneStridedVector{$elty},
                              beta::Number,
                              y::oneStridedVector{$elty})

            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), trans, alpha, A.handle, x, beta, y)
            y
        end
    end
end

function sparse_optimize_gemv!(trans::Char, A::oneSparseMatrixCSR)
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_gemv(sycl_queue(queue), trans, A.handle)
    return A
end

for (fname, elty) in ((:onemklSsparse_gemm, :Float32),
                      (:onemklDsparse_gemm, :Float64),
                      (:onemklCsparse_gemm, :ComplexF32),
                      (:onemklZsparse_gemm, :ComplexF64))
    @eval begin
        function sparse_gemm!(transa::Char,
                              transb::Char,
                              alpha::Number,
                              A::oneSparseMatrixCSR{$elty},
                              B::oneStridedMatrix{$elty},
                              beta::Number,
                              C::oneStridedMatrix{$elty})

            mB, nB = size(B)
            mC, nC = size(C)
            (nB != nC) && (transb == 'N') && throw(ArgumentError("B and C must have the same number of columns."))
            (mB != nC) && (transb != 'N') && throw(ArgumentError("Bᵀ and C must have the same number of columns."))
            nrhs = size(B, 2)
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(C), device(C))
            $fname(sycl_queue(queue), 'C', transa, transb, alpha, A.handle, B, nrhs, ldb, beta, C, ldc)
            C
        end
    end
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

            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, alpha, A.handle, x, beta, y)
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

            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, trans, diag, alpha, A.handle, x, beta, y)
            y
        end
    end
end

function sparse_optimize_trmv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSR)
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trmv(sycl_queue(queue), uplo, trans, diag, A.handle)
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
                              A::oneSparseMatrixCSR{$elty},
                              x::oneStridedVector{$elty},
                              y::oneStridedVector{$elty})

            queue = global_queue(context(y), device(y))
            $fname(sycl_queue(queue), uplo, trans, diag, A.handle, x, y)
            y
        end
    end
end

function sparse_optimize_trsv!(uplo::Char, trans::Char, diag::Char, A::oneSparseMatrixCSR)
    queue = global_queue(context(A.nzVal), device(A.nzVal))
    onemklXsparse_optimize_trsv(sycl_queue(queue), uplo, trans, diag, A.handle)
    return A
end
