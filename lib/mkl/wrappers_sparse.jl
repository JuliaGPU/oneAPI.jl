export oneSparseMatrixCSR

mutable struct oneSparseMatrixCSR{T}
    handle::matrix_handle_t
    type::Type{T}
    m::Int
    n::Int
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
            row_ptr = oneVector{$intty}(At.colptr)
            col_ind = oneVector{$intty}(At.rowval)
            val = oneVector{$elty}(At.nzval)
            queue = global_queue(context(val), device(val))
            $fname(sycl_queue(queue), handle_ptr[], m, n, 'O', row_ptr, col_ind, val)
            return oneSparseMatrixCSR{$elty}(handle_ptr[], $elty, m, n)
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
