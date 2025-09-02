## (GE) general matrix-matrix multiplication batched
for (fname, elty) in
        ((:onemklHgemm_batch, :Float16),
         (:onemklSgemm_batch, :Float32),
         (:onemklDgemm_batch, :Float64),
         (:onemklCgemm_batch, :ComplexF32),
         (:onemklZgemm_batch, :ComplexF64))
    @eval begin
        function gemm_batched!(transA::Char,
                               transB::Char,
                               alpha::Number,
                               A::Vector{<:oneStridedMatrix{$elty}},
                               B::Vector{<:oneStridedMatrix{$elty}},
                               beta::Number,
                               C::Vector{<:oneStridedMatrix{$elty}})
            if length(A) != length(B) || length(A) != length(C)
                throw(DimensionMismatch(""))
            end
            for (As,Bs,Cs) in zip(A,B,C)
                m = size(As, transA == 'N' ? 1 : 2)
                k = size(As, transA == 'N' ? 2 : 1)
                n = size(Bs, transB == 'N' ? 2 : 1)
                if m != size(Cs,1) || n != size(Cs,2) || k != size(Bs, transB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end

            m = size(A[1], transA == 'N' ? 1 : 2)
            k = size(A[1], transA == 'N' ? 2 : 1)
            n = size(B[1], transB == 'N' ? 2 : 1)
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            Cptrs = unsafe_batch(C)
            bsize = length(A)
            m_dev = oneVector{Int}(fill(m,bsize))
            n_dev = oneVector{Int}(fill(n,bsize))
            k_dev = oneVector{Int}(fill(k,bsize))
            lda_dev = oneVector{Int}(fill(lda,bsize))
            ldb_dev = oneVector{Int}(fill(ldb,bsize))
            ldc_dev = oneVector{Int}(fill(ldc,bsize))
            alpha_dev = oneVector{$elty}(fill(alpha,bsize))
            beta_dev = oneVector{$elty}(fill(beta,bsize))
            groupsize_dev = oneVector{Int}(fill(1,bsize))

            queue = global_queue(context(A[1]), device(A[1]))
            $fname(sycl_queue(queue), transA, transB, m_dev, n_dev, k_dev, alpha_dev, Aptrs, lda_dev, Bptrs,
                   ldb_dev, beta_dev, Cptrs, ldc_dev, length(A), groupsize_dev)
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)
            unsafe_free!(m_dev)
            unsafe_free!(n_dev)
            unsafe_free!(k_dev)
            unsafe_free!(lda_dev)
            unsafe_free!(ldb_dev)
            unsafe_free!(ldc_dev)
            unsafe_free!(alpha_dev)
            unsafe_free!(beta_dev)
            unsafe_free!(groupsize_dev)
            C
        end
    end
end

function gemm_batched(transA::Char, transB::Char, alpha::Number,
                      A::Vector{<:oneStridedMatrix{T}}, B::Vector{<:oneStridedMatrix{T}}) where T
    C = oneMatrix{T}[similar(B[1], (size(A[1], transA == 'N' ? 1 : 2),size(B[1], transB == 'N' ? 2 : 1))) for i in 1:length(A)]
    gemm_batched!(transA, transB, alpha, A, B, zero(T), C )
end
function gemm_batched(transA::Char, transB::Char,
                      A::Vector{<:oneStridedMatrix{T}}, B::Vector{<:oneStridedMatrix{T}}) where T
    gemm_batched(transA, transB, one(T), A, B)
end

## (TR) triangular triangular matrix solution batched
for (fname, elty) in
        ((:onemklDtrsm_batch, :Float64),
         (:onemklStrsm_batch, :Float32),
         (:onemklCtrsm_batch, :ComplexF32),
         (:onemklZtrsm_batch, :ComplexF64))
    @eval begin
        function trsm_batched!(side::Char,
                               uplo::Char,
                               transa::Char,
                               diag::Char,
                               alpha::Number,
                               A::Vector{<:oneStridedMatrix{$elty}},
                               B::Vector{<:oneStridedMatrix{$elty}})
            if length(A) != length(B)
                throw(DimensionMismatch(""))
            end
            for (As,Bs) in zip(A,B)
                mA, nA = size(As)
                m,n = size(Bs)
                if mA != nA throw(DimensionMismatch("A must be square")) end
                if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm_batched!")) end
            end

            m,n = size(B[1])
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            bsize = length(A)
            m_dev = oneVector{Int}(fill(m,bsize))
            n_dev = oneVector{Int}(fill(n,bsize))
            lda_dev = oneVector{Int}(fill(lda,bsize))
            ldb_dev = oneVector{Int}(fill(ldb,bsize))
            alpha_dev = oneVector{$elty}(fill(alpha,bsize))
            groupsize_dev = oneVector{Int}(fill(1,bsize))
            queue = global_queue(context(A[1]), device(A[1]))
            $fname(sycl_queue(queue), side, uplo, transa, diag, m_dev, n_dev, alpha_dev, Aptrs, lda_dev, Bptrs, ldb_dev, length(A), groupsize_dev)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)
            unsafe_free!(m_dev)
            unsafe_free!(n_dev)
            unsafe_free!(lda_dev)
            unsafe_free!(ldb_dev)
            unsafe_free!(alpha_dev)
            unsafe_free!(groupsize_dev)
            B
        end
    end
end
function trsm_batched(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number,
                      A::Vector{<:oneStridedMatrix{T}}, B::Vector{<:oneStridedMatrix{T}}) where T
    trsm_batched!(side, uplo, transa, diag, alpha, A, copy(B) )
end

## (L3: symm) symmetric matrix-matrix and matrix-vector multiplication
for (fname, elty) in ((:onemklSsymm, :Float32),
                      (:onemklDsymm, :Float64),
                      (:onemklCsymm, :ComplexF32),
                      (:onemklZsymm, :ComplexF64))
    @eval begin
        function symm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       B::oneStridedVecOrMat{$elty},
                       beta::Number,
                       C::oneStridedVecOrMat{$elty})
            k, nA = size(A)
            if k != nA throw(DimensionMismatch("Matrix A must be square")) end
            m = side == 'L' ? k : size(B,1)
            n = side == 'L' ? size(B,2) : k
            if m != size(C,1) || n != size(C,2) || k != size(B, side == 'L' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), side, uplo, m, n, alpha, A, lda, B, ldb,
                   beta, C, ldc)
            C
        end
    end
end
function symm(side::Char,
                uplo::Char,
                alpha::Number,
                A::oneStridedVecOrMat{T},
                B::oneStridedVecOrMat{T}) where T
    symm!(side, uplo, alpha, A, B, zero(T), similar(B))
end
function symm(side::Char,
                uplo::Char,
                A::oneStridedVecOrMat{T},
                B::oneStridedVecOrMat{T}) where T
    symm(side, uplo, one(T), A, B)
end

## syrk
for (fname, elty) in ((:onemklSsyrk, :Float32),
                      (:onemklDsyrk, :Float64),
                      (:onemklCsyrk, :ComplexF32),
                      (:onemklZsyrk, :ComplexF64))
    @eval begin
        function syrk!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       beta::Number,
                       C::oneStridedMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
    end
end
function syrk(uplo::Char,
               trans::Char,
               alpha::Number,
               A::oneStridedVecOrMat{T}) where T
        n = size(A, trans == 'N' ? 1 : 2)
        syrk!(uplo, trans, alpha, A, zero(T), similar(A, (n, n)))
end
syrk(uplo::Char, trans::Char, A::oneStridedVecOrMat) =
    syrk(uplo, trans, one(eltype(A)), A)

## syr2k
for (fname, elty) in ((:onemklDsyr2k,:Float64),
                      (:onemklSsyr2k,:Float32),
                      (:onemklZsyr2k,:ComplexF64),
                      (:onemklCsyr2k,:ComplexF32))
    @eval begin
        function syr2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::oneStridedVecOrMat{$elty},
                        B::oneStridedVecOrMat{$elty},
                        beta::Number,
                        C::oneStridedVecOrMat{$elty})
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1) throw(DimensionMismatch(
                "Inner dimensions of op(A) and op(B.') must match")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
    end
end
function syr2k(uplo::Char,
               trans::Char,
               alpha::Number,
               A::oneStridedVecOrMat{T},
               B::oneStridedVecOrMat{T}) where T
        n = size(A, trans == 'N' ? 1 : 2)
        syr2k!(uplo, trans, convert(T, alpha), A, B, zero(T), similar(A, (n, n)))
end
syr2k(uplo::Char, trans::Char, A::oneStridedVecOrMat, B::oneStridedVecOrMat) =
        syr2k(uplo, trans, one(eltype(A)), A, B)

## herk
for (fname, elty) in ((:onemklZherk, :ComplexF64),
                      (:onemklCherk, :ComplexF32))
    @eval begin
        function herk!(uplo::Char,
                       trans::Char,
                       alpha::Real,
                       A::oneStridedVecOrMat{$elty},
                       beta::Real,
                       C::oneStridedMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("herk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
   end
end
function herk(uplo::Char, trans::Char, alpha::Real, A::oneStridedVecOrMat{T}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    herk!(uplo, trans, alpha, A, zero(real(T)), similar(A, (n,n)))
end
herk(uplo::Char, trans::Char, A::oneStridedVecOrMat{T}) where T =
    herk(uplo, trans, one(real(T)), A)

## her2k
for (fname, elty) in ((:onemklZher2k,:ComplexF64),
                      (:onemklCher2k,:ComplexF32))
    @eval begin
        function her2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::oneStridedVecOrMat{$elty},
                        B::oneStridedVecOrMat{$elty},
                        beta::Real,
                        C::oneStridedMatrix{$elty})
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1)
                throw(DimensionMismatch("Inner dimensions of op(A) and op(B.') must match"))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
   end
end
function her2k(uplo::Char,
                trans::Char,
                alpha::Number,
                A::oneStridedVecOrMat{T},
                B::oneStridedVecOrMat{T}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    her2k!(uplo, trans, alpha, A, B, zero(real(T)), similar(A, (n,n)))
end
her2k(uplo::Char, trans::Char,
      A::oneStridedVecOrMat{T}, B::oneStridedVecOrMat{T}) where T =
    her2k(uplo, trans, one(T), A, B)

# level 2
## gemv
for (fname, elty) in ((:onemklSgemv, :Float32),
                      (:onemklDgemv, :Float64),
                      (:onemklCgemv, :ComplexF32),
                      (:onemklZgemv, :ComplexF64))
    @eval begin
        function gemv!(trans::Char,
                       alpha::Number,
                       a::oneStridedArray{$elty},
                       x::oneStridedArray{$elty},
                       beta::Number,
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
             # handle trans
             m,n = size(a)
             # check dimensions
             length(x) == (trans == 'N' ? n : m) && length(y) ==
                          (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
             # compute increments
             lda = max(1,stride(a,2))
             incx = stride(x,1)
             incy = stride(y,1)
             $fname(sycl_queue(queue), trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
             y
        end
    end
end
function gemv(trans::Char,
              alpha::Number,
              a::oneStridedArray{T},
              x::oneStridedArray{T}) where T
    gemv!(trans, alpha, a, x, zero(T), similar(x, size(a, (trans == 'N' ? 1 : 2))))
end

function gemv(trans::Char,
              a::oneStridedArray{T},
              x::oneStridedArray{T}) where T
    gemv!(trans, one(T), a, x, zero(T), similar(x, size(a, (trans == 'N' ? 1 : 2))))
end

### hemv
for (fname, elty) in ((:onemklChemv,:ComplexF32),
                      (:onemklZhemv,:ComplexF64))
    @eval begin
        function hemv!(uplo::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty},
                       beta::Number,
                       y::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
    end
end
function hemv(uplo::Char, alpha::Number, A::oneStridedVecOrMat{T},
                x::oneStridedVecOrMat{T}) where T
    hemv!(uplo, alpha, A, x, zero(T), similar(x))
end
function hemv(uplo::Char, A::oneStridedVecOrMat{T},
                x::oneStridedVecOrMat{T}) where T
    hemv(uplo, one(T), A, x)
end

### hbmv, (HB) Hermitian banded matrix-vector multiplication
for (fname, elty) in ((:onemklChbmv,:ComplexF32),
                      (:onemklZhbmv,:ComplexF64))
    @eval begin

        function hbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       x::oneStridedVector{$elty},
                       beta::Number,
                       y::oneStridedVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
    end
end
function hbmv(uplo::Char, k::Integer, alpha::Number,
                A::oneStridedMatrix{T}, x::oneStridedVector{T}) where T
    n = size(A,2)
    hbmv!(uplo, k, alpha, A, x, zero(T), similar(x, n))
end
function hbmv(uplo::Char, k::Integer, A::oneStridedMatrix{T},
                x::oneStridedVector{T}) where T
    hbmv(uplo, k, one(T), A, x)
end

### her
for (fname, elty) in ((:onemklCher,:ComplexF32),
                      (:onemklZher,:ComplexF64))
    @eval begin
        function her!(uplo::Char,
                      alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      A::oneStridedVecOrMat{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, A, lda)
            A
        end
    end
end

### her2
for (fname, elty) in ((:onemklCher2,:ComplexF32),
                      (:onemklZher2,:ComplexF64))
    @eval begin
        function her2!(uplo::Char,
                      alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      y::oneStridedVecOrMat{$elty},
                      A::oneStridedVecOrMat{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            length(y) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, y, incy, A, lda)
            A
        end
    end
end

# level 1
## axpy
for (fname, elty) in
        ((:onemklDaxpy,:Float64),
         (:onemklSaxpy,:Float32),
         (:onemklHaxpy,:Float16),
         (:onemklZaxpy,:ComplexF64),
         (:onemklCaxpy,:ComplexF32))
    @eval begin
        function axpy!(n::Integer,
                       alpha::Number,
                       x::oneStridedArray{$elty},
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
            alpha = $elty(alpha)
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1), y, stride(y,1))
            y
        end
    end
end

## axpby
for (fname, elty) in
        ((:onemklDaxpby,:Float64),
         (:onemklSaxpby,:Float32),
         (:onemklZaxpby,:ComplexF64),
         (:onemklCaxpby,:ComplexF32))
    @eval begin
        function axpby!(n::Integer,
                        alpha::Number,
                        x::oneStridedArray{$elty},
                        beta::Number,
                        y::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
            alpha = $elty(alpha)
            beta = $elty(beta)
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1), beta, y, stride(y,1))
            y
        end
    end
end

## rot
for (fname, elty, cty, sty, supty) in ((:onemklSrot,:Float32,:Float32,:Float32,:Number),
                                       (:onemklDrot,:Float64,:Float64,:Float64,:Number),
                                       (:onemklCrot,:ComplexF32,:Float32,:ComplexF32,:Number),
                                       (:onemklZrot,:ComplexF64,:Float64,:ComplexF64,:Number),
                                       (:onemklCSrot,:ComplexF32,:Float32,:Float32,:Real),
                                       (:onemklZDrot,:ComplexF64,:Float64,:Float64,:Real))
    @eval begin
        function rot!(n::Integer,
                      x::oneStridedArray{$elty},
                      y::oneStridedArray{$elty},
                      c::Real,
                      s::$supty)
            queue = global_queue(context(x), device())
            c = $cty(c)
            s = $sty(s)
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1), c, s)
            x, y
        end
    end
end

function axpy!(n::Integer,
            alpha::Number,
            x::oneStridedArray{ComplexF16},
            y::oneStridedArray{ComplexF16})
    wide_x = widen.(x)
    wide_y = widen.(y)
    axpy!(n, alpha, wide_x, wide_y)
    thin_y = convert(typeof(y), wide_y)
    copyto!(y, thin_y)
    return y
end

## scal
for (fname, elty) in
    ((:onemklDscal,:Float64),
     (:onemklSscal,:Float32),
     (:onemklHscal,:Float16),
     (:onemklZscal,:ComplexF64),
     (:onemklCscal,:ComplexF32))
    @eval begin
        function scal!(n::Integer,
                       alpha::$elty,
                       x::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1))
            x
        end
    end
end

function scal!(n::Integer,
            alpha::Number,
            x::oneStridedArray{ComplexF16})
    wide_x = widen.(x)
    scal!(n, convert(ComplexF32, alpha), wide_x)
    thin_x = convert(typeof(x), wide_x)
    copyto!(x, thin_x)
    return x
end

## nrm2
for (fname, elty, ret_type) in
    ((:onemklDnrm2, :Float64,:Float64),
     (:onemklSnrm2, :Float32,:Float32),
     (:onemklHnrm2, :Float16,:Float16),
     (:onemklCnrm2, :ComplexF32,:Float32),
     (:onemklZnrm2, :ComplexF64,:Float64))
    @eval begin
        function nrm2(n::Integer, x::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
            result = oneArray{$ret_type}([0]);
            $fname(sycl_queue(queue), n, x, stride(x,1), result)
            res = Array(result)
            return res[1]
        end
    end
end

nrm2(x::oneStridedArray) = nrm2(length(x), x)

function nrm2(n::Integer, x::oneStridedArray{ComplexF16})
    wide_x = widen.(x)
    nrm = nrm2(n, wide_x)
    return convert(Float16, nrm)
end

## dot
for (jname, fname, elty) in
        ((:dot, :onemklSdot,:Float32),
         (:dot, :onemklDdot,:Float64),
         (:dot, :onemklHdot,:Float16),
         (:dotc, :onemklCdotc, :ComplexF32),
         (:dotc, :onemklZdotc, :ComplexF64),
         (:dotu, :onemklCdotu, :ComplexF32),
         (:dotu, :onemklZdotu, :ComplexF64))
    @eval begin
        function $jname(n::Integer,
                         x::oneStridedArray{$elty},
                         y::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
            result = oneArray{$elty}([0]);
            $fname(sycl_queue(queue), n, x, stride(x,1), y, stride(y,1), result)
            res = Array(result)
            return res[1]
        end
    end
end

function dotc(n::Integer, x::oneStridedArray{ComplexF16}, y::oneStridedArray{ComplexF16})
    convert(ComplexF16, dotc(n, convert(oneArray{ComplexF32}, x), convert(oneArray{ComplexF32}, y)))
end

function dotu(n::Integer, x::oneStridedArray{ComplexF16}, y::oneStridedArray{ComplexF16})
    convert(ComplexF16, dotu(n, convert(oneArray{ComplexF32}, x), convert(oneArray{ComplexF32}, y)))
end

# level 2
# sbmv, symmetric banded matrix-vector multiplication
for (fname, elty) in ((:onemklSsbmv, :Float32),
                      (:onemklDsbmv, :Float64))
    @eval begin
        function sbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       a::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty},
                       beta::Number,
                       y::oneStridedVecOrMat{$elty})
            m, n = size(a)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            queue = global_queue(context(x), device())
            lda = max(1, stride(a,2))
            incx = stride(x,1)
            incy = stride(y,1)
            alpha = $elty(alpha)
            beta = $elty(beta)
            $fname(sycl_queue(queue), uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
            y
        end
    end
end
function sbmv(uplo::Char, k::Integer, alpha::Number,
                a::oneStridedArray{T}, x::oneStridedArray{T}) where T
    n = size(a,2)
    sbmv!(uplo, k, alpha, a, x, zero(T), similar(x, n))
end
function sbmv(uplo::Char, k::Integer, a::oneStridedArray{T},
                x::oneStridedArray{T}) where T
    sbmv(uplo, k, one(T), a, x)
end

for (fname, elty, celty) in ((:onemklCSscal, :Float32, :ComplexF32),
                             (:onemklZDscal, :Float64, :ComplexF64))
    @eval begin
        function scal!(n::Integer,
                       alpha::$elty,
                       x::oneStridedArray{$celty})
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1))
        end
    end
end

# level 2
# ger
for (fname, elty) in ((:onemklSger, :Float32),
                      (:onemklDger, :Float64),
                      (:onemklCgerc, :ComplexF32),
                      (:onemklZgerc, :ComplexF64))
    @eval begin
        function ger!(alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      y::oneStridedVecOrMat{$elty},
                      a::oneStridedVecOrMat{$elty})
            m,n = size(a)
            m == length(x) || throw(DimensionMismatch(""))
            n == length(y) || throw(DimensionMismatch(""))
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), m, n, alpha, x, stride(x,1), y, stride(y,1), a, max(1,stride(a,2)))
            a
        end
    end
end

# spr
for (fname, elty) in ((:onemklSspr, :Float32),
                      (:onemklDspr, :Float64))
    @eval begin
        function spr!(uplo::Char,
                      alpha::Number,
                      x::oneStridedVector{$elty},
                      A::oneStridedVector{$elty})
            n = round(Int, (sqrt(8*length(A))-1)/2)
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, A)
            A
        end
    end
end

#symv
for (fname, elty) in ((:onemklSsymv,:Float32),
                      (:onemklDsymv,:Float64))
    # Note that the complex symv are not BLAS but auxiliary functions in LAPACK
    @eval begin
        function symv!(uplo::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty},
                       beta::Number,
                       y::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
    end
end
function symv(uplo::Char, alpha::Number, A::oneStridedVecOrMat{T}, x::oneStridedVecOrMat{T}) where T
        symv!(uplo, alpha, A, x, zero(T), similar(x))
end
function symv(uplo::Char, A::oneStridedVecOrMat{T}, x::oneStridedVecOrMat{T}) where T
    symv(uplo, one(T), A, x)
end

# syr
for (fname, elty) in ((:onemklSsyr,:Float32),
                      (:onemklDsyr,:Float64))
    @eval begin
        function syr!(uplo::Char,
                      alpha::Number,
                      x::oneStridedVecOrMat{$elty},
                      A::oneStridedVecOrMat{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, A, lda)
            A
        end
    end
end

#
# BLAS
#

# level 1
## copy
for (fname, elty) in
        ((:onemklDcopy,:Float64),
         (:onemklScopy,:Float32),
         (:onemklZcopy,:ComplexF64),
         (:onemklCcopy,:ComplexF32))
    @eval begin
        function copy!(n::Integer,
                       x::oneStridedArray{$elty},
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1))
            y
        end
    end
end

function copy!(n::Integer, x::oneStridedArray{T}, y::oneStridedArray{T}) where {T <: Union{Float16, ComplexF16}}
    copyto!(y,x)
end

## asum
for (fname, elty, ret_type) in
    ((:onemklSasum, :Float32, :Float32),
     (:onemklDasum, :Float64, :Float64),
     (:onemklCasum, :ComplexF32, :Float32),
     (:onemklZasum, :ComplexF64, :Float64))
    @eval begin
        function asum(n::Integer,
                      x::oneStridedArray{$elty})
            result = oneArray{$ret_type}([0])
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), n, x, stride(x, 1), result)
            res = Array(result)
            return res[1]
        end
    end
end

## iamax
for (fname, elty) in
    ((:onemklDiamax_64,:Float64),
     (:onemklSiamax_64,:Float32),
     (:onemklZiamax_64,:ComplexF64),
     (:onemklCiamax_64,:ComplexF32))
    @eval begin
        function iamax(x::oneStridedArray{$elty})
            n = length(x)
            queue = global_queue(context(x), device())
            result = oneArray{Int64}([0]);
            $fname(sycl_queue(queue), n, x, stride(x, 1), result, 'O')
            return Array(result)[1]
        end
    end
end

## iamin
for (fname, elty) in
    ((:onemklDiamin_64,:Float64),
     (:onemklSiamin_64,:Float32),
     (:onemklZiamin_64,:ComplexF64),
     (:onemklCiamin_64,:ComplexF32))
    @eval begin
        function iamin(x::StridedArray{$elty})
            n = length(x)
            result = oneArray{Int64}([0]);
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue),n, x, stride(x, 1), result, 'O')
            return Array(result)[1]
        end
    end
end

## swap
for (fname, elty) in ((:onemklSswap,:Float32),
                      (:onemklDswap,:Float64),
                      (:onemklCswap,:ComplexF32),
                      (:onemklZswap,:ComplexF64))
    @eval begin
        function swap!(n::Integer,
            x::oneStridedArray{$elty},
            y::oneStridedArray{$elty})
            # Assuming both memory allocated on same device & context
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1))
            x, y
        end
    end
end

# level 2
# gbmv
for (fname, elty) in ((:onemklSgbmv, :Float32),
                      (:onemklDgbmv, :Float64),
                      (:onemklCgbmv, :ComplexF32),
                      (:onemklZgbmv, :ComplexF64))
    @eval begin
        function gbmv!(trans::Char,
                       m::Integer,
                       kl::Integer,
                       ku::Integer,
                       alpha::Number,
                       a::oneStridedArray{$elty},
                       x::oneStridedArray{$elty},
                       beta::Number,
                       y::oneStridedArray{$elty})
            n = size(a,2)
            length(x) == (trans == 'N' ? n : m) && length(y) ==
                         (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            queue = global_queue(context(x), device())
            lda = max(1, stride(a,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(sycl_queue(queue), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
            y
        end
    end
end
function gbmv(trans::Char,
              m::Integer,
              kl::Integer,
              ku::Integer,
              alpha::Number,
              a::oneStridedArray{T},
              x::oneStridedArray{T}) where T
    n = size(a,2)
    leny = trans == 'N' ? m : n
    queue = global_queue(context(x), device())
    gbmv!(trans, m, kl, ku, alpha, a, x, zero(T), similar(x, leny))
end
function gbmv(trans::Char,
              m::Integer,
              kl::Integer,
              ku::Integer,
              a::oneStridedArray{T},
              x::oneStridedArray{T}) where T
    queue = global_queue(context(x), device())
    gbmv(trans, m, kl, ku, one(T), a, x)
end

# spmv
for (fname, elty) in ((:onemklSspmv, :Float32),
                      (:onemklDspmv, :Float64))
    @eval begin
        function spmv!(uplo::Char,
                       alpha::Number,
                       A::oneStridedVector{$elty},
                       x::oneStridedVector{$elty},
                       beta::Number,
                       y::oneStridedVector{$elty})
            n = round(Int, (sqrt(8*length(A))-1)/2)
            if n != length(x) || n != length(y)
                throw(DimensionMismatch(""))
            end
            incx = stride(x,1)
            incy = stride(y,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, n, alpha, A, x, incx, beta, y, incy)
            y
        end
    end
end

function spmv(uplo::Char, alpha::Number,
              A::oneStridedVector{T}, x::oneStridedVector{T}) where T
    spmv!(uplo, alpha, A, x, zero(T), similar(x))
end

function spmv(uplo::Char, A::oneStridedVector{T}, x::oneStridedVector{T}) where T
    spmv(uplo, one(T), A, x)
end

# tbsv, (TB) triangular banded matrix solve
for (fname, elty) in ((:onemklStbsv, :Float32),
                      (:onemklDtbsv, :Float64),
                      (:onemklCtbsv, :ComplexF32),
                      (:onemklZtbsv, :ComplexF64))
    @eval begin
        function tbsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::oneStridedMatrix{$elty},
                       x::oneStridedVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, trans, diag, n, k, A, lda, x, incx)
            x
        end
    end
end
function tbsv(uplo::Char, trans::Char, diag::Char, k::Integer,
              A::oneStridedMatrix{T}, x::oneStridedVector{T}) where T
    tbsv!(uplo, trans, diag, k, A, copy(x))
end

# tbmv
### tbmv, (TB) triangular banded matrix-vector multiplication
for (fname, elty) in ((:onemklStbmv,:Float32),
                      (:onemklDtbmv,:Float64),
                      (:onemklCtbmv,:ComplexF32),
                      (:onemklZtbmv,:ComplexF64))
    @eval begin
        function tbmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, trans, diag, n, k, A, lda, x, incx)
            x
        end
    end
end
function tbmv(uplo::Char,
                trans::Char,
                diag::Char,
                k::Integer,
                A::oneStridedVecOrMat{T},
                x::oneStridedVecOrMat{T}) where T
    tbmv!(uplo, trans, diag, k, A, copy(x))
end

### trmv, Triangular matrix-vector multiplication
for (fname, elty) in ((:onemklStrmv, :Float32),
                      (:onemklDtrmv, :Float64),
                      (:onemklCtrmv, :ComplexF32),
                      (:onemklZtrmv, :ComplexF64))
    @eval begin
        function trmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, trans, diag, n, A, lda, x, incx)
            x
        end
    end
end
function trmv(uplo::Char,
              trans::Char,
              diag::Char,
              A::oneStridedVecOrMat{T},
              x::oneStridedVecOrMat{T}) where T
    trmv!(uplo, trans, diag, A, copy(x))
end

### trsv, Triangular matrix-vector solve
for (fname, elty) in ((:onemklStrsv, :Float32),
                      (:onemklDtrsv, :Float64),
                      (:onemklCtrsv, :ComplexF32),
                      (:onemklZtrsv, :ComplexF64))
    @eval begin
        function trsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::oneStridedVecOrMat{$elty},
                       x::oneStridedVecOrMat{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            queue = global_queue(context(x), device())
            $fname(sycl_queue(queue), uplo, trans, diag, n, A, lda, x, incx)
            x
        end
    end
end
function trsv(uplo::Char,
                trans::Char,
                diag::Char,
                A::oneStridedVecOrMat{T},
                x::oneStridedVecOrMat{T}) where T
    trsv!(uplo, trans, diag, A, copy(x))
end

# level 3

for (mmname, smname, elty) in
        ((:onemklDtrmm, :onemklDtrsm, :Float64),
         (:onemklStrmm, :onemklStrsm, :Float32),
         (:onemklZtrmm, :onemklZtrsm, :ComplexF64),
         (:onemklCtrmm, :onemklCtrsm, :ComplexF32))
    @eval begin
        function trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trmm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            queue = global_queue(context(A), device())
            $mmname(sycl_queue(queue), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
            B
        end

        function trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            queue = global_queue(context(A), device())
            $smname(sycl_queue(queue), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
            B
        end
    end
end
function trmm(side::Char,
              uplo::Char,
              transa::Char,
              diag::Char,
              alpha::Number,
              A::oneStridedMatrix{T},
              B::oneStridedMatrix{T}) where T
    trmm!(side, uplo, transa, diag, alpha, A, copy(B))
end
function trsm(side::Char,
                uplo::Char,
                transa::Char,
                diag::Char,
                alpha::Number,
                A::oneStridedMatrix{T},
                B::oneStridedMatrix{T}) where T
    trsm!(side, uplo, transa, diag, alpha, A, copy(B))
end

for (mmname_variant, smname_variant, elty) in
        ((:onemklDtrmm_variant, :onemklDtrsm_variant, :Float64),
         (:onemklStrmm_variant, :onemklStrsm_variant, :Float32),
         (:onemklZtrmm_variant, :onemklZtrsm_variant, :ComplexF64),
         (:onemklCtrmm_variant, :onemklCtrsm_variant, :ComplexF32))
    @eval begin
        function trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       beta::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty},
                       C::oneStridedMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trmm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $mmname_variant(sycl_queue(queue), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            B
        end

        function trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       beta::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty},
                       C::oneStridedMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $smname_variant(sycl_queue(queue), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            B
        end
    end
end
function trmm!(side::Char,
               uplo::Char,
               transa::Char,
               diag::Char,
               alpha::Number,
               A::oneStridedMatrix{T},
               B::oneStridedMatrix{T},
               C::oneStridedMatrix{T}) where T
    trmm!(side, uplo, transa, diag, alpha, zero(T), A, B, C)
end
function trsm!(side::Char,
               uplo::Char,
               transa::Char,
               diag::Char,
               alpha::Number,
               A::oneStridedMatrix{T},
               B::oneStridedMatrix{T},
               C::oneStridedMatrix{T}) where T
    trsm!(side, uplo, transa, diag, alpha, zero(T), A, B, C)
end

## hemm
for (fname, elty) in ((:onemklZhemm,:ComplexF64),
                      (:onemklChemm,:ComplexF32))
    @eval begin
        function hemm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::oneStridedMatrix{$elty},
                       B::oneStridedMatrix{$elty},
                       beta::Number,
                       C::oneStridedMatrix{$elty})
            mA, nA = size(A)
            m, n = size(B)
            mC, nC = size(C)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if ((m != mC) || (n != nC)) throw(DimensionMismatch("B and C must have same dimensions")) end
            if ((side == 'L') && (mA != m)) throw(DimensionMismatch("")) end
            if ((side == 'R') && (mA != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
    end
end
function hemm(uplo::Char,
                trans::Char,
                alpha::Number,
                A::oneStridedMatrix{T},
                B::oneStridedMatrix{T}) where T
    m,n = size(B)
    hemm!( uplo, trans, alpha, A, B, zero(T), similar(B, (m,n) ) )
end
hemm(uplo::Char, trans::Char, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where T=
    hemm( uplo, trans, one(T), A, B)

for (fname, elty) in
        ((:onemklDgemm,:Float64),
         (:onemklSgemm,:Float32),
         (:onemklHgemm,:Float16),
         (:onemklZgemm,:ComplexF64),
         (:onemklCgemm,:ComplexF32))
    @eval begin
        function gemm!(transA::Char,
                       transB::Char,
                       alpha::Number,
                       A::oneStridedVecOrMat{$elty},
                       B::oneStridedVecOrMat{$elty},
                       beta::Number,
                       C::oneStridedVecOrMat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end

            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))

            device() == device(B) == device(C) || error("Multi-device GEMM not supported")
            context(A) == context(B) == context(C) || error("Multi-context GEMM not supported")
            queue = global_queue(context(A), device())

            alpha = $elty(alpha)
            beta = $elty(beta)

            $fname(sycl_queue(queue), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
    end
end
function gemm(transA::Char,
                transB::Char,
                alpha::Number,
                A::oneStridedVecOrMat{T},
                B::oneStridedVecOrMat{T}) where T
    gemm!(transA, transB, alpha, A, B, zero(T),
            similar(B, (size(A, transA == 'N' ? 1 : 2),
                        size(B, transB == 'N' ? 2 : 1))))
end
function gemm(transA::Char,
                transB::Char,
                A::oneStridedVecOrMat{T},
                B::oneStridedVecOrMat{T}) where T
    gemm(transA, transB, one(T), A, B)
end

## dgmm
for (fname, elty) in ((:onemklSdgmm, :Float32),
                      (:onemklDdgmm, :Float64),
                      (:onemklCdgmm, :ComplexF32),
                      (:onemklZdgmm, :ComplexF64))
    @eval begin
        function dgmm!(mode::Char,
                       A::oneStridedMatrix{$elty},
                       X::oneStridedVector{$elty},
                       C::oneStridedMatrix{$elty})
            m, n = size(C)
            mA, nA = size(A)
            lx = length(X)
            if ((mA != m) || (nA != n )) throw(DimensionMismatch("")) end
            if ((mode == 'L') && (lx != m)) throw(DimensionMismatch("")) end
            if ((mode == 'R') && (lx != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(X,1)
            ldc = max(1,stride(C,2))
            queue = global_queue(context(A), device())
            $fname(sycl_queue(queue), mode, m, n, A, lda, X, incx, C, ldc)
            C
        end
    end
end
function dgmm(mode::Char, A::oneStridedMatrix{T}, X::oneStridedVector{T}) where T
    m,n = size(A)
    dgmm!( mode, A, X, similar(A, (m,n) ) )
end

for (fname, elty) in
        ((:onemklHgemm_batch_strided, Float16),
         (:onemklSgemm_batch_strided, Float32),
         (:onemklDgemm_batch_strided, Float64),
         (:onemklCgemm_batch_strided, ComplexF32),
         (:onemklZgemm_batch_strided, ComplexF64))
    @eval begin
        function gemm_strided_batched!(transA::Char,
                                    transB::Char,
                                    alpha::Number,
                                    A::AbstractArray{$elty, 3},
                                    B::AbstractArray{$elty, 3},
                                    beta::Number,
                                    C::AbstractArray{$elty, 3})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)

            @assert size(A, 3) == size(C, 3) || size(A, 3) == 1 "batch size mismatch: A != C"
            @assert size(B, 3) == size(C, 3) || size(B, 3) == 1 "batch size mismatch: B != C"

            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))

            strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
            strideB = size(B, 3) == 1 ? 0 : stride(B, 3)
            strideC = stride(C, 3)
            batchCount = size(C, 3)
            queue = global_queue(context(A), device())
            alpha = $elty(alpha)
            beta = $elty(beta)
            $fname(sycl_queue(queue), transA, transB, m, n, k, alpha, A, lda, strideA, B,
                    ldb, strideB, beta, C, ldc, strideC, batchCount)
            C
        end
    end
end
function gemm_strided_batched(transA::Char, transB::Char, alpha::Number,
                              A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    C = similar(B, (size(A, transA == 'N' ? 1 : 2),
                    size(B, transB == 'N' ? 2 : 1),
                    max(size(A, 3), size(B, 3))))
    gemm_strided_batched!(transA, transB, alpha, A, B, zero(T), C )
end
function gemm_strided_batched(transA::Char, transB::Char, A::AbstractArray{T, 3},
                              B::AbstractArray{T,3}) where T
    gemm_strided_batched(transA, transB, one(T), A, B)
end
