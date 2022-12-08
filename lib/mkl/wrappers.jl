#
# Auxiliary
#

function Base.convert(::Type{onemklTranspose}, trans::Char)
    if trans == 'N'
        return ONEMKL_TRANSPOSE_NONTRANS
    elseif trans == 'T'
        return ONEMKL_TRANSPOSE_TRANS
    elseif trans == 'C'
        return ONEMLK_TRANSPOSE_CONJTRANS
    else
        throw(ArgumentError("Unknown transpose $trans"))
    end
end

function Base.convert(::Type{onemklUplo}, uplo::Char)
    if uplo == 'U'
        return ONEMKL_UPLO_UPPER
    elseif uplo == 'L'
        return ONEMKL_UPLO_LOWER
    else
        throw(ArgumentError("Unknown transpose $uplo"))
    end
end

function Base.convert(::Type{onemklDiag}, diag::Char)
    if diag == 'N'
        return ONEMKL_DIAG_NONUNIT
    elseif diag == 'U'
        return ONEMKL_DIAG_UNIT
    else
        throw(ArgumentError("Unknown transpose $diag"))
    end
end

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
            queue = global_queue(context(x), device(x))
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

        function gemv(trans::Char,
                      alpha::Number,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            gemv!(trans, alpha, a, x, zero($elty), similar(x, $elty, size(a, (trans == 'N' ? 1 : 2))))
        end

        function gemv(trans::Char,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            gemv!(trans, one($elty), a, x, zero($elty), similar(x, $elty, size(a, (trans == 'N' ? 1 : 2))))
        end
    end
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end

        function hemv(uplo::Char, alpha::Number, A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            hemv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function hemv(uplo::Char, A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            hemv(uplo, one($elty), A, x)
        end
    end
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            y
        end

        function hbmv(uplo::Char, k::Integer, alpha::Number,
                      A::oneStridedMatrix{$elty}, x::oneStridedVector{$elty})
            n = size(A,2)
            hbmv!(uplo, k, alpha, A, x, zero($elty), similar(x, $elty, n))
        end

        function hbmv(uplo::Char, k::Integer, A::oneStridedMatrix{$elty},
                      x::oneStridedVector{$elty})
            hbmv(uplo, k, one($elty), A, x)
        end

    end
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
            queue = global_queue(context(x), device(x))
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, alpha, x, incx, y, incy, A, lda)
            A
        end
    end
end

# level 1
## axpy primitive
for (fname, elty) in 
        ((:onemklDaxpy,:Float64),
         (:onemklSaxpy,:Float32),
         (:onemklZaxpy,:ComplexF64),
         (:onemklCaxpy,:ComplexF32))
    @eval begin
        function axpy!(n::Integer,
                       alpha::Number,
                       x::oneStridedArray{$elty},
                       y::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            alpha = $elty(alpha)
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1), y, stride(y,1))
            y
        end
    end
end

## scal
for (fname, elty) in
    ((:onemklDscal,:Float64),
     (:onemklSscal,:Float32),
     (:onemklZscal,:ComplexF64),
     (:onemklCscal,:ComplexF32))
    @eval begin
        function scal!(n::Integer,
                       alpha::$elty,
                       x::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, alpha, x, stride(x,1))
            x
        end
    end
end

## nrm2
for (fname, elty, ret_type) in
    ((:onemklDnrm2, :Float64,:Float64),
     (:onemklSnrm2, :Float32,:Float32),
     (:onemklCnrm2, :ComplexF32,:Float32),
     (:onemklZnrm2, :ComplexF64,:Float64))
    @eval begin
        function nrm2(n::Integer, x::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            result = oneArray{$ret_type}([0]);
            $fname(sycl_queue(queue), n, x, stride(x,1), result)
            res = Array(result)
            return res[1]
        end
    end
end

## dot
for (jname, fname, elty) in
        ((:dot, :onemklSdot,:Float32),
         (:dot, :onemklDdot,:Float64),
         (:dotc, :onemklCdotc, :ComplexF32),
         (:dotc, :onemklZdotc, :ComplexF64),
         (:dotu, :onemklCdotu, :ComplexF32),
         (:dotu, :onemklZdotu, :ComplexF64))
    @eval begin
        function $jname(n::Integer,
                         x::oneStridedArray{$elty},
                         y::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            result = oneArray{$elty}([0]);
            $fname(sycl_queue(queue), n, x, stride(x,1), y, stride(y,1), result)
            res = Array(result)
            return res[1]
        end
    end
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
            queue = global_queue(context(x), device(x))
            lda = max(1, stride(a,2))
            incx = stride(x,1)
            incy = stride(y,1)
            alpha = $elty(alpha)
            beta = $elty(beta)
            $fname(sycl_queue(queue), uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
            y
        end

        function sbmv(uplo::Char, k::Integer, alpha::Number,
                      a::oneStridedArray{$elty}, x::oneStridedArray{$elty})
            n = size(a,2)
            sbmv!(uplo, k, alpha, a, x, zero($elty), similar(x, $elty, n))
        end

        function sbmv(uplo::Char, k::Integer, a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            sbmv(uplo, k, one($elty), a, x)
        end
    end
end

for (fname, elty, celty) in ((:onemklCsscal, :Float32, :ComplexF32),
                             (:onemklZdscal, :Float64, :ComplexF64))
    @eval begin
        function scal!(n::Integer, 
                       alpha::$elty,
                       x::oneStridedArray{$celty})
            queue = global_queue(context(x), device(x))
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), m, n, alpha, x, stride(x,1), y, stride(y,1), a, max(1,stride(a,2)))
            a
        end
    end
end

#symv
for (fname, elty) in ((:onemklSsymv,:Float32),
                      (:onemklDsymv,:Float64))
    # Note that the complex symv are not BLAS but auiliary functions in LAPACK
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end

        function symv(uplo::Char, alpha::Number, A::oneStridedVecOrMat{$elty}, x::oneStridedVecOrMat{$elty})
                symv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function symv(uplo::Char, A::oneStridedVecOrMat{$elty}, x::oneStridedVecOrMat{$elty})
            symv(uplo, one($elty), A, x)
        end

    end
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
            queue = global_queue(context(x), device(x))
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, x, stride(x, 1), y, stride(y, 1))
            y
        end
    end
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), n, x, stride(x, 1), result)
            res = Array(result)
            return res[1]
        end
    end
end

## iamax
for (fname, elty) in
    ((:onemklDamax,:Float64),
     (:onemklSamax,:Float32),
     (:onemklZamax,:ComplexF64),
     (:onemklCamax,:ComplexF32))
    @eval begin
        function iamax(x::oneStridedArray{$elty})
            n = length(x)
            queue = global_queue(context(x), device(x))
            result = oneArray{Int64}([0]);
            $fname(sycl_queue(queue), n, x, stride(x, 1), result)
            return Array(result)[1]+1
        end
    end
end

## iamin
for (fname, elty) in
    ((:onemklDamin,:Float64),
     (:onemklSamin,:Float32),
     (:onemklZamin,:ComplexF64),
     (:onemklCamin,:ComplexF32))
    @eval begin
        function iamin(x::StridedArray{$elty})
            n = length(x)
            result = oneArray{Int64}([0]);
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue),n, x, stride(x, 1), result)
            return Array(result)[1]+1
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
            queue = global_queue(context(x), device(x))
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
            queue = global_queue(context(x), device(x))
            lda = max(1, stride(a,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(sycl_queue(queue), trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
            y
        end

        function gbmv(trans::Char,
                      m::Integer, 
                      kl::Integer,
                      ku::Integer,
                      alpha::Number,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            n = size(a,2)
            leny = trans == 'N' ? m : n
            queue = global_queue(context(x), device(x))
            gbmv!(trans, m, kl, ku, alpha, a, x, zero($elty), similar(x, $elty, leny))   
        end

        function gbmv(trans::Char,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      a::oneStridedArray{$elty},
                      x::oneStridedArray{$elty})
            queue = global_queue(context(x), device(x))
            gbmv(trans, m, kl, ku, one($elty), a, x)
        end
    end
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, trans, diag, n, k, A, lda, x, incx)
            x
        end

        function tbmv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      k::Integer,
                      A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            tbmv!(uplo, trans, diag, k, A, copy(x))
        end
    end
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, trans, diag, n, A, lda, x, incx)
            x
        end

        function trmv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            trmv!(uplo, trans, diag, A, copy(x))
        end
    end
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
            queue = global_queue(context(x), device(x))
            $fname(sycl_queue(queue), uplo, trans, diag, n, A, lda, x, incx)
            x
        end
        function trsv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      A::oneStridedVecOrMat{$elty},
                      x::oneStridedVecOrMat{$elty})
            trsv!(uplo, trans, diag, A, copy(x))
        end
    end
end

# level 3

for (fname, elty) in
        ((:onemklDgemm,:Float64),
         (:onemklSgemm,:Float32),
         (:onemklHgemm, :Float16),
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

            device(A) == device(B) == device(C) || error("Multi-device GEMM not supported")
            context(A) == context(B) == context(C) || error("Multi-context GEMM not supported")
            queue = global_queue(context(A), device(A))

            alpha = $elty(alpha)
            beta = $elty(beta)

            $fname(sycl_queue(queue), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end

        function gemm(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::oneStridedVecOrMat{$elty},
                      B::oneStridedVecOrMat{$elty})
            gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end

        function gemm(transA::Char,
                      transB::Char,
                      A::oneStridedVecOrMat{$elty},
                      B::oneStridedVecOrMat{$elty})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end
