# potrf
for (bname, fname, elty) in ((:onemklSpotrf_scratchpad_size, :onemklSpotrf, :Float32),
                             (:onemklDpotrf_scratchpad_size, :onemklDpotrf, :Float64),
                             (:onemklCpotrf_scratchpad_size, :onemklCpotrf, :ComplexF32),
                             (:onemklZpotrf_scratchpad_size, :onemklZpotrf, :ComplexF64))
    @eval begin
        function potrf!(uplo::Char,
                        A::oneStridedMatrix{$elty})
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), uplo, n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), uplo, n, A, lda, scratchpad, scratchpad_size)

            return A
        end
    end
end

# potrs
for (bname, fname, elty) in ((:onemklSpotrs_scratchpad_size, :onemklSpotrs, :Float32),
                             (:onemklDpotrs_scratchpad_size, :onemklDpotrs, :Float64),
                             (:onemklCpotrs_scratchpad_size, :onemklCpotrs, :ComplexF32),
                             (:onemklZpotrs_scratchpad_size, :onemklZpotrs, :ComplexF64))
    @eval begin
        function potrs!(uplo::Char,
                        A::oneStridedMatrix{$elty},
                        B::oneStridedVecOrMat{$elty})
            chkuplo(uplo)
            n = checksquare(A)
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B,2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), uplo, n, nrhs, lda, ldb)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), uplo, n, nrhs, A, lda, B, ldb, scratchpad, scratchpad_size)

            return B
        end
    end
end

# potri
for (bname, fname, elty) in ((:onemklSpotri_scratchpad_size, :onemklSpotri, :Float32),
                             (:onemklDpotri_scratchpad_size, :onemklDpotri, :Float64),
                             (:onemklCpotri_scratchpad_size, :onemklCpotri, :ComplexF32),
                             (:onemklZpotri_scratchpad_size, :onemklZpotri, :ComplexF64))
    @eval begin
        function potri!(uplo::Char,
                        A::oneStridedMatrix{$elty})
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), uplo, n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), uplo, n, A, lda, scratchpad, scratchpad_size)

            return A
        end
    end
end

#sytrf
for (bname, fname, elty) in ((:onemklSsytrf_scratchpad_size, :onemklSsytrf, :Float32),
                             (:onemklDsytrf_scratchpad_size, :onemklDsytrf, :Float64),
                             (:onemklCsytrf_scratchpad_size, :onemklCsytrf, :ComplexF32),
                             (:onemklZsytrf_scratchpad_size, :onemklZsytrf, :ComplexF64))
    @eval begin
        function sytrf!(uplo::Char,
                        A::oneStridedMatrix{$elty},
                        ipiv::oneStridedVector{Int64})
            chkuplo(uplo)
            n = checksquare(A)
            lda = max(1, stride(A, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), uplo, n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), uplo, n, A, lda, ipiv, scratchpad, scratchpad_size)

            return A, ipiv
        end

        function sytrf!(uplo::Char, A::oneStridedMatrix{$elty})
            n = checksquare(A)
            ipiv = oneVector{Int64}(undef, n)
            sytrf!(uplo, A, ipiv)
        end
    end
end

# getrf
for (bname, fname, elty) in ((:onemklSgetrf_scratchpad_size, :onemklSgetrf, :Float32),
                             (:onemklDgetrf_scratchpad_size, :onemklDgetrf, :Float64),
                             (:onemklCgetrf_scratchpad_size, :onemklCgetrf, :ComplexF32),
                             (:onemklZgetrf_scratchpad_size, :onemklZgetrf, :ComplexF64))
    @eval begin
        function getrf!(A::oneStridedMatrix{$elty})
            m, n = size(A)
            ipiv = oneVector{Int64}(undef, min(m, n))
            getrf!(A, ipiv)
        end

        function getrf!(A::oneStridedMatrix{$elty}, ipiv::oneStridedVector{Int64})
            m,n = size(A)
            lda = max(1, stride(A, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), m, n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), m, n, A, lda, ipiv, scratchpad, scratchpad_size)

            return A, ipiv
        end
    end
end

# getrs
for (bname, fname, elty) in ((:onemklSgetrs_scratchpad_size, :onemklSgetrs, :Float32),
                             (:onemklDgetrs_scratchpad_size, :onemklDgetrs, :Float64),
                             (:onemklCgetrs_scratchpad_size, :onemklCgetrs, :ComplexF32),
                             (:onemklZgetrs_scratchpad_size, :onemklZgetrs, :ComplexF64))
    @eval begin
        function getrs!(trans::Char,
                        A::oneStridedMatrix{$elty},
                        ipiv::oneStridedVector{Int64},
                        B::oneStridedVecOrMat{$elty})

            # Support transa = 'C' for real matrices
            trans = $elty <: Real && trans == 'C' ? 'T' : trans

            chktrans(trans)
            n = checksquare(A)
            if size(B, 1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match dimension of A, $n"))
            end
            if length(ipiv) != n
                throw(DimensionMismatch("length of ipiv, $(length(ipiv)), must match dimension of A, $n"))
            end
            nrhs = size(B, 2)
            lda  = max(1, stride(A, 2))
            ldb  = max(1, stride(B, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), trans, n, nrhs, lda, ldb)
            scratchpad = oneVector{UInt8}(undef, scratchpad_size)
            $fname(sycl_queue(queue), trans, n, nrhs, A, lda, ipiv, B, ldb, scratchpad, scratchpad_size)

            return B
        end
    end
end

# getri
for (bname, fname, elty) in ((:onemklSgetri_scratchpad_size, :onemklSgetri, :Float32),
                             (:onemklDgetri_scratchpad_size, :onemklDgetri, :Float64),
                             (:onemklCgetri_scratchpad_size, :onemklCgetri, :ComplexF32),
                             (:onemklZgetri_scratchpad_size, :onemklZgetri, :ComplexF64))
    @eval begin
        function getri!(A::oneStridedMatrix{$elty}, ipiv::oneStridedVector{Int64})
            n = checksquare(A)
            lda = max(1, stride(A, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), n, A, lda, ipiv, scratchpad, scratchpad_size)

            return A
        end
    end
end

# geqrf
for (bname, fname, elty) in ((:onemklSgeqrf_scratchpad_size, :onemklSgeqrf, :Float32),
                             (:onemklDgeqrf_scratchpad_size, :onemklDgeqrf, :Float64),
                             (:onemklCgeqrf_scratchpad_size, :onemklCgeqrf, :ComplexF32),
                             (:onemklZgeqrf_scratchpad_size, :onemklZgeqrf, :ComplexF64))
    @eval begin
        function geqrf!(A::oneStridedMatrix{$elty})
            m, n = size(A)
            tau = oneVector{$elty}(undef, min(m, n))
            geqrf!(A, tau)
        end

        function geqrf!(A::oneStridedMatrix{$elty}, tau::oneVector{$elty})
            m,n = size(A)
            lda = max(1, stride(A, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), m, n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), m, n, A, lda, tau, scratchpad, scratchpad_size)

            return A, tau
        end
    end
end

# ormqr and unmqr
for (bname, fname, elty) in ((:onemklSormqr_scratchpad_size, :onemklSormqr, :Float32),
                             (:onemklDormqr_scratchpad_size, :onemklDormqr, :Float64),
                             (:onemklCunmqr_scratchpad_size, :onemklCunmqr, :ComplexF32),
                             (:onemklZunmqr_scratchpad_size, :onemklZunmqr, :ComplexF64))
    @eval begin
        function ormqr!(
            side::Char, trans::Char, A::oneStridedMatrix{$elty},
            tau::oneStridedVector{$elty}, C::oneStridedVecOrMat{$elty})

            trans = ($elty <: Real && trans == 'C') ? 'T' : trans
            chkside(side)
            chktrans(trans)

            m, n = (ndims(C) == 2) ? size(C) : (size(C, 1), 1)
            k = length(tau)
            mA  = size(A, 1)

            side == 'L' && m != mA && throw(DimensionMismatch(
                "for a left-sided multiplication, the first dimension of C, $m, must equal the second dimension of A, $mA"))
            side == 'R' && n != mA && throw(DimensionMismatch(
                "for a right-sided multiplication, the second dimension of C, $m, must equal the second dimension of A, $mA"))
            side == 'L' && k > m && throw(DimensionMismatch(
                "invalid number of reflectors: k = $k should be ≤ m = $m"))
            side == 'R' && k > n && throw(DimensionMismatch(
                "invalid number of reflectors: k = $k should be ≤ n = $n"))

            lda = max(1, stride(A, 2))
            ldc = max(1, stride(C, 2))

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), side, trans, m, n, k, lda, ldc)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), side, trans, m, n, k, A, lda, tau, C, ldc, scratchpad, scratchpad_size)

            return C
        end
    end
end

## orgqr and ungqr
for (bname, fname, elty) in ((:onemklSorgqr_scratchpad_size, :onemklSorgqr, :Float32),
                             (:onemklDorgqr_scratchpad_size, :onemklDorgqr, :Float64),
                             (:onemklCungqr_scratchpad_size, :onemklCungqr, :ComplexF32),
                             (:onemklZungqr_scratchpad_size, :onemklZungqr, :ComplexF64))
    @eval begin
        function orgqr!(A::oneStridedMatrix{$elty}, tau::oneStridedVector{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            k = length(tau)

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), m, n, k, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), m, n, k, A, lda, tau, scratchpad, scratchpad_size)

            return A
        end
    end
end

#gebrd
for (bname, fname, elty, relty) in ((:onemklSgebrd_scratchpad_size, :onemklSgebrd, :Float32, :Float32),
                                    (:onemklDgebrd_scratchpad_size, :onemklDgebrd, :Float64, :Float64),
                                    (:onemklCgebrd_scratchpad_size, :onemklCgebrd, :ComplexF32, :Float32),
                                    (:onemklZgebrd_scratchpad_size, :onemklZgebrd, :ComplexF64, :Float64))
    @eval begin
        function gebrd!(A::oneStridedMatrix{$elty})
            m, n = size(A)
            lda  = max(1, stride(A, 2))

            k = min(m, n)
            D = oneVector{$relty}(undef, k)
            E = oneVector{$elty}(undef, k)
            tauq = oneVector{$elty}(undef, k)
            taup = oneVector{$elty}(undef, k)

            queue = global_queue(context(A), device(A))
            scratchpad_size = $bname(sycl_queue(queue), m, n, lda)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), m, n, A, lda, D, E, tauq, taup, scratchpad, scratchpad_size)

            A, D, E, tauq, taup
        end
    end
end

# getrf_batch
for (bname, fname, elty) in ((:onemklSgetrf_batch_scratchpad_size, :onemklSgetrf_batch, :Float32),
                             (:onemklDgetrf_batch_scratchpad_size, :onemklDgetrf_batch, :Float64),
                             (:onemklCgetrf_batch_scratchpad_size, :onemklCgetrf_batch, :ComplexF32),
                             (:onemklZgetrf_batch_scratchpad_size, :onemklZgetrf_batch, :ComplexF64))
    @eval begin
        function getrf_batched!(A::Vector{<:oneMatrix{$elty}})
            group_count = length(A)
            group_sizes = ones(Int64, group_count)
            m = [size(A[i], 1) for i=1:group_count]
            n = [size(A[i], 2) for i=1:group_count]
            lda = [max(1, stride(A[i], 2)) for i=1:group_count]
            ipiv = [oneVector{Int64}(undef, min(m[i], n[i])) for i=1:group_count]
            Aptrs = unsafe_batch(A)
            ipivptrs = unsafe_batch(ipiv)

            queue = global_queue(context(A[1]), device(A[1]))
            scratchpad_size = $bname(sycl_queue(queue), m, n, lda, group_count, group_sizes)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), m, n, Aptrs, lda, ipivptrs, group_count, group_sizes, scratchpad, scratchpad_size)

            unsafe_free!(Aptrs)
            unsafe_free!(ipivptrs)

            return ipiv, A
        end
    end
end

# getri_batch
for (bname, fname, elty) in ((:onemklSgetri_batch_scratchpad_size, :onemklSgetri_batch, :Float32),
                             (:onemklDgetri_batch_scratchpad_size, :onemklDgetri_batch, :Float64),
                             (:onemklCgetri_batch_scratchpad_size, :onemklCgetri_batch, :ComplexF32),
                             (:onemklZgetri_batch_scratchpad_size, :onemklZgetri_batch, :ComplexF64))
    @eval begin
        function getri_batched!(A::Vector{<:oneMatrix{$elty}}, ipiv::Vector{<:oneVector{Int64}})
            group_count = length(A)
            group_sizes = ones(Int64, group_count)
            n = [checksquare(A[i]) for i=1:group_count]
            lda = [max(1, stride(A[i], 2)) for i=1:group_count]
            Aptrs = unsafe_batch(A)
            ipivptrs = unsafe_batch(ipiv)

            queue = global_queue(context(A[1]), device(A[1]))
            scratchpad_size = $bname(sycl_queue(queue), n, lda, group_count, group_sizes)
            scratchpad = oneVector{$elty}(undef, scratchpad_size)
            $fname(sycl_queue(queue), n, Aptrs, lda, ipivptrs, group_count, group_sizes, scratchpad, scratchpad_size)

            unsafe_free!(Aptrs)
            unsafe_free!(ipivptrs)

            return ipiv, A
        end
    end
end

# LAPACK
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        LinearAlgebra.LAPACK.potrf!(uplo::Char, A::oneStridedMatrix{$elty}) = oneMKL.potrf!(uplo, A)
        LinearAlgebra.LAPACK.potrs!(uplo::Char, A::oneStridedMatrix{$elty}, B::oneStridedVecOrMat{$elty}) = oneMKL.potrs!(uplo, A, B)
        LinearAlgebra.LAPACK.sytrf!(uplo::Char, A::oneStridedMatrix{$elty}) = oneMKL.sytrf!(uplo, A)
        LinearAlgebra.LAPACK.sytrf!(uplo::Char, A::oneStridedMatrix{$elty}, ipiv::oneStridedVector{Int64}) = oneMKL.sytrf!(uplo, A, ipiv)
        LinearAlgebra.LAPACK.geqrf!(A::oneStridedMatrix{$elty}) = oneMKL.geqrf!(A)
        LinearAlgebra.LAPACK.geqrf!(A::oneStridedMatrix{$elty}, tau::oneStridedVector{$elty}) = oneMKL.geqrf!(A, tau)
        LinearAlgebra.LAPACK.getrf!(A::oneStridedMatrix{$elty}) = oneMKL.getrf!(A)
        LinearAlgebra.LAPACK.getrf!(A::oneStridedMatrix{$elty}, ipiv::oneStridedVector{Int64}) = oneMKL.getrf!(A, ipiv)
        LinearAlgebra.LAPACK.getrs!(trans::Char, A::oneStridedMatrix{$elty}, ipiv::oneStridedVector{Int64}, B::oneStridedVecOrMat{$elty}) = oneMKL.getrs!(trans, A, ipiv, B)
        LinearAlgebra.LAPACK.ormqr!(side::Char, trans::Char, A::oneStridedMatrix{$elty}, tau::oneStridedVector{$elty}, C::oneStridedVecOrMat{$elty}) = oneMKL.ormqr!(side, trans, A, tau, C)
        LinearAlgebra.LAPACK.orgqr!(A::oneStridedMatrix{$elty}, tau::oneStridedVector{$elty}) = oneMKL.orgqr!(A, tau)
    end
end
