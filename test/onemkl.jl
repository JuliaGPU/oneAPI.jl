if Sys.iswindows()
@warn "Skipping unsupported oneKML tests"
else

using oneAPI
using oneAPI.oneMKL: band, bandex, oneSparseMatrixCSR, oneSparseMatrixCOO, oneSparseMatrixCSC

using SparseArrays
using LinearAlgebra

m = 20
n = 35
k = 13

############################################################################################
@testset "level 1" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        @testset "copy" begin
            A = oneArray(rand(T, m))
            B = oneArray{T}(undef, m)
            oneMKL.copy!(m,A,B)
            @test Array(A) == Array(B)
        end

        @testset "axpy" begin
            alpha = rand(T)
            @test testf(axpy!, alpha, rand(T,m), rand(T,m))
        end

        @testset "axpby" begin
            alpha = rand(T)
            beta = rand(T)
            @test testf(axpby!, alpha, rand(T,m), beta, rand(T,m))
        end

        @testset "rotate" begin
            @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
            @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
        end

        @testset "reflect" begin
            @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
            @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
        end

        @testset "scal" begin
            # Test scal primitive [alpha/x: F32, F64, CF32, CF64]
            alpha = rand(T,1)
            @test testf(rmul!, rand(T,m), alpha[1])

            # Test scal primitive [alpha - F32, F64, x - CF32, CF64]
            A = rand(T,m)
            gpuA = oneArray(A)
            if T === ComplexF32
                alphaf32 = rand(Float32, 1)
                oneMKL.scal!(m, alphaf32[1], gpuA)
                @test Array(A .* alphaf32[1]) ≈ Array(gpuA)
            end

            if T === ComplexF64
                alphaf64 = rand(Float64, 1)
                oneMKL.scal!(m, alphaf64[1], gpuA)
                @test Array(A .* alphaf64[1]) ≈ Array(gpuA)
            end
        end

        @testset "nrm2" begin
            @test testf(norm, rand(T,m))
        end

        @testset "iamax/iamin" begin
            a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
            ca = oneArray(a)
            @test BLAS.iamax(a) == oneMKL.iamax(ca)
            @test oneMKL.iamin(ca) == 3
        end

        @testset "swap" begin
            x = rand(T, m)
            y = rand(T, m)
            dx = oneArray(x)
            dy = oneArray(y)
            oneMKL.swap!(m, dx, dy)
            @test Array(dx) == y
            @test Array(dy) == x
        end

        @testset "dot" begin
            @test testf(dot, rand(T,m), rand(T,m))
            if T == ComplexF32 || T == ComplexF64
                @test testf(oneMKL.dotu, m,
                            oneArray(rand(T,m)),
                            oneArray(rand(T,m)))
            end
        end

        @testset "asum" begin
            @test testf(BLAS.asum, rand(T,m))

        end
    end

    @testset for T in [Float16, ComplexF16]
        alpha = rand(T,1)
        A = oneArray(rand(T, m))
        B = oneArray{T}(undef, m)
        oneMKL.copy!(m,A,B)
        @test Array(A) == Array(B)

        @test testf(axpy!, alpha[1], rand(T,m), rand(T,m))
        @test testf(norm, rand(T,m))
        @test testf(dot, rand(T, m), rand(T, m))
        @test testf(*, transpose(rand(T, m)), rand(T,m))
        @test testf(*, rand(T, m)', rand(T,m))
        @test testf(rmul!, rand(T,m), alpha[1])

        if T <: ComplexF16
            @test testf(dot, rand(T, m), rand(T, m))
            x = rand(T, m)
            y = rand(T, m)
            dx = oneArray(x)
            dy = oneArray(y)
            dz = dot(dx, dy)
            z = dot(x, y)
            @test dz ≈ z
        end
    end
end

@testset "level 2" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)

        @testset "gemv" begin
            @test testf(*, rand(T, m, n), rand(T, n))
            @test testf(*, transpose(rand(T, m, n)), rand(T, m))
            @test testf(*, rand(T, m, n)', rand(T, m))
            x = rand(T, m)
            A = rand(T, m, m + 1 )
            y = rand(T, m)
            dx = oneArray(x)
            dA = oneArray(A)
            dy = oneArray(y)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            A = rand(T, m + 1, m )
            dA = oneArray(A)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            x = rand(T, m)
            A = rand(T, n, m)
            dx = oneArray(x)
            dA = oneArray(A)
            alpha = rand(T)
            dy = oneMKL.gemv('N', alpha, dA, dx)
            hy = collect(dy)
            @test hy ≈ alpha * A * x
            dy = oneMKL.gemv('N', dA, dx)
            hy = collect(dy)
            @test hy ≈ A * x
        end

        @testset "banded methods" begin
            # bands
            ku = 2
            kl = 3
            # generate banded matrix
            A = rand(T, m,n)
            A = bandex(A, kl, ku)

            # get packed format
            Ab = band(A, kl, ku)
            d_Ab = oneArray(Ab)
            x = rand(T, n)
            d_x = oneArray(x)
            synchronize()

            @testset "gbmv!" begin
                # Test: y = alpha * A * x + beta * y
                y = rand(T, m)
                d_y = oneArray(y)
                synchronize()
                oneMKL.gbmv!('N', m, kl, ku, alpha, d_Ab, d_x, beta, d_y)
                BLAS.gbmv!('N', m, kl, ku, alpha, Ab, x, beta, y)
                h_y = Array(d_y)
                @test y ≈ h_y

                # Test: y = alpha * transpose(A) * x + beta * y
                x = rand(T, n)
                d_x = oneArray(x)
                y = rand(T,m)
                d_y = oneArray(y)
                synchronize()
                oneMKL.gbmv!('T', m, kl, ku, alpha, d_Ab, d_y, beta, d_x)
                BLAS.gbmv!('T', m, kl, ku, alpha, Ab, y, beta, x)
                h_x = Array(d_x)
                @test x ≈ h_x

                # Test: y = alpha * A'*x + beta * y
                x = rand(T,n)
                d_x = oneArray(x)
                y = rand(T,m)
                d_y = oneArray(y)
                synchronize()
                oneMKL.gbmv!('C', m, kl, ku, alpha, d_Ab, d_y, beta, d_x)
                BLAS.gbmv!('C', m, kl, ku, alpha, Ab, y, beta, x)
                h_x = Array(d_x)
                @test x ≈ h_x

                # Test: alpha=1 version without y
                d_y = oneMKL.gbmv('N', m, kl, ku, d_Ab, d_x)
                y = BLAS.gbmv('N', m, kl, ku, Ab, x)
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "gbmv" begin
                # test y = alpha*A*x
                x = rand(T,n)
                d_x = oneArray(x)
                d_y = oneMKL.gbmv('N', m, kl, ku, alpha, d_Ab, d_x)
                y = zeros(T,m)
                y = BLAS.gbmv('N',m,kl,ku,alpha,Ab,x)
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            A = rand(T,m,m)
            A = A + A'
            nbands = 3
            @test m >= 1+nbands
            A = bandex(A,nbands,nbands)
            # convert to 'upper' banded storage format
            AB = band(A,0,nbands)
            # construct x
            x = rand(T,m)
            d_AB = oneArray(AB)
            d_x = oneArray(x)
            if T <:Union{ComplexF32,ComplexF64}
                @testset "hbmv!" begin
                    y = rand(T,m)
                    d_y = oneArray(y)
                    # hbmv!
                    oneMKL.hbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
                    y = alpha*(A*x) + beta*y
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
                @testset "hbmv" begin
                    d_y = oneMKL.hbmv('U',nbands,d_AB,d_x)
                    y = A*x
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
            else
                @testset "sbmv!" begin
                    y = rand(T,m)
                    d_y = oneArray(y)
                    # sbmv!
                    oneMKL.sbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
                    y = alpha*(A*x) + beta*y
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
                @testset "sbmv" begin
                    d_y = oneMKL.sbmv('U',nbands,d_AB,d_x)
                    y = A*x
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
            end
            # generate triangular matrix
            A = rand(T,m,m)
            # restrict to 3 bands
            nbands = 3
            @test m >= 1+nbands
            A = bandex(A,0,nbands)
            # convert to 'upper' banded storage format
            AB = band(A,0,nbands)
            d_AB = oneArray(AB)
            @testset "tbmv!" begin
                y = rand(T, m)
                # move to host
                d_y = oneArray(y)
                # tbmv!
                oneMKL.tbmv!('U','N','N',nbands,d_AB,d_y)
                y = A*y
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end
            @testset "tbmv" begin
                # tbmv
                d_y = oneMKL.tbmv('U','N','N',nbands,d_AB,d_x)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

        end

        @testset "ger!" begin
            A = rand(T,m,m)
            x = rand(T,m)
            y = rand(T,m)
            dA = oneArray(A)
            dx = oneArray(x)
            dy = oneArray(y)
            # perform rank one update
            dB = copy(dA)
            oneMKL.ger!(alpha,dx,dy,dB)
            B = (alpha*x)*y' + A
            # move to host and compare
            hB = Array(dB)
            @test B ≈ hB
        end

        @testset "Triangular" begin
            @testset "trmv!" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = copy(dx)
                # execute trmv!
                oneMKL.trmv!('U','N','N',dA,d_y)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trmv" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = copy(dx)
                d_y = oneMKL.trmv('U','N','N',dA,dx)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv!" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = copy(dx)
                # execute trsv!
                oneMKL.trsv!('U','N','N',dA,d_y)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = oneMKL.trsv('U','N','N',dA,dx)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv (adjoint)" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = oneMKL.trsv('U','C','N',dA,dx)
                y = adjoint(A)\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end

            @testset "trsv (transpose)" begin
                sA = rand(T,m,m)
                sA = sA + transpose(sA)
                A = triu(sA)
                dA = oneArray(A)
                x = rand(T, m)
                dx = oneArray(x)
                d_y = oneMKL.trsv('U','T','N',dA,dx)
                y = transpose(A)\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end
        end
    end

    @testset for T in intersect(eltypes, [ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        @testset "hemv!" begin
            A = rand(T,m,n)
            dA = oneArray(A)
            sA = rand(T,m,m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            x = rand(T,m)
            dx = oneArray(x)
            y = rand(T,m)
            dy = oneArray(y)
            synchronize()

            # execute on host
            BLAS.hemv!('U',alpha,hA,x,beta,y)
            # execute on device
            oneMKL.hemv!('U',alpha,dhA,dx,beta,dy)

            # compare results
            hy = Array(dy)
            @test y ≈ hy
        end

        @testset "hemv" begin
            A = rand(T,m,n)
            dA = oneArray(A)
            sA = rand(T,m,m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            x = rand(T,m)
            dx = oneArray(x)
            y = rand(T,m)
            dy = oneArray(y)
            synchronize()

            y = BLAS.hemv('U',hA,x)
            # execute on device
            dy = oneMKL.hemv('U',dhA,dx)
            # compare results
            hy = Array(dy)
            @test y ≈ hy
        end

        @testset "her!" begin
            A = rand(T,m,n)
            dA = oneArray(A)
            sA = rand(T,m,m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            x = rand(T,m)
            dx = oneArray(x)
            dB = copy(dhA)
            # perform rank one update
            oneMKL.her!('U',real(alpha),dx,dB)
            B = (real(alpha)*x)*x' + hA
            # move to host and compare upper triangles
            hB = Array(dB)
            B = triu(B)
            hB = triu(hB)
            @test B ≈ hB
        end

        @testset "her2!" begin
            A = rand(T,m,n)
            dA = oneArray(A)
            sA = rand(T,m,m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            x = rand(T,m)
            dx = oneArray(x)
            y = rand(T,m)
            dy = oneArray(y)
            dB = copy(dhA)
            oneMKL.her2!('U',real(alpha),dx,dy,dB)
            B = (real(alpha)*x)*y' + y*(real(alpha)*x)' + hA
            # move to host and compare upper triangles
            hB = Array(dB)
            B = triu(B)
            hB = triu(hB)
            @test B ≈ hB
        end
    end

    @testset "symmetric" begin
        @testset for T in intersect(eltypes, [Float32, Float64])
            alpha = rand(T)
            beta = rand(T)
            A = rand(T,m,m)
            A = A + A'
            nbands = 3
            @test m >= 1+nbands
            A = bandex(A,nbands,nbands)
            # convert to 'upper' banded storage format
            AB = band(A,0,nbands)
            # construct x
            x = rand(T,m)
            d_AB = oneArray(AB)
            d_x = oneArray(x)
            @testset "symv tests" begin
                x = rand(T,m)
                sA = rand(T, m, m)
                sA = sA + transpose(sA)
                dsA = oneArray(sA)
                dx = oneArray(x)
                synchronize()

                @testset "symv!" begin
                    # generate vectors
                    y = rand(T,m)
                    # copy to device
                    dy = oneArray(y)
                    synchronize()
                    # execute on host
                    BLAS.symv!('U',alpha,sA,x,beta,y)
                    # execute on device
                    oneMKL.symv!('U',alpha,dsA,dx,beta,dy)
                    # compare results
                    hy = Array(dy)
                    @test y ≈ hy
                end

                @testset "symv" begin
                    y = BLAS.symv('U',sA,x)
                    # execute on device
                    dy = oneMKL.symv('U',dsA,dx)
                    # compare results
                    hy = Array(dy)
                    @test y ≈ hy
                end
            end

            @testset "syr!" begin
                x = rand(T,m)
                sA = rand(T, m, m)
                sA = sA + transpose(sA)
                dsA = oneArray(sA)
                dx = oneArray(x)
                dB = copy(dsA)
                oneMKL.syr!('U',alpha,dx,dB)
                B = (alpha*x)*transpose(x) + sA
                # move to host and compare upper triangles
                hB = Array(dB)
                B = triu(B)
                hB = triu(hB)
                @test B ≈ hB
            end
        end
    end
end

@testset "level 3" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        B = rand(T,m,n)
        C = rand(T,m,n)
        Bbad = rand(T,m+1,n+1)
        d_B = oneArray(B)
        d_C = oneArray(C)
        d_Bbad = oneArray(Bbad)
        sA = rand(T,m,m)
        sA = sA + transpose(sA)
        dsA = oneArray(sA)

        @testset "symm!" begin
            oneMKL.symm!('L','U',alpha,dsA,d_B,beta,d_C)
            C = (alpha*sA)*B + beta*C
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.symm!('L','U',alpha,dsA,d_Bbad,beta,d_C)
        end

        @testset "symm" begin
            d_C = oneMKL.symm('L','U',dsA,d_B)
            C = sA*B
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.symm('L','U',dsA,d_Bbad)
        end

        @testset "syrk" begin
            A = rand(T,m,k)
            d_A = oneArray(A)
            d_C = oneMKL.syrk('U','N',d_A)
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end

        A = rand(T,m,k)
        B = rand(T,m,k)
        Bbad = rand(T,m+1,k+1)
        C = rand(T,m,m)
        C = C + transpose(C)
        # move to device
        d_A = oneArray(A)
        d_B = oneArray(B)
        d_Bbad = oneArray(Bbad)
        d_C = oneArray(C)
        @testset "syr2k!" begin
            # compute
            C = alpha*(A*transpose(B) + B*transpose(A)) + beta*C
            oneMKL.syr2k!('U','N',alpha,d_A,d_B,beta,d_C)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch oneMKL.syr2k!('U','N',alpha,d_A,d_Bbad,beta,d_C)
        end

        @testset "syr2k" begin
            C = alpha*(A*transpose(B) + B*transpose(A))
            d_C = oneMKL.syr2k('U','N',alpha,d_A,d_B)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
        end

        @testset "trmm!" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*A*B
            oneMKL.trmm!('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(dB)
            @test C ≈ h_C
        end

        @testset "trmm" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*A*B
            dC = oneMKL.trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end

        @testset "left trsm!" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(A\B)
            dC = copy(dB)
            oneMKL.trsm!('L','U','N','N',alpha,dA,dC)
            @test C ≈ Array(dC)
        end

        @testset "left trsm" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(A\B)
            dC = oneMKL.trsm('L','U','N','N',alpha,dA,dB)
            @test C ≈ Array(dC)
        end

        @testset "left trsm (adjoint)" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(adjoint(A)\B)
            dC = oneMKL.trsm('L','U','C','N',alpha,dA,dB)
            @test C ≈ Array(dC)
        end

        @testset "left trsm (transpose)" begin
            A = triu(rand(T, m, m))
            B = rand(T,m,n)
            dA = oneArray(A)
            dB = oneArray(B)
            C = alpha*(transpose(A)\B)
            dC = oneMKL.trsm('L','U','T','N',alpha,dA,dB)
            @test C ≈ Array(dC)
        end

        let A = rand(T, m,m), B = triu(rand(T, m, m)), alpha = rand(T)
            dA = oneArray(A)
            dB = oneArray(B)

            @testset "right trsm!" begin
                C = alpha*(A/B)
                dC = copy(dA)
                oneMKL.trsm!('R','U','N','N',alpha,dB,dC)
                @test C ≈ Array(dC)
            end

            @testset "right trsm" begin
                C = alpha*(A/B)
                dC = oneMKL.trsm('R','U','N','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
            @testset "right trsm (adjoint)" begin
                C = alpha*(A/adjoint(B))
                dC = oneMKL.trsm('R','U','C','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
            @testset "right trsm (transpose)" begin
                C = alpha*(A/transpose(B))
                dC = oneMKL.trsm('R','U','T','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
        end
        if T <:Union{ComplexF32,ComplexF64}
            @testset "hemm!" begin
                B = rand(T,m,n)
                C = rand(T,m,n)
                d_B = oneArray(B)
                d_C = oneArray(C)
                hA = rand(T,m,m)
                hA = hA + hA'
                dhA = oneArray(hA)
                # compute
                C = alpha*(hA*B) + beta*C
                oneMKL.hemm!('L','L',alpha,dhA,d_B,beta,d_C)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end

            @testset "hemm" begin
                B = rand(T,m,n)
                C = rand(T,m,n)
                d_B = oneArray(B)
                d_C = oneArray(C)
                hA = rand(T,m,m)
                hA = hA + hA'
                dhA = oneArray(hA)

                C = hA*B
                d_C = oneMKL.hemm('L','U',dhA,d_B)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end

            @testset "herk!" begin
                B = rand(T,m,n)
                C = rand(T,m,n)
                d_B = oneArray(B)
                d_C = oneArray(C)
                hA = rand(T,m,m)
                hA = hA + hA'
                dhA = oneArray(hA)
                A = rand(T,m,k)
                d_A = oneArray(A)
                d_C = oneArray(dhA)
                oneMKL.herk!('U','N',real(alpha),d_A,real(beta),d_C)
                C = real(alpha)*(A*A') + real(beta)*hA
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(C)
                @test C ≈ h_C
            end

            @testset "herk" begin
                B = rand(T,m,n)
                C = rand(T,m,n)
                d_B = oneArray(B)
                d_C = oneArray(C)
                hA = rand(T,m,m)
                hA = hA + hA'
                dhA = oneArray(hA)
                A = rand(T,m,k)
                d_A = oneArray(A)
                d_C = oneMKL.herk('U','N',d_A)
                C = A*A'
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(C)
                @test C ≈ h_C
            end

            @testset "her2k!" begin
                A = rand(T,m,k)
                B = rand(T,m,k)
                Bbad = rand(T,m+1,k+1)
                C = rand(T,m,m)
                C = C + transpose(C)
                # move to device
                d_A = oneArray(A)
                d_B = oneArray(B)
                d_Bbad = oneArray(Bbad)
                d_C = oneArray(C)
                elty1 = T
                elty2 = real(T)
                # generate parameters
                α = rand(elty1)
                β = rand(elty2)
                C = C + C'
                d_C = oneArray(C)
                C = α*(A*B') + conj(α)*(B*A') + β*C
                oneMKL.her2k!('U','N',α,d_A,d_B,β,d_C)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
                @test_throws DimensionMismatch oneMKL.her2k!('U','N',α,d_A,d_Bbad,β,d_C)
            end

            @testset "her2k" begin
                A = rand(T,m,k)
                B = rand(T,m,k)
                Bbad = rand(T,m+1,k+1)
                C = rand(T,m,m)
                C = C + transpose(C)
                # move to device
                d_A = oneArray(A)
                d_B = oneArray(B)
                d_Bbad = oneArray(Bbad)
                d_C = oneArray(C)
                C = A*B' + B*A'
                d_C = oneMKL.her2k('U','N',d_A,d_B)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
        end
    end

    @testset for T in intersect(eltypes, [Float16, Float32, Float64, ComplexF32, ComplexF64])
        @testset "gemm!" begin
            alpha = rand(T)
            beta = rand(T)
            A = rand(T,m,k)
            B = rand(T,k,n)
            Bbad = rand(T,k+1,n+1)
            C1 = rand(T,m,n)
            C2 = copy(C1)
            d_A = oneArray(A)
            d_B = oneArray(B)
            d_Bbad = oneArray(Bbad)
            d_C1 = oneArray(C1)
            d_C2 = oneArray(C2)
            hA = rand(T,m,m)
            hA = hA + hA'
            dhA = oneArray(hA)
            sA = rand(T,m,m)
            sA = sA + transpose(sA)
            dsA = oneArray(sA)
            oneMKL.gemm!('N','N',alpha,d_A,d_B,beta,d_C1)
            mul!(d_C2, d_A, d_B)
            h_C1 = Array(d_C1)
            h_C2 = Array(d_C2)
            C1 = (alpha*A)*B + beta*C1
            C2 = A*B
            # compare
            @test C1 ≈ h_C1
            @test C2 ≈ h_C2
            @test_throws ArgumentError mul!(dhA, dhA, dsA)
            @test_throws DimensionMismatch mul!(d_C1, d_A, dsA)

            d_c = oneMKL.gemm('N', 'N', d_A, d_B)
            C = A * B
            C2 = d_A * d_B
            h_C = Array(d_c)
            h_C2 = Array(C2)
            @test C ≈ h_C
            @test C ≈ h_C2
        end
    end
end

@testset "Batch Primitives" begin
    @testset for T in intersect(eltypes, [Float16, Float32, Float64, ComplexF32, ComplexF64])
        alpha = rand(T)
        beta = rand(T)
        group_count = 10
        @testset "Gemm Batch" begin
            # generate matrices
            bA = [rand(T,m,k) for i in 1:group_count]
            bB = [rand(T,k,n) for i in 1:group_count]
            bC = [rand(T,m,n) for i in 1:group_count]
            # move to device
            bd_A = oneArray{T, 2}[]
            bd_B = oneArray{T, 2}[]
            bd_C = oneArray{T, 2}[]
            bd_bad = oneArray{T, 2}[]
            for i in 1:length(bA)
                push!(bd_A,oneArray(bA[i]))
                push!(bd_B,oneArray(bB[i]))
                push!(bd_C,oneArray(bC[i]))
                if i < length(bA) - 2
                    push!(bd_bad,oneArray(bC[i]))
                end
            end

            @testset "gemm_batched!" begin
                # C = (alpha*A)*B + beta*C
                oneMKL.gemm_batched!('N','N',alpha,bd_A,bd_B,beta,bd_C)
                for i in 1:length(bd_C)
                    bC[i] = (alpha*bA[i])*bB[i] + beta*bC[i]
                    h_C = Array(bd_C[i])
                    #compare
                    @test bC[i] ≈ h_C
                end
                @test_throws DimensionMismatch oneMKL.gemm_batched!('N','N',alpha,bd_A,bd_bad,beta,bd_C)
            end

            @testset "gemm_batched" begin
                bd_C = oneMKL.gemm_batched('N','N',bd_A,bd_B)
                for i in 1:length(bA)
                    bC = bA[i]*bB[i]
                    h_C = Array(bd_C[i])
                    @test bC ≈ h_C
                end
                @test_throws DimensionMismatch oneMKL.gemm_batched('N','N',alpha,bd_A,bd_bad)
            end
        end

        if T <:Union{Float32, Float64, ComplexF32, ComplexF64}
            @testset "Trsm Batch" begin
                @testset "trsm_batched!" begin
                    bA = [rand(T,m,m) for i in 1:group_count]
                    map!((x) -> triu(x), bA, bA)
                    bB = [rand(T,m,n) for i in 1:group_count]
                    bBbad = [rand(T,m,n) for i in 1:(group_count-1)]
                    # move to device
                    bd_A = oneArray{T, 2}[]
                    bd_B = oneArray{T, 2}[]
                    bd_Bbad = oneArray{T, 2}[]
                    for i in 1:length(bA)
                        push!(bd_A,oneArray(bA[i]))
                        push!(bd_B,oneArray(bB[i]))
                    end
                    for i in 1:length(bBbad)
                        push!(bd_Bbad,oneArray(bBbad[i]))
                    end
                    # compute
                    oneMKL.trsm_batched!('L','U','N','N',alpha,bd_A,bd_B)
                    @test_throws DimensionMismatch oneMKL.trsm_batched!('L','U','N','N',alpha,bd_A,bd_Bbad)
                    # move to host and compare
                    for i in 1:length(bd_B)
                        bC = alpha*(bA[i]\bB[i])
                        h_C = Array(bd_B[i])
                        #compare
                        @test bC ≈ h_C
                    end
                end

                @testset "trsm_batched" begin
                    # generate parameter alpha = rand(elty)
                    # generate matrices
                    bA = [rand(T,m,m) for i in 1:group_count]
                    map!((x) -> triu(x), bA, bA)
                    bB = [rand(T,m,n) for i in 1:group_count]
                    # move to device
                    bd_A = oneArray{T, 2}[]
                    bd_B = oneArray{T, 2}[]
                    for i in 1:length(bA)
                        push!(bd_A,oneArray(bA[i]))
                        push!(bd_B,oneArray(bB[i]))
                    end
                    # compute
                    bd_C = oneMKL.trsm_batched('L','U','N','N',alpha,bd_A,bd_B)
                    # move to host and compare
                    for i in 1:length(bd_C)
                        bC = alpha*(bA[i]\bB[i])
                        h_C = Array(bd_C[i])
                        @test bC ≈ h_C
                    end
                end
            end
        end
    end
end

@testset "gemm_batch_strided" begin
    @testset for elty in intersect(eltypes, [Float16, Float32, Float64, ComplexF32, ComplexF64])
        nbatch = 10
        alpha = rand(elty)
        beta = rand(elty)
        @testset "gemm_strided_batched!" begin
            bA = rand(elty, m, k, nbatch)
            bB = rand(elty, k, n, nbatch)
            bC = rand(elty, m, n, nbatch)
            bbad = rand(elty, m+1, n+1, nbatch)
            # move to device
            bd_A = oneArray{elty, 3}(bA)
            bd_B = oneArray{elty, 3}(bB)
            bd_C = oneArray{elty, 3}(bC)
            bd_bad = oneArray{elty, 3}(bbad)

            oneMKL.gemm_strided_batched!('N', 'N', alpha, bd_A, bd_B, beta, bd_C)
            for i in 1:nbatch
                bC[:, :, i] = (alpha * bA[:, :, i]) * bB[:, :, i] + beta * bC[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C

            @test_throws DimensionMismatch oneMKL.gemm_strided_batched!('N', 'N', alpha, bd_A, bd_B, beta, bd_bad)
        end

        @testset "gemm_strided_batched" begin
            # Host buffers
            bA = rand(elty, m, k, nbatch)
            bB = rand(elty, k, n, nbatch)
            bC = rand(elty, m, n, nbatch)
            bbad = rand(elty, m+1, n+1, nbatch)
            # Move host data to device
            bd_A = oneArray{elty, 3}(bA)
            bd_B = oneArray{elty, 3}(bB)
            bd_C = oneArray{elty, 3}(bC)
            bd_bad = oneArray{elty, 3}(bbad)
            # Compute oneMKL strided batch
            bd_C = oneMKL.gemm_strided_batched('N', 'N', bd_A, bd_B)
            #Compute Host
            for i in 1:nbatch
                bC[:, :, i] = bA[:, :, i] * bB[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C

            # generate matrices
            bA = rand(elty, k, m, nbatch)
            bB = rand(elty, k, n, nbatch)
            bC = zeros(elty, m, n, nbatch)
            # move to device
            bd_A = oneArray{elty, 3}(bA)
            bd_B = oneArray{elty, 3}(bB)
            bd_C = oneMKL.gemm_strided_batched('T', 'N', bd_A, bd_B)
            for i in 1:nbatch
                bC[:, :, i] = transpose(bA[:, :, i]) * bB[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C
            @test_throws DimensionMismatch oneMKL.gemm_strided_batched('N', 'N', alpha, bd_A, bd_bad)
        end
    end
end

@testset "SPARSE" begin
    @testset "$T" for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        @testset "oneSparseMatrixCSR" begin
            for S in (Int32, Int64)
                A = sprand(T, 20, 10, 0.5)
                A = SparseMatrixCSC{T, S}(A)
                B = oneSparseMatrixCSR(A)
                A2 = SparseMatrixCSC(B)
                @test A == A2
            end
        end

        @testset "oneSparseMatrixCSC" begin
            (T isa Complex) && continue
            for S in (Int32, Int64)
                A = sprand(T, 20, 10, 0.5)
                A = SparseMatrixCSC{T, S}(A)
                B = oneSparseMatrixCSC(A)
                A2 = SparseMatrixCSC(B)
                @test A == A2
            end
        end

        @testset "oneSparseMatrixCOO" begin
            for S in (Int32, Int64)
                A = sprand(T, 20, 10, 0.5)
                A = SparseMatrixCSC{T, S}(A)
                B = oneSparseMatrixCOO(A)
                A2 = SparseMatrixCSC(B)
                @test A == A2
            end
        end

        @testset "sparse gemv" begin
            @testset  "$SparseMatrix" for SparseMatrix in (oneSparseMatrixCOO, oneSparseMatrixCSR, oneSparseMatrixCSC)
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    (T <: Complex) && (SparseMatrix == oneSparseMatrixCSC) && continue
                    A = sprand(T, 20, 10, 0.5)
                    x = transa == 'N' ? rand(T, 10) : rand(T, 20)
                    y = transa == 'N' ? rand(T, 20) : rand(T, 10)

                    dA = SparseMatrix(A)
                    dx = oneVector{T}(x)
                    dy = oneVector{T}(y)

                    alpha = rand(T)
                    beta = rand(T)
                    oneMKL.sparse_optimize_gemv!(transa, dA)
                    oneMKL.sparse_gemv!(transa, alpha, dA, dx, beta, dy)
                    @test alpha * opa(A) * x + beta * y ≈ collect(dy)
                end
            end
        end

        @testset "sparse gemm" begin
            @testset  "$SparseMatrix" for SparseMatrix in (oneSparseMatrixCSR, oneSparseMatrixCSC)
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    @testset "transb = $transb" for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        (transb == 'N') || continue
                        (T <: Complex) && (SparseMatrix == oneSparseMatrixCSC) && continue
                        A = sprand(T, 10, 10, 0.5)
                        B = transb == 'N' ? rand(T, 10, 2) : rand(T, 2, 10)
                        C = rand(T, 10, 2)

                        dA = SparseMatrix(A)
                        dB = oneMatrix{T}(B)
                        dC = oneMatrix{T}(C)

                        alpha = rand(T)
                        beta = rand(T)
                        oneMKL.sparse_optimize_gemm!(transa, dA)
                        oneMKL.sparse_gemm!(transa, transb, alpha, dA, dB, beta, dC)
                        @test alpha * opa(A) * opb(B) + beta * C ≈ collect(dC)
                    end
                end
            end
        end

        @testset "sparse symv" begin
            @testset  "$SparseMatrix" for SparseMatrix in (oneSparseMatrixCSR, oneSparseMatrixCSC)
                @testset "uplo = $uplo" for uplo in ('L', 'U')
                    (T <: Complex) && (SparseMatrix == oneSparseMatrixCSC) && continue
                    A = sprand(T, 10, 10, 0.5)
                    A = A + A'
                    x = rand(T, 10)
                    y = rand(T, 10)

                    dA = uplo == 'L' ? SparseMatrix(A |> tril) : SparseMatrix(A |> triu)
                    dx = oneVector{T}(x)
                    dy = oneVector{T}(y)

                    alpha = rand(T)
                    beta = rand(T)
                    oneMKL.sparse_symv!(uplo, alpha, dA, dx, beta, dy)
                    @test alpha * A * x + beta * y ≈ collect(dy)
                end
            end
        end

        @testset "sparse trmv" begin
            @testset  "$SparseMatrix" for SparseMatrix in (oneSparseMatrixCSR, oneSparseMatrixCSC)
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    for (uplo, diag, wrapper) in [('L', 'N', LowerTriangular), ('L', 'U', UnitLowerTriangular),
                                                  ('U', 'N', UpperTriangular), ('U', 'U', UnitUpperTriangular)]
                        (transa == 'N') || continue
                        (T <: Complex) && (SparseMatrix == oneSparseMatrixCSC) && continue
                        A = sprand(T, 10, 10, 0.5)
                        x = rand(T, 10)
                        y = rand(T, 10)

                        B = uplo == 'L' ? tril(A) : triu(A)
                        B = diag == 'U' ? B - Diagonal(B) + I : B
                        dA = SparseMatrix(B)
                        dx = oneVector{T}(x)
                        dy = oneVector{T}(y)

                        alpha = rand(T)
                        beta = rand(T)

                        oneMKL.sparse_optimize_trmv!(uplo, transa, diag, dA)
                        oneMKL.sparse_trmv!(uplo, transa, diag, alpha, dA, dx, beta, dy)
                        @test alpha * wrapper(opa(A)) * x + beta * y ≈ collect(dy)
                    end
                end
            end
        end

        @testset "sparse trsv" begin
            @testset  "$SparseMatrix" for SparseMatrix in (oneSparseMatrixCSR, oneSparseMatrixCSC)
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    for (uplo, diag, wrapper) in [('L', 'N', LowerTriangular), ('L', 'U', UnitLowerTriangular),
                                                  ('U', 'N', UpperTriangular), ('U', 'U', UnitUpperTriangular)]
                        (transa == 'N') || continue
                        (T <: Complex) && (SparseMatrix == oneSparseMatrixCSC) && continue
                        alpha = rand(T)
                        A = rand(T, 10, 10) + I
                        A = sparse(A)
                        x = rand(T, 10)
                        y = rand(T, 10)

                        B = uplo == 'L' ? tril(A) : triu(A)
                        B = diag == 'U' ? B - Diagonal(B) + I : B
                        dA = SparseMatrix(B)
                        dx = oneVector{T}(x)
                        dy = oneVector{T}(y)

                        oneMKL.sparse_optimize_trsv!(uplo, transa, diag, dA)
                        oneMKL.sparse_trsv!(uplo, transa, diag, alpha, dA, dx, dy)
                        y = wrapper(opa(A)) \ (alpha * x)
                        @test y ≈ collect(dy)
                    end
                end
            end
        end

        @testset "sparse trsm" begin
            @testset  "$SparseMatrix" for SparseMatrix in (oneSparseMatrixCSR, oneSparseMatrixCSC)
                @testset "transa = $transa" for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    @testset "transx = $transx" for (transx, opx) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        (transx != 'N') && continue
                        for (uplo, diag, wrapper) in [('L', 'N', LowerTriangular), ('L', 'U', UnitLowerTriangular),
                                                      ('U', 'N', UpperTriangular), ('U', 'U', UnitUpperTriangular)]
                            (transa == 'N') || continue
                            (T <: Complex) && (SparseMatrix == oneSparseMatrixCSC) && continue
                            alpha = rand(T)
                            A = rand(T, 10, 10) + I
                            A = sparse(A)
                            X = transx == 'N' ? rand(T, 10, 4) : rand(T, 4, 10)
                            Y = rand(T, 10, 4)

                            B = uplo == 'L' ? tril(A) : triu(A)
                            B = diag == 'U' ? B - Diagonal(B) + I : B
                            dA = SparseMatrix(B)
                            dX = oneMatrix{T}(X)
                            dY = oneMatrix{T}(Y)

                            oneMKL.sparse_optimize_trsm!(uplo, transa, diag, dA)
                            oneMKL.sparse_trsm!(uplo, transa, transx, diag, alpha, dA, dX, dY)
                            Y = wrapper(opa(A)) \ (alpha * opx(X))
                            @test Y ≈ collect(dY)

                            oneMKL.sparse_optimize_trsm!(uplo, transa, diag, 4, dA)
                        end
                    end
                end
            end
        end
    end
end

@testset "LAPACK" begin
    @testset "$elty" for elty in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        m = 15
        n = 10
        p = 5

        @testset "geqrf!" begin
            A = rand(elty, m, n)
            d_A = oneArray(A)
            d_A, tau = oneMKL.geqrf!(d_A)
            tau_c = zeros(elty, n)
            LinearAlgebra.LAPACK.geqrf!(A, tau_c)
            @test tau_c ≈ Array(tau)
        end

        @testset "geqrf! -- orgqr!" begin
            A = rand(elty, m, n)
            dA = oneArray(A)
            dA, τ = oneMKL.geqrf!(dA)
            oneMKL.orgqr!(dA, τ)
            @test dA' * dA ≈ I
        end

        @testset "ormqr!" begin
            @testset "side = $side" for side in ['L', 'R']
                @testset "trans = $trans" for (trans, op) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    (trans == 'T') && (elty <: Complex) && continue
                    A = rand(elty, m, n)
                    dA = oneArray(A)
                    dA, dτ = oneMKL.geqrf!(dA)

                    hI = Matrix{elty}(I, m, m)
                    dI = oneArray(hI)
                    dH = oneMKL.ormqr!(side, 'N', dA, dτ, dI)
                    @test dH' * dH ≈ I

                    C = side == 'L' ? rand(elty, m, n) : rand(elty, n, m)
                    dC = oneArray(C)
                    dD = side == 'L' ? op(dH) * dC : dC * op(dH)

                    oneMKL.ormqr!(side, trans, dA, dτ, dC)
                    @test dC ≈ dD
                end
            end
        end

        @testset "potrf! -- potrs!" begin
            A = rand(elty,n,n)
            A = A*A' + I
            B = rand(elty,n,p)
            d_A = oneArray(A)
            d_B = oneArray(B)

            oneMKL.potrf!('L',d_A)
            oneMKL.potrs!('U',d_A,d_B)
            LAPACK.potrf!('L',A)
            LAPACK.potrs!('U',A,B)
            @test B ≈ collect(d_B)
        end

        # @testset "sytrf!" begin
        #     A = rand(elty,n,n)
        #     A = A + A'
        #     d_A = oneArray(A)
        #     d_A, d_ipiv = oneMKL.sytrf!('U',d_A)
        #     h_A = collect(d_A)
        #     h_ipiv = collect(d_ipiv)
        #     A, ipiv = LAPACK.sytrf!('U',A)
        #     @test ipiv == h_ipiv
        #     @test A ≈ h_A
        # end

        @testset "getrf! -- getri!" begin
            A = rand(elty, m, m)
            d_A = oneArray(A)
            d_A, d_ipiv = oneMKL.getrf!(d_A)
            h_A, ipiv = LAPACK.getrf!(A)
            @test h_A ≈ Array(d_A)

            d_A = oneMKL.getri!(d_A, d_ipiv)
            h_A = LAPACK.getri!(h_A, ipiv)
            @test h_A ≈ Array(d_A)
        end

        @testset "getrf_batched! -- getri_batched!" begin
            bA = [rand(elty, m, m) for i in 1:p]
            d_bA = oneMatrix{elty}[]
            for i in 1:p
                push!(d_bA, oneMatrix(bA[i]))
            end

            d_ipiv, d_bA = oneMKL.getrf_batched!(d_bA)
            h_bA = [collect(d_bA[i]) for i in 1:p]

            ipiv = Vector{Int64}[]
            for i = 1:p
                _, ipiv_i, info = LAPACK.getrf!(bA[i])
                push!(ipiv, ipiv_i)
                @test bA[i] ≈ h_bA[i]
            end

            d_ipiv, d_bA = oneMKL.getri_batched!(d_bA, d_ipiv)
            h_bA = [collect(d_bA[i]) for i in 1:p]
            for i = 1:p
                LAPACK.getri!(bA[i], ipiv[i])
                @test bA[i] ≈ h_bA[i]
            end
        end

        # @testset "getrs_batched!" begin
        #     bA = [rand(elty, m, m) for i in 1:p]
        #     bB = [rand(elty, m, n) for i in 1:p]
        #     d_bA = oneMatrix{elty}[]
        #     d_bB = oneMatrix{elty}[]
        #     for i in 1:p
        #         push!(d_bA, oneMatrix(bA[i]))
        #         push!(d_bB, oneMatrix(bB[i]))
        #     end

        #     d_ipiv, d_bA = oneMKL.getrf_batched!(d_bA)
        #     d_bX = oneMKL.getrs_batched!(d_bA, d_ipiv, d_bB)
        #     h_bX = [collect(d_bX[i]) for i in 1:p]
        #     for i = 1:p
        #         @test bA[i] * hbX[i] ≈ bB[i]
        #     end
        # end

        @testset "potrf_batched! -- potrs_batched!" begin
            A = [rand(elty,n,n) for i = 1:p]
            A = [A[i]' * A[i] + I for i = 1:p]
            B = [rand(elty,n,p) for i = 1:p]
            d_A = oneMatrix{elty}[]
            d_B = oneMatrix{elty}[]
            for i in 1:p
                push!(d_A, oneMatrix(A[i]))
                push!(d_B, oneMatrix(B[i]))
            end

            oneMKL.potrf_batched!(d_A)
            oneMKL.potrs_batched!(d_A, d_B)
            for i = 1:p
                LAPACK.potrf!('L', A[i])
                LAPACK.potrs!('L', A[i], B[i])
                @test B[i] ≈ collect(d_B[i])
            end
        end

        @testset "geqrf_batched! -- -- orgqr_batched!" begin
            A = [rand(elty,m,n) for i in 1:p]
            d_A = oneMatrix{elty}[]
            for i in 1:p
                push!(d_A, oneMatrix(A[i]))
            end

            d_tau, d_A = oneMKL.geqrf_batched!(d_A)
            oneMKL.orgqr_batched!(d_A, d_tau)
            for d_Ai in d_A
                @test d_Ai' * d_Ai ≈ I
            end
        end

        @testset "gebrd!" begin
            A = rand(elty,m,n)
            d_A = oneArray(A)
            d_A, d_D, d_E, d_tauq, d_taup = oneMKL.gebrd!(d_A)
            h_A = collect(d_A)
            h_D = collect(d_D)
            h_E = collect(d_E)
            h_tauq = collect(d_tauq)
            h_taup = collect(d_taup)
            A,d,e,q,p = LAPACK.gebrd!(A)
            @test A ≈ h_A
            @test d ≈ h_D
            @test e[min(m,n)-1] ≈ h_E[min(m,n)-1]
            @test q ≈ h_tauq
            @test p ≈ h_taup
        end

        @testset "gesvd!" begin
            A = rand(elty,m,n)
            d_A = oneMatrix(A)
            U, Σ, Vt = oneMKL.gesvd!('A', 'A', d_A)
            @test A ≈ collect(U[:,1:n] * Diagonal(Σ) * Vt)

            for jobu in ('A', 'S', 'N', 'O')
                for jobvt in ('A', 'S', 'N', 'O')
                    (jobu == 'A') && (jobvt == 'A') && continue
                    (jobu == 'O') && (jobvt == 'O') && continue
                    d_A = oneMatrix(A)
                    U2, Σ2, Vt2 = oneMKL.gesvd!(jobu, jobvt, d_A)
                    @test Σ ≈ Σ2
                end
            end
        end

        @testset "syevd! -- heevd!" begin
            @testset "uplo = $uplo" for uplo in ('L', 'U')
                A = rand(elty,n,n)
                B = A + A'
                A = uplo == 'L' ? tril(B) : triu(B)
                d_A = oneMatrix(A)
                W, V = elty <: Real ? oneMKL.syevd!('V', uplo, d_A) : oneMKL.heevd!('V', uplo, d_A)
                @test B ≈ collect(V * Diagonal(W) * V')

                d_A = oneMatrix(A)
                d_W = elty <: Real ? oneMKL.syevd!('N', uplo, d_A) : oneMKL.heevd!('N', uplo, d_A)
            end
        end

        @testset "sygvd! -- hegvd!" begin
            A = rand(elty,m,m)
            B = rand(elty,m,m)
            A = A*A' + I
            B = B*B' + I
            d_A = oneArray(A)
            d_B = oneArray(B)
            d_W, d_VA, d_VB = elty <: Real ? oneMKL.sygvd!(1, 'V','U', d_A, d_B) : oneMKL.hegvd!(1, 'V','U', d_A, d_B)
            h_W = collect(d_W)
            h_VA = collect(d_VA)
            h_VB = collect(d_VB)
            Eig = eigen(Hermitian(A), Hermitian(B))
            @test Eig.values ≈ h_W
            @test A * h_VA ≈ B * h_VA * Diagonal(h_W) rtol=1e-4
            @test h_VA' * B * h_VA ≈ I
        end
    end
end

end # oneMKL tests
