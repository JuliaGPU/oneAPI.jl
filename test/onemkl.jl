using oneAPI
using oneAPI.oneMKL: band, bandex

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
            alpha = rand(T,1)
            @test testf(axpy!, alpha[1], rand(T,m), rand(T,m))
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

            @testset "gbmv!" begin
                # Test: y = alpha * A * x + beta * y
                y = rand(T, m)
                d_y = oneArray(y)
                oneMKL.gbmv!('N', m, kl, ku, alpha, d_Ab, d_x, beta, d_y)
                BLAS.gbmv!('N', m, kl, ku, alpha, Ab, x, beta, y)
                h_y = Array(d_y)
                @test y ≈ h_y

                # Test: y = alpha * transpose(A) * x + beta * y
                x = rand(T, n)
                d_x = oneArray(x)
                y = rand(T,m)
                d_y = oneArray(y)
                oneMKL.gbmv!('T', m, kl, ku, alpha, d_Ab, d_y, beta, d_x)
                BLAS.gbmv!('T', m, kl, ku, alpha, Ab, y, beta, x)
                h_x = Array(d_x)
                @test x ≈ h_x

                # Test: y = alpha * A'*x + beta * y
                x = rand(T,n)
                d_x = oneArray(x)
                y = rand(T,m)
                d_y = oneArray(y)
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
                #@test_throws DimensionMismatch CUBLAS.trsv!('U','N','N',dA,CUDA.rand(elty,m+1))
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
                @testset "symv!" begin
                    # generate vectors
                    y = rand(T,m)
                    # copy to device
                    dy = oneArray(y)
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