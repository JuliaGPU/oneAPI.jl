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
            oneMKL.trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(dB)
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
    @testset for T in [Float16, Float32, Float64, ComplexF32, ComplexF64]
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