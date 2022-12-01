using oneAPI
using oneAPI.oneMKL

using LinearAlgebra

m = 20

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
