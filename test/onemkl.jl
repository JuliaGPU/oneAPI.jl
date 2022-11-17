using oneAPI
using oneAPI.oneMKL

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
    end
end

@testset "level 2" begin
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
    end
end
