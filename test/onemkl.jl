using oneAPI
using oneAPI.oneMKL

using LinearAlgebra

m = 20

############################################################################################
@testset "level 1" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        A = oneArray(rand(T, m))
        B = oneArray{T}(undef, m)
        oneMKL.copy!(m,A,B)
        @test Array(A) == Array(B)

        @testset "nrm2" begin
            # Test nrm2 primitive
            @test testf(norm, rand(T,m))
        end

        @testset "iamax/iamin" begin
            # testing oneMKL max and min
            a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
            ca = oneArray(a)
            @test BLAS.iamax(a)  == oneMKL.iamax(ca)
            @test oneMKL.iamin(ca) == 3
        end
    end # level 1 testset
end
