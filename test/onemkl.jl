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

        # testing oneMKL max and min
        a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
        ca = oneArray(a)
        @test BLAS.iamax(a)  == oneMKL.iamax(ca)
        @test oneMKL.iamin(ca) == 3

        # testing swap
        x = rand(T, m)
        y = rand(T, m)
        dx = oneArray(x)
        dy = oneArray(y)
        oneMKL.swap!(m, dx, dy)
        h_x = collect(dx)
        h_y = collect(dy)
        @test h_x ≈ y
        @test h_y ≈ x
    end # level 1 testset
end
