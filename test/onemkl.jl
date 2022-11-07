using oneAPI
using oneAPI.oneMKL
<<<<<<< HEAD
=======

>>>>>>> master
using LinearAlgebra

m = 20
n = 35
k = 13

<<<<<<< HEAD
#####
@testset "level 1" begin
    @testset for T in eltypes
        if T <:oneMKL.onemklFloat
            A = rand(T,m)
			B = rand(T, m)
            alpha = rand()
			@test testf(axpy!, alpha, A, B)
        end
    end
=======
############################################################################################
@testset "level 1" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        A = oneArray(rand(T, m))
        B = oneArray{T}(undef, m)
        oneMKL.copy!(m,A,B)
        @test Array(A) == Array(B)
    end # level 1 testset
>>>>>>> master
end
