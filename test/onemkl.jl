using oneAPI
using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13

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
end
