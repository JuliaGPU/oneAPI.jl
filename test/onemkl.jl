using oneAPI
using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13

########################
@testset "level 1" begin
	@testset for T in eltypes
		if T <:oneMKL.onemklFloat
			@test testf(rmul!, rand(T,m), Ref(rand()))
		end
	end
end
