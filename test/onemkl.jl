using oneAPI
using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13

########################
@testset "level 1" begin
	@testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
		alpha = rand()
		A = rand(T,m)
		@test testf(rmul!, A, Ref(alpha[1]))
	end
end
