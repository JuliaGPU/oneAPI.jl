using oneAPI
using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13

################
@testset "level 1" begin
	@testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
		A = rand(T,m)
		gpuA = oneArray(A)
		res = oneArray(A)
		#oneMKL.nrm2(m, gpuA, res)
		@show res
	end
end
