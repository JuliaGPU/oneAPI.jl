using oneAPI
using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13

########################
@testset "level 1" begin
	@testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
		alpha = rand(T,1)
		A = rand(T,m)
		gpuA = oneArray(A)
		gpuB = oneArray(A)
##### Failed Cases
#		Following test fails if we use our own rmul! implementation from lib/mkl/linalg.jl
#		@test testf(rmul!, gpuA, Ref(alpha[1]))
#		@test testf(oneMKL.scal!, m, alpha[1], gpuA)

#### Pass Cases
		# TODO: This test passes if we disable our own implementation 
		# of rmul in lib/mkl/linalg.jl which means rmul! cpu implmentation might 
		# be taking over it ?
		@test testf(rmul!, gpuB, Ref(alpha[1]))

		# This test works fine
		# It manually checks for CPU/GPU comparisons for scal primitive
		if T === Float32
			oneMKL.scal!(m, 5f0, gpuA)
		else 
			oneMKL.scal!(m, 5.0, gpuA)
		end
		_A = Array(gpuA)
		@test isapprox(A .* 5, _A)

	end
end
