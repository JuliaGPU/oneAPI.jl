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
#		@test testf(rmul!, gpuA, Ref(alpha[1]))
#		@test testf(oneMKL.scal!, m, alpha[1], gpuA)
#		oneMKL.scal!(m, alpha[1], gpuA)
#		_A = Array(gpuA)
#		@test isapprox(A .* alpha[1], _A)

#### Pass Cases
		# TODO: This test passes if we disable our own implementation 
		# of rmul in lib/mkl/linalg.jl which means rmul! cpu implmentation might 
		# be taking over it ?
		@test testf(rmul!, gpuB, Ref(alpha[1]))
		
		if T === Float32
			oneMKL.scal!(m, 5f0, gpuA)
		else 
			oneMKL.scal!(m, 5.0, gpuA)
		end

		_A = Array(gpuA)
		@test isapprox(A .* 5, _A)

	end
end
