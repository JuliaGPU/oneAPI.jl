using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13


########################
@testset "level 1" begin
	@testset for T in eltypes
		A = rand(T, m)
		gpuA = oneArray(A)
		if T === Float32
			oneMKL.scal!(m, 5f0, gpuA)
		else
			oneMKL.scal!(m, 5.0, gpuA)
		end
		_A = Array(gpuA)
	    @test isapprox(A .* 5, _A)
	end
end
