using oneAPI.oneMKL
using LinearAlgebra

m = 20
n = 35
k = 13

eltypes=[Float32, ComplexF32]
const float64_supported = oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
if (float64_supported)
    append!(eltypes, [Float64, ComplexF64])
end

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
