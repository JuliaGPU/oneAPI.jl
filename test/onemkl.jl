#using oneAPI.oneMKL
#using LinearAlgebra

m = 20
n = 35
k = 13

const eltypes=[Float32, ComplexF32]
const float64_supported = oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
if (float64_supported)
    append!(eltypes, [Float64, ComplexF64])
end

########################
@testset "level 1" begin
	@testset for T in eltypes
	    A = oneArray(rand(T, m))
	    B = A
	    alpha = 2
	    B = B.* 2
	    println("CPU Result")
	    @show B

	    oneMKL.scal!(m, alpha, A)
	    println("GPU Result")
	    @show A

	    @test Array(A) == Array(B)
	end
end
