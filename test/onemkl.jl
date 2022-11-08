using oneAPI
using oneAPI.oneMKL

using LinearAlgebra

m = 20 
n = 35
k = 13

############################################################################################
@testset "level 1" begin
    @testset for T in intersect(eltypes, [Float32, Float64, ComplexF32, ComplexF64])
        A = oneArray(rand(T, m))
        B = oneArray{T}(undef, m)
        oneMKL.copy!(m,A,B)
        @test Array(A) == Array(B)

		# Test scal primitive [alpha/x: F32, F64, CF32, CF64]
		alpha = rand(T,1)
		@test testf(rmul!, rand(T,m), alpha[1])

		# Test scal primitive [alpha - F32, F64, x - CF32, CF64] 
		A = rand(T,m)
		gpuA = oneArray(A)
		if T === ComplexF32
			alphaf32 = rand(Float32, 1)
			B = A .* alphaf32[1]
			oneMKL.scal!(m, alphaf32[1], gpuA)
			@test Array(B) == Array(gpuA)
		end

		if T === ComplexF64
			alphaf64 = rand(Float64, 1)
			B = A .* alphaF64[1]
			oneMKL.scal!(m, alphaf64[1], gpuA)
			@test Array(B) == Array(gpuA)
		end
		
    end # level 1 testset
end
