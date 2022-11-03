using oneAPI
using oneAPI.oneMKL

@testset "level 1" begin
  @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
      a = convert.(T, [1.0, 2.0, -0.8, 5.0, 3.0])
      ca = oneArray(a)
      @test BLAS.iamax(a)  == oneMKL.iamax(ca)
      @test oneMKL.iamin(ca) == 3
  end
end
