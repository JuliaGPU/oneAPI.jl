using LinearAlgebra
import Adapt

@testset "constructors" begin
  xs = oneArray{Int}(undef, 2, 3)
  @test collect(oneArray([1 2; 3 4])) == [1 2; 3 4]
  @test testf(vec, rand(5,3))
  @test Base.elsize(xs) == sizeof(Int)
  @test oneArray{Int, 2}(xs) === xs

  @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
  @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

  @test collect(oneAPI.zeros(2, 2)) == zeros(2, 2)
  @test collect(oneAPI.ones(2, 2)) == ones(2, 2)

  @test collect(oneAPI.fill(0, 2, 2)) == zeros(2, 2)
  @test collect(oneAPI.fill(1, 2, 2)) == ones(2, 2)
end
