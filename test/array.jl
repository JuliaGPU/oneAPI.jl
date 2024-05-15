using LinearAlgebra
import Adapt

@testset "constructors" begin
  xs = oneArray{Int}(undef, 2, 3)
  @test collect(oneArray([1 2; 3 4])) == [1 2; 3 4]
  @test testf(vec, rand(Float32, 5,3))
  @test Base.elsize(xs) == sizeof(Int)
  @test oneArray{Int, 2}(xs) === xs

  @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
  @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

  @test collect(oneAPI.zeros(Float32, 2, 2)) == zeros(Float32, 2, 2)
  @test collect(oneAPI.ones(Float32, 2, 2)) == ones(Float32, 2, 2)

  @test collect(oneAPI.fill(0, 2, 2)) == zeros(Int, 2, 2)
  @test collect(oneAPI.fill(1, 2, 2)) == ones(Int, 2, 2)
end

@testset "adapt" begin
  A = rand(Float32, 3, 3)
  dA = oneArray(A)
  @test Adapt.adapt(Array, dA) == A
  @test Adapt.adapt(oneArray, A) isa oneArray
  @test Array(Adapt.adapt(oneArray, A)) == A
end

@testset "reshape" begin
  A = [1 2 3 4
       5 6 7 8]
  gA = reshape(oneArray(A),1,8)
  _A = reshape(A,1,8)
  _gA = Array(gA)
  @test all(_A .== _gA)
  A = [1,2,3,4]
  gA = reshape(oneArray(A),4)
end

@testset "fill(::SubArray)" begin
  xs = oneAPI.zeros(Float32, 3)
  fill!(view(xs, 2:2), 1)
  @test Array(xs) == [0,1,0]
end

@testset "shared buffers & unsafe_wrap" begin
  a = oneVector{Int,oneL0.SharedBuffer}(undef, 2)

  # check that basic operations work on arrays backed by shared memory
  fill!(a, 40)
  a .+= 2
  @test Array(a) == [42, 42]

  # derive an Array object and test that the memory keeps in sync
  b = unsafe_wrap(Array, a)
  b[1] = 100
  @test Array(a) == [100, 42]
  oneAPI.@sync copyto!(a, 2, [200], 1, 1)
  @test b == [100, 200]
end

# https://github.com/JuliaGPU/CUDA.jl/issues/2191
@testset "preserving buffer types" begin
  a = oneVector{Int,oneL0.SharedBuffer}([1])
  @test oneAPI.buftype(a) == oneL0.SharedBuffer

  # unified-ness should be preserved
  b = a .+ 1
  @test oneAPI.buftype(b) == oneL0.SharedBuffer

  # when there's a conflict, we should defer to unified memory
  c = oneVector{Int,oneL0.HostBuffer}([1])
  d = oneVector{Int,oneL0.DeviceBuffer}([1])
  e = c .+ d
  @test oneAPI.buftype(e) == oneL0.SharedBuffer
end

@testset "resizing" begin
  a = oneArray([1,2,3])

  resize!(a, 3)
  @test length(a) == 3
  @test Array(a) == [1,2,3]

  resize!(a, 5)
  @test length(a) == 5
  @test Array(a)[1:3] == [1,2,3]

  resize!(a, 2)
  @test length(a) == 2
  @test Array(a)[1:2] == [1,2]

  b = oneArray{Int}(undef, 0)
  @test length(b) == 0
  resize!(b, 1)
  @test length(b) == 1
end
