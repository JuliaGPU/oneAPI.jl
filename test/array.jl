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

@testset "derived array lifetime" begin
  parent = oneArray{UInt8}(undef, 1)

  # Construct through an ephemeral derived array, whose finalizer shares the same
  # reference-counted allocation with the returned view.
  function ephemeral_derived_view(parent)
    intermediate = @view parent[:]
    @view intermediate[:]
  end

  derived = ephemeral_derived_view(parent)
  @test derived isa oneArray{UInt8}

  # Exercise finalization while deriving. Without preserving the immediate parent in
  # GPUArrays.derive, its finalizer can mark the DataRef as freed before it is copied.
  if Threads.nthreads() > 1
    stop_gc = Threads.Atomic{Bool}(false)
    gc_task = Threads.@spawn while !stop_gc[]
      GC.gc(false)
      yield()
    end
    try
      Threads.@threads for _ in 1:min(Threads.nthreads(), 4)
        for _ in 1:100
          a = ephemeral_derived_view(parent)
          oneAPI.unsafe_free!(a)
        end
      end
    finally
      stop_gc[] = true
      wait(gc_task)
    end
  end
end

@testset "reinterpret of view with non-aligned offset" begin
  # reinterpreting a view to a larger element type where the byte offset
  # is not a multiple of the new element size
  a = oneArray(Int32[1,2,3,4,5,6,7,8,9])
  v = view(a, 2:7)  # offset of 1 Int32 = 4 bytes
  r = reinterpret(Int64, v)  # Int64 = 8 bytes; 4 is not a multiple of 8
  @test Array(r) == reinterpret(Int64, @view Array(a)[2:7])
end

@testset "aliasing" begin
  x = oneArray([1, 2])
  y = view(x, 2:2)
  @test Base.mightalias(x, x)
  @test Base.mightalias(x, y)
  z = view(x, 1:1)
  @test Base.mightalias(x, z)
  @test !Base.mightalias(y, z)

  a = copy(y)::typeof(x)
  @test !Base.mightalias(x, a)
  b = Base.unaliascopy(y)::typeof(y)
  @test !Base.mightalias(x, b)

  # contiguous views are oneArrays with an offset into the parent's memory,
  # which should still alias wrapped arrays (like SubArrays) of that memory
  x = oneArray(1:16)
  @test Base.mightalias(view(x, 2:16), view(x, 15:-1:1))
  @test Base.mightalias(view(x, 1:2:15), view(x, 2:9))
  @test Base.mightalias(view(x, 2:16), view(reinterpret(Int32, x), 1:2:31))
  @test !Base.mightalias(view(x, 2:16), view(oneArray(1:16), 15:-1:1))

  # so in-place broadcasts between them should make a copy first
  n = 2^20
  x = oneArray{Float32}(1:n)
  view(x, 2:n) .= view(x, n-1:-1:1)
  @test Array(x) == [1; n-1:-1:1]

  # empty arrays alias nothing, also on Julia 1.10
  @test !Base.mightalias(oneArray(Int[]), oneArray(Float32[]))
  @test !Base.mightalias(view(oneArray(zeros(Float32, 2, 0)), 1:1, :), oneArray(Int[]))

  # disjoint parts of one array may be each other's source and destination
  x = oneArray(collect(1:10))
  @test Array(sum!(view(x, 1:1), view(x, 2:10))) == [54]
  @test Array(cumsum!(view(x, 1:5), view(x, 6:10))) == cumsum(6:10)
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

  # the same works for arrays backed by host memory
  c = oneVector{Int,oneL0.HostBuffer}([1, 2])
  d = unsafe_wrap(Array, c)
  @test d == [1, 2]
  d[1] = 100
  @test Array(c) == [100, 2]
end

@testset "reductions of host-accessible arrays" begin
  for B in (oneL0.SharedBuffer, oneL0.HostBuffer)
    a = oneArray{Float32, 1, B}(fill(1.0f0, 1024))
    @test sum(a) == 1024
    @test maximum(a) == 1
  end
end

# https://github.com/JuliaGPU/oneAPI.jl/issues/661: kernels have to declare indirect access,
# or an allocation that reuses the address of a freed one isn't visible to them. Whether a
# reused shared allocation reads as zeros depends on allocator state, so churn through
# enough allocations (two live ones, freed in alternating order, plus collections of the
# reductions' outputs) to hit it reliably.
@testset "reusing freed $B allocations" for B in (oneL0.DeviceBuffer, oneL0.SharedBuffer, oneL0.HostBuffer)
    results = map(1:100) do i
        a = oneArray{Float32, 1, B}(fill(1.0f0, 1024))
        b = oneArray{Float32, 1, B}(fill(2.0f0, 1024))
        r = (sum(a), maximum(b))
        oneAPI.unsafe_free!(isodd(i) ? a : b)
        oneAPI.unsafe_free!(isodd(i) ? b : a)
        i % 10 == 0 && GC.gc(false)
        r
    end
    @test all(==((1024, 2)), results)
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

# Reductions and scans come from GPUArrays (AcceleratedKernels). oneAPI used to work around
# Intel problems in its own kernels; these tests check the cases those workarounds covered.

@testset "strided reductions" begin
    # The Aurora LTS stack miscompiled strided global reads in oneAPI's reduction kernel, both
    # for strided inputs (`a == transpose(b)`) and for reductions whose innermost reduced axis
    # is strided (dim 1 kept). Int32 keeps the comparison exact.
    A = rand(Int32(1):Int32(4), 64, 64)
    dA = oneArray(A)
    @test dA == transpose(oneArray(collect(transpose(A))))
    @test sum(transpose(dA)) == sum(A)
    @test Array(sum(transpose(dA); dims=1)) == sum(transpose(A); dims=1)
    for (sz, dts) in (
            ((2, 512, 64), ((1, 3), (2,), (3,), (2, 3), (1, 2, 3), 1)),
            ((3, 256, 48), ((1, 3), (2,), (1, 2, 3))),
            ((7, 300, 20), ((1, 3), (2,), (3,))),
            ((2, 128, 8, 32), ((1, 3), (1, 4), (2, 4), (1, 2, 4))),
        )
        A = rand(Int32(1):Int32(4), sz...)
        dA = oneArray(A)
        for dt in dts
            @test Array(sum(dA; dims = dt)) == sum(A; dims = dt)
        end
    end
end

@testset "narrow integer reduction with overflowing map" begin
    # IGC < 2.34.4+2 kept the result of a 16-bit `mad` in the accumulator at full width, so
    # the signed `max` compared the unwrapped value (JuliaGPU/oneAPI.jl#670, TGL and DG2).
    A = Complex{Int16}[22264-2672im -3184-13367im 20267+16808im -29600+21497im;
                       -2+6036im -13334-32552im -4391+23096im 1804+13481im;
                       -19764+32524im 20734-5522im 30645+17973im 21749-26470im]
    mk = A -> view(permutedims(A), [1, 2, 3, 4], :)
    R = Base.mapreducedim!(abs2, max, zeros(Int16, 1, 3), mk(A))
    dR = Base.mapreducedim!(abs2, max, oneArray(zeros(Int16, 1, 3)), mk(oneArray(A)))
    @test Array(dR) == R
end

@testset "sub-word reductions" begin
    # Writing 1- and 2-byte values to local memory clobbered adjacent bytes on some Intel GPUs
    for T in (Bool, Int8, UInt8, Int16, UInt16), n in (31, 257, 4097, 1_000_003)
        A = T == Bool ? rand(Bool, n) : rand(T(0):T(1), n)
        dA = oneArray(A)
        @test reduce(|, dA) == reduce(|, A)
        @test reduce(xor, dA) == reduce(xor, A)
        @test maximum(dA) == maximum(A)
        @test minimum(dA) == minimum(A)
    end
    for T in (Bool, Int8, Int16), (sz, dims) in (((64, 1000), 1), ((64, 1000), 2), ((17, 33, 65), (1, 3)))
        A = T == Bool ? rand(Bool, sz...) : rand(T(0):T(1), sz...)
        @test Array(reduce(xor, oneArray(A); dims, init=zero(T))) == reduce(xor, A; dims, init=zero(T))
        @test Array(maximum(oneArray(A); dims)) == maximum(A; dims)
    end
end

@testset "scans" begin
    # Scans with a block size of 128 or more were wrong on some Intel GPUs
    for T in (Int32, Float32), n in (127, 128, 129, 256, 1000, 100_000, 1_000_000)
        A = rand(T(0):T(3), n)
        @test Array(cumsum(oneArray(A))) == cumsum(A)
    end
    A = rand(Int32(0):Int32(3), 300, 700)
    @test Array(cumsum(oneArray(A); dims=1)) == cumsum(A; dims=1)
    @test Array(cumsum(oneArray(A); dims=2)) == cumsum(A; dims=2)
end
