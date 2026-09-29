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

# arrays that occupy whole pages, so that wrapping them cannot be rejected because their
# pages partially overlap memory that is already wrapped
function page_aligned_array(T, n)
  pagesize = ccall(:getpagesize, Cint, ())
  ref = Ref{Ptr{Cvoid}}()
  bytes = cld(n * sizeof(T), pagesize) * pagesize
  @assert ccall(:posix_memalign, Cint, (Ptr{Ptr{Cvoid}}, Csize_t, Csize_t), ref, pagesize, bytes) == 0
  return unsafe_wrap(Array, Ptr{T}(ref[]), n; own=true)
end

# wrap an array and memory inside of it, free the outer wrapper, and return the inner one
@noinline function wrap_inner()
  big = page_aligned_array(Float32, 4096)
  outer = unsafe_wrap(oneArray, big)
  inner = GC.@preserve big unsafe_wrap(oneArray, pointer(big) + 4096, 16)
  oneAPI.unsafe_free!(outer)
  return inner, WeakRef(big)
end

# create wrappers that are unreachable once this returns
@noinline function wrap_garbage(n)
  for _ in 1:n
    unsafe_wrap(oneArray, page_aligned_array(Float32, 4096)) .+= 1
  end
end

function wrap_kernel(a)
  i = get_global_id()
  @inbounds a[i] = i
  return
end

@testset "wrapping host memory" begin
  if !oneAPI.system_memmap_supported(oneAPI.driver())
    @test_throws ArgumentError unsafe_wrap(oneArray, Float32[1])
  else
    a = Float32[1, 2, 3, 4]
    b = unsafe_wrap(oneArray, a)
    @test b isa oneVector{Float32, oneL0.HostBuffer}
    @test size(b) == size(a)
    @test UInt(pointer(b)) == UInt(pointer(a))

    # changes are visible in both directions
    b .+= 1
    synchronize()
    @test a == [2, 3, 4, 5]
    a[1] = 10
    c = oneArray{Float32, 1, oneL0.DeviceBuffer}(undef, 4)
    c .= b    # read on the device, not via the host
    @test Array(c) == [10, 3, 4, 5]
    @test pointer(unsafe_wrap(Array, b)) == pointer(a)

    for AT in [oneArray, oneArray{Float32}, oneArray{Float32, 1},
               oneArray{Float32, 1, oneL0.HostBuffer}],
        f in [x -> unsafe_wrap(AT, pointer(x), length(x)),
              x -> unsafe_wrap(AT, pointer(x), size(x)),
              x -> unsafe_wrap(AT, x)]
      d = f(a)
      @test d isa oneVector{Float32, oneL0.HostBuffer}
      @test Array(d) == a
    end
    let m = rand(Float32, 3, 4)
      d = unsafe_wrap(oneArray, m)
      @test d isa oneMatrix{Float32}
      d .*= 2
      synchronize()
      @test Array(d) == m
    end
    @test isempty(Array(unsafe_wrap(oneArray, Float32[])))

    # large arrays, and copies to and from regular device arrays
    let x = copyto!(page_aligned_array(Float32, 10^6), rand(Float32, 10^6))
      ref = x .* 2
      d = unsafe_wrap(oneArray, x)
      d .*= 2
      synchronize()
      @test x == ref
      e = oneArray(ref .+ 1)
      copyto!(d, e)
      synchronize()
      @test x == ref .+ 1
    end

    fill!(b, 42)
    synchronize()
    @test all(==(42), a)
    @oneapi items=length(b) wrap_kernel(b)
    synchronize()
    @test a == [1, 2, 3, 4]

    # memory within pages that are already mapped can be wrapped too
    nmappings = length(oneAPI.system_mappings)
    let big = page_aligned_array(Float32, 4096)
      wbig = unsafe_wrap(oneArray, big)
      GC.@preserve big begin
        inner = unsafe_wrap(oneArray, pointer(big) + 4096, 16)
        wbig .= 1
        inner .= 2
        synchronize()
        @test big[1] == 1 && big[1025] == 2
        # partially overlapping memory cannot be wrapped
        @test_throws ArgumentError unsafe_wrap(oneArray, pointer(big) + 4 * 4096 - 16, 1024)
        oneAPI.unsafe_free!(inner)
      end
      oneAPI.unsafe_free!(wbig)
    end
    # small arrays share pages, which are mapped once
    # a mapping keeps the memory it covers alive, even after the wrapper that created it
    # is freed while other wrappers still use the mapping
    inner, big_ref = wrap_inner()
    for _ in 1:3
      GC.gc(true)
    end
    @test big_ref.value !== nothing
    inner .= 5
    synchronize()
    @test Array(inner) == fill(5, 16)
    oneAPI.unsafe_free!(inner)
    @test length(oneAPI.system_mappings) == nmappings

    xs = [Float32[i, i] for i in 1:8]
    ws = [unsafe_wrap(oneArray, x) for x in xs]
    push!(ws, unsafe_wrap(oneArray, xs[1]))   # wrapping the same memory twice
    for w in ws
      w .+= 1
    end
    synchronize()
    @test xs[1] == [3, 3]
    @test xs[2] == [3, 3]
    foreach(oneAPI.unsafe_free!, ws)
    @test length(oneAPI.system_mappings) == nmappings

    # wrappers freed by the GC are released asynchronously
    wrap_garbage(10)
    synchronize()
    for _ in 1:100
      GC.gc(true)
      length(oneAPI.system_mappings) == nmappings && break
      sleep(0.1)
    end
    @test length(oneAPI.system_mappings) == nmappings

    # the wrapper keeps the array alive
    d = unsafe_wrap(oneArray, fill!(page_aligned_array(Float32, 1024), 1))
    GC.gc(true)
    @test sum(d) == 1024

    @test_throws ArgumentError resize!(b, 5)
    @test_throws ArgumentError unsafe_wrap(oneVector{Float32, oneL0.DeviceBuffer}, a)
    @test_throws ArgumentError unsafe_wrap(oneArray, Ptr{Float32}(C_NULL), 1)
    @test_throws ArgumentError unsafe_wrap(oneArray, pointer(a), (-1,))
    @test_throws ArgumentError unsafe_wrap(oneArray, Ptr{Float32}(typemax(UInt) - 15), 16)
    GC.@preserve a begin
      @test_throws ArgumentError unsafe_wrap(oneArray, Ptr{Float32}(pointer(a) + 1), 1)
    end
  end
end

@testset "reductions of host-accessible arrays" begin
  for B in (oneL0.SharedBuffer, oneL0.HostBuffer)
    a = oneArray{Float32, 1, B}(fill(1.0f0, 1024))
    @test sum(a) == 1024
    @test maximum(a) == 1
  end
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

@testset "strided mixed reductions" begin
    # The Aurora LTS IGC miscompiles a reduction kernel's global reads when the *innermost*
    # reduced axis is strided (dim 1 kept, e.g. `dims=2`); mapreducedim! routes those to a
    # coalesced kernel. Reductions that also reduce dim 1 (e.g. `dims=(1,3)`) keep a contiguous
    # innermost axis and stay correct on the workgroup-per-slice kernel — including with a small
    # leading dim, where the contiguous run is short. Use Int32 (exact, associative) so the
    # check is immune to Float32 accumulation-order rounding.
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

@testset "mapreducedim! returning same type" begin
  R = transpose(oneAPI.zeros(Float32, 2, 3))
  A = oneArray(rand(Float32, 3, 2, 10))
  @test @inferred(oneAPI.GPUArrays.mapreducedim!(identity, +, R, A)) === R
end
