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

@testset "element type checks" begin
  if !float64_supported
    @test_throws "Float64 is not supported" oneArray{Float64}(undef, 1)
    @test_throws "Float64 is not supported" oneArray{ComplexF64}(undef, 1)
  end
  if !float16_supported
    @test_throws "Float16 is not supported" oneArray{Float16}(undef, 1)
  end

  # every array construction performs this check, so it shouldn't query the device
  oneAPI.check_eltype(Float32)
  @test @allocated(oneAPI.check_eltype(Float32)) == 0
end

@testset "adapt" begin
  A = rand(Float32, 3, 3)
  dA = oneArray(A)
  @test Adapt.adapt(Array, dA) == A
  @test Adapt.adapt(oneArray, A) isa oneArray
  @test Array(Adapt.adapt(oneArray, A)) == A
  @test Adapt.adapt(oneArray{Float32}, [1 2;3 4]) isa oneArray{Float32, 2, oneL0.DeviceBuffer}

  # unspecified type parameters are taken from the source array
  @test Adapt.adapt(oneMatrix, [1 2;3 4]) isa oneArray{Int, 2, oneL0.DeviceBuffer}
  @test Adapt.adapt(oneArray{<:Any, <:Any, oneL0.SharedBuffer}, [1 2;3 4]) isa oneArray{Int, 2, oneL0.SharedBuffer}
  @test Adapt.adapt(oneVector{<:Any, oneL0.HostBuffer}, [1, 2]) isa oneArray{Int, 1, oneL0.HostBuffer}
  @test Adapt.adapt(oneArray{Float32, <:Any, oneL0.SharedBuffer}, [1 2;3 4]) isa oneArray{Float32, 2, oneL0.SharedBuffer}
  let xs = Adapt.adapt(oneArray{<:Any, <:Any, oneL0.SharedBuffer}, (Float32[1], Int32[1 2]))
    @test xs[1] isa oneArray{Float32, 1, oneL0.SharedBuffer}
    @test xs[2] isa oneArray{Int32, 2, oneL0.SharedBuffer}
  end
  @test_throws MethodError Adapt.adapt(oneVector{Float32}, [1 2;3 4])
  @test_throws ArgumentError Adapt.adapt(oneArray{<:Real}, ComplexF32[1])
  let xs = oneArray{Float32, 1, oneL0.SharedBuffer}([1])
    @test Adapt.adapt(oneArray, xs) === xs
    @test Adapt.adapt(oneArray{Float32}, xs) === xs
    @test Adapt.adapt(oneArray{Int32}, xs) isa oneArray{Int32, 1, oneL0.SharedBuffer}
    @test Adapt.adapt(typeof(xs), xs) === xs
    @test Adapt.adapt(oneArray{<:Any, <:Any, oneL0.DeviceBuffer}, xs) isa oneArray{Float32, 1, oneL0.DeviceBuffer}
  end
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

# https://github.com/JuliaGPU/oneAPI.jl/issues/652: host accesses wait for outstanding work
# on the memory, also when it was submitted by another task. The kernel takes long enough
# for a missing wait to be observed.
function slow_increment(x, n)
  for _ in 1:n
    x = x * 1.0000001f0 + 1f0
  end
  return x
end
function slow_increment!(a, n)
  i = get_global_id()
  @inbounds a[i] = slow_increment(a[i], n)
  return
end
@testset "host access to $B arrays waits for GPU work" for B in (oneL0.SharedBuffer, oneL0.HostBuffer)
  n = 1_000_000
  launch(a) = @oneapi items=256 groups=cld(length(a), 256) slow_increment!(a, n)
  once = slow_increment(0f0, n)
  twice = slow_increment(once, n)

  # broadcast, then scalar indexing
  a = oneArray{Float32, 1, B}(undef, 1024)
  a .= 42
  @test a[1] == a[end] == 42

  # a kernel submitted by another task
  a = oneArray{Float32, 1, B}(zeros(Float32, 1024))
  wait(@async launch(a))
  @test a[1] ≈ a[end] ≈ once

  # wrapping the memory in an Array
  a = oneArray{Float32, 1, B}(zeros(Float32, 1024))
  launch(a)
  A = unsafe_wrap(Array, a)
  @test A[1] ≈ A[end] ≈ once

  # copying to an Array, which happens on the CPU
  a = oneArray{Float32, 1, B}(zeros(Float32, 1024))
  wait(@async launch(a))
  A = Array(a)
  @test A[1] ≈ A[end] ≈ once

  # work from another task has to wait for this task's work on the same memory
  a = oneArray{Float32, 1, B}(zeros(Float32, 1024))
  launch(a)
  wait(@async launch(a))
  @test a[1] ≈ a[end] ≈ twice

  # a kernel compiled without launching, accessing memory it captures
  let a = oneArray{Float32, 1, B}(zeros(Float32, 1024))
    f = () -> (slow_increment!(a, n); return)
    kernel = @oneapi launch=false f()
    kernel(; items=256, groups=cld(length(a), 256))
    @test a[1] ≈ a[end] ≈ once
  end

  # a kernel submitted to an explicitly created queue
  queue = oneL0.ZeCommandQueue(context(), device(); mode=oneL0.ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS)
  a = oneArray{Float32, 1, B}(zeros(Float32, 1024))
  @oneapi items=256 groups=cld(length(a), 256) queue=queue slow_increment!(a, n)
  @test a[1] ≈ a[end] ≈ once
end

@testset "sub-allocation" begin
  n = 1_000_000
  once = slow_increment(0f0, n)
  base(x) = pointer(oneL0.lookup_alloc(context(), pointer(x)))

  if oneAPI.POOL[]
    # small arrays share driver allocations, which every kernel launch would otherwise make
    # resident one by one
    a = oneArray{Float32}(undef, 16)
    b = oneArray{Float32}(undef, 16)
    @test base(a) == base(b)

    c = oneArray{Float32}(undef, 1 << 20)
    @test base(c) == pointer(c)
  end

  # memory reused on the same stream is only written after earlier work on it
  a = oneArray(zeros(Float32, 1024))
  @oneapi items=256 groups=4 slow_increment!(a, n)
  ptr = pointer(a)
  oneAPI.unsafe_free!(a)
  b = oneArray{Float32}(undef, 1024)
  oneAPI.POOL[] && @test pointer(b) == ptr
  copyto!(b, ones(Float32, 1024))
  @test Array(b) == ones(Float32, 1024)

  # and from another task, only after waiting for it. that task only gets the memory if
  # there are no other chunks of that size, so use them up.
  n_chunk = oneAPI.MAX_CHUNK ÷ sizeof(Float32)
  a = oneArray(zeros(Float32, n_chunk))
  @oneapi items=256 groups=4 slow_increment!(a, n)
  others = []
  if oneAPI.POOL[]
    class = oneAPI.chunk_class(oneAPI.MAX_CHUNK)
    available() = oneAPI.with_pool_lock() do
      pool = oneAPI.pools[(context(), device())]
      !isempty(pool.fresh[class]) || !isempty(pool.free[class])
    end
    while available()
      push!(others, oneArray{Float32}(undef, n_chunk))
    end
  end
  ptr = oneAPI.device_pointer(a)
  oneAPI.unsafe_free!(a)
  stream = oneAPI.global_stream(context(), device())
  inherited, b = fetch(@async begin
    b = oneArray{Float32}(undef, n_chunk)
    # the new array is owned by the stream that used the memory, so that using it here
    # waits for the kernel first (which this GPU might do anyway, executing work in order)
    inherited = b.data[].owner === stream
    copyto!(b, ones(Float32, n_chunk))
    inherited, b
  end)
  if oneAPI.POOL[]
    @test oneAPI.device_pointer(b) == ptr
    @test inherited
  end
  synchronize()
  @test Array(b) == ones(Float32, n_chunk)

  # many small arrays keep their contents, also when unused memory is given back
  xs = [oneArray(fill(Float32(i), i)) for i in 1:300]
  for i in 1:2:300
    oneAPI.unsafe_free!(xs[i])
  end
  if oneAPI.POOL[]
    before = oneAPI._allocated_bytes[]
    ys = [oneArray{Float32}(undef, 1 << 16) for _ in 1:8]   # a slab of their own
    foreach(oneAPI.unsafe_free!, ys)
    oneAPI.trim_pools!()   # (which also frees other empty slabs)
    @test oneAPI._allocated_bytes[] <= before
  end
  @test all(i -> Array(xs[i]) == fill(Float32(i), i), 2:2:300)
  for i in 2:2:300
    xs[i] = oneArray(fill(Float32(i), i))
  end
  @test all(i -> Array(xs[i]) == fill(Float32(i), i), 2:2:300)
end

# run `f(gate)`, with a host event `gate` that is signaled once `f` returns, or else after
# `timeout` seconds. `f` may block its thread until the gate opens, so a timeout signaled
# by a task (or a `Timer`) that has to run on that thread never would: it's signaled from
# an OS thread of its own.
mutable struct GateSignaller
  gate::oneL0.ZeEvent
  deadline::Float64
  @atomic done::Bool
end
function signal_gate(p::Ptr{Cvoid})
  s = unsafe_pointer_to_objref(p)::GateSignaller
  while !(@atomic s.done) && time() < s.deadline
    ccall(:usleep, Cint, (Cuint,), 10_000)
    GC.safepoint()
  end
  oneL0.unchecked_zeEventHostSignal(s.gate)
  return C_NULL
end
function with_gate(f; timeout=10)
  gate = oneL0.ZeEventPool(context(), 1)[1]
  s = GateSignaller(gate, time() + timeout, false)
  thread = Ref{UInt}(0)
  GC.enable(false)  # freeing memory may wait for the gated stream on some stacks
  try
    GC.@preserve s begin
      cb = @cfunction(signal_gate, Ptr{Cvoid}, (Ptr{Cvoid},))
      err = ccall(:pthread_create, Cint, (Ptr{UInt}, Ptr{Cvoid}, Ptr{Cvoid}, Any),
                  thread, C_NULL, cb, s)
      err == 0 || error("pthread_create failed: $(Libc.strerror(err))")
      try
        f(gate)
      finally
        @atomic s.done = true
        oneL0.signal(gate)
        ccall(:pthread_join, Cint, (UInt, Ptr{Cvoid}), thread[], C_NULL)
      end
    end
  finally
    GC.enable(true)
  end
end

# https://github.com/JuliaGPU/oneAPI.jl/issues/719: on integrated devices, small uploads
# copy their source and return without waiting for the GPU, larger ones wait for the copy
@testset "uploads" begin
  stream = oneAPI.global_stream(context(), device())
  staged = oneAPI.stages_uploads(device())

  # with the stream held up on an event, an upload can only finish before it's signaled
  # if it doesn't wait for the stream. on the LTS workaround, every submission would.
  staged && oneL0.sync_each_submission(false) do
    for len in (1, 1024, oneAPI.MAX_STAGED ÷ sizeof(Float32))
      src = rand(Float32, len)
      expected = copy(src)
      a = oneArray{Float32}(undef, len)
      with_gate() do gate
        oneL0.execute!(list -> oneL0.append_wait!(list, gate), stream)
        copyto!(a, src)
        @test !Base.isdone(gate)
        fill!(src, 0)
      end
      @test Array(a) == expected
    end
  end

  # sources changed right after uploading them while the GPU is busy, with more uploads
  # than the ring and its events hold at once
  n = 100_000
  function upload_all(lens)
    busy = oneArray(zeros(Float32, 1024))
    as = oneArray{Float32}[]
    expected = Vector{Float32}[]
    for len in lens
      @oneapi items=256 groups=4 slow_increment!(busy, n)
      src = rand(Float32, len)
      push!(expected, copy(src))
      push!(as, oneArray(src))
      fill!(src, 0)
    end
    map(Array, as) == expected
  end
  @test upload_all(repeat([1, 3, 1000, 100_000], 50))
  @test upload_all(fill(oneAPI.MAX_STAGED ÷ sizeof(Float32), 10))

  # also when not staged
  @test upload_all([oneAPI.MAX_STAGED ÷ sizeof(Float32) + 1])

  # and from several tasks at once, sharing the ring
  @test all(fetch.([Threads.@spawn upload_all(repeat([1, 1000, 100_000], 20)) for _ in 1:4]))

  # a stream held up on an event doesn't hold up staged uploads from other tasks. once
  # their uploads fill the ring, the synchronizing fallback can: the driver may run all
  # streams on one engine, in order. reading them back may wait for copies queued on other
  # streams too, so only do that once the gate is open.
  staged && oneL0.sync_each_submission(false) do
    b = oneArray{Float32}(undef, 1)
    as = with_gate() do gate
      wait(@async begin
        oneL0.execute!(list -> oneL0.append_wait!(list, gate),
                       oneAPI.global_stream(context(), device()))
        copyto!(b, [42f0])
      end)
      as = [oneArray(fill(Float32(i), 1000)) for i in 1:oneAPI.MAX_STAGED_COPIES÷2]
      @test !Base.isdone(gate)
      as
    end
    @test all(i -> Array(as[i]) == fill(Float32(i), 1000), eachindex(as))
    @test Array(b) == [42f0]
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

  # resized buffers are released through the pool, like freshly allocated ones
  c = oneArray{Float32}(undef, 4)
  resize!(c, 0)
  oneAPI.unsafe_free!(c)

  # (with buffers big enough not to be sub-allocated, which `_allocated_bytes` doesn't track)
  GC.enable(false)
  try
    d = oneArray{Float32}(undef, 1 << 19)
    before = oneAPI._allocated_bytes[]
    resize!(d, 1 << 20)
    oneAPI.unsafe_free!(d)
    @test oneAPI._allocated_bytes[] == before - sizeof(Float32) * (1 << 19)
  finally
    GC.enable(true)
  end
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

@testset "branching reduction with loop-carried booleans" begin
    # Before libigc_jll 2.34.4+3, IGC promoted its own replacements, leaving undefined
    # operands in loop-carried i1 values (JuliaGPU/oneAPI.jl#678). This 2-D reduction
    # triggers the bug on Julia 1.12 with --check-bounds=yes.
    A = rand(Int32(1):Int32(4), 64, 64)
    dA = oneArray(A)
    dB = oneArray(A)
    init = (; is_missing=false, is_equal=true)
    mapper(a, b) = (; is_missing=false, is_equal=a == b)
    branching(a, b) = (a.is_missing || b.is_missing) ?
        (; is_missing=true, is_equal=false) :
        (; is_missing=false, is_equal=a.is_equal & b.is_equal)

    @test mapreduce(mapper, branching, dA, dB; init) == init

    B = copy(A)
    B[end] += Int32(1)
    copyto!(dB, B)
    @test mapreduce(mapper, branching, dA, dB; init) ==
        (; is_missing=false, is_equal=false)
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
