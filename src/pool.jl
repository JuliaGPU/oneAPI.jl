# Track total allocated GPU memory (device + shared buffers) for proactive GC.
# This mirrors AMDGPU.jl's approach: trigger GC before OOM so that finalizers
# can free stale GPU buffers that Julia's GC hasn't collected yet (Julia's GC
# only sees CPU memory pressure, not GPU memory pressure).
const _allocated_bytes = Threads.Atomic{Int64}(0)
const _total_mem_cache = Threads.Atomic{Int64}(0)

function _get_total_mem(dev)
    cached = _total_mem_cache[]
    cached > 0 && return cached
    total = only(oneL0.memory_properties(dev)).totalSize
    Threads.atomic_cas!(_total_mem_cache, Int64(0), Int64(total))
    return _total_mem_cache[]
end

# Per-process memory budget for the proactive-GC thresholds below. `_allocated_bytes`
# only tracks this process, and GC in this process cannot free buffers held by another,
# so when several processes share one device (e.g. parallel test workers) each must
# budget against its share of the card, not the whole card — otherwise their combined
# usage exceeds physical memory before any of them feels pressure. ONEAPI_MEMORY_LIMIT
# accepts a byte count ("4000000000") or a percentage of device memory ("50%"); unset
# means the full device.
const _memory_limit_cache = Threads.Atomic{Int64}(-1)

function _memory_limit(dev)
    cached = _memory_limit_cache[]
    cached >= 0 && return cached
    total = _get_total_mem(dev)
    str = get(ENV, "ONEAPI_MEMORY_LIMIT", "")
    limit = if isempty(str)
        total
    elseif endswith(str, "%")
        pct = parse(Int, chop(str))
        0 < pct <= 100 || error("ONEAPI_MEMORY_LIMIT percentage must be in (0, 100]")
        total * pct ÷ 100
    else
        parse(Int64, str)
    end
    Threads.atomic_cas!(_memory_limit_cache, Int64(-1), Int64(limit))
    return _memory_limit_cache[]
end

function _maybe_gc(dev, bytes)
    allocated = _allocated_bytes[]
    allocated <= 0 && return
    total_mem = _memory_limit(dev)
    allocated + bytes > total_mem * 0.4 && release_pending!()
    if allocated + bytes > total_mem * 0.8
        # Flush deferred resource releases (e.g., MKL sparse handles) from previous GC
        # cycles first — these are safe to release now because they were deferred earlier.
        # Do this BEFORE GC to avoid racing with new finalizers.
        oneL0._run_reclaim_callbacks()
        # Full GC to collect old-generation objects whose finalizers free GPU memory.
        GC.gc(true)
    elseif allocated + bytes > total_mem * 0.4
        GC.gc(false)
    else
        return
    end
    # the memory of collected arrays only becomes available once their slabs are empty
    trim_pools!()
    # and the driver may only have deferred freeing it (see `release`)
    oneL0.sweep_deferred_frees()
    return
end

# GPUCompiler implements 8- and 16-bit atomics with 32-bit atomics on the containing aligned
# word, so every buffer has to extend to a multiple of 4 bytes and be aligned accordingly.
# The array's dimensions stay as requested; only the buffer gets bigger.
function atomic_allocation(bytes::Int, alignment::Int)
    padded = bytes + (-bytes & 3)
    padded < bytes && throw(ArgumentError("allocation of $bytes bytes is too large"))
    return padded, max(alignment, 4)
end

function allocate(::Type{oneL0.DeviceBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.DeviceBuffer(ZE_NULL, bytes, ctx, dev)
    bytes, alignment = atomic_allocation(bytes, alignment)

    _maybe_gc(dev, bytes)
    buf = device_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)
    Threads.atomic_add!(_allocated_bytes, Int64(bytes))

    return buf
end

function allocate(::Type{oneL0.SharedBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.SharedBuffer(ZE_NULL, bytes, ctx, dev)
    bytes, alignment = atomic_allocation(bytes, alignment)

    # TODO: support cross-device shared buffers (by setting `dev=nothing`)

    _maybe_gc(dev, bytes)
    buf = shared_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)
    Threads.atomic_add!(_allocated_bytes, Int64(bytes))

    return buf
end

function allocate(::Type{oneL0.HostBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.HostBuffer(ZE_NULL, bytes, ctx)
    bytes, alignment = atomic_allocation(bytes, alignment)
    buf = host_alloc(ctx, bytes, alignment)
    # Host USM must be made resident on the device, exactly like the device and shared
    # allocations above. A GPU kernel that reads a non-resident host buffer can take a
    # NotPresent pagefault under GC/allocation churn, even though host USM is nominally
    # device-accessible.
    make_resident(ctx, dev, buf)
    return buf
end

# Allocate the memory backing an array, together with its ownership state, and release it.
# Arrays only allocate and free memory through these.
function alloc_managed(::Type{B}, ctx, dev, bytes::Int, alignment::Int) where {B}
    if B === oneL0.DeviceBuffer && POOL[] && 0 < bytes && max(bytes, alignment) <= MAX_CHUNK
        managed = pool_alloc(ctx, dev, bytes, alignment)
        managed === nothing || return managed
    end
    return Managed(allocate(B, ctx, dev, bytes, alignment))
end

function release(managed::Managed)
    mem = managed.mem
    if mem isa oneL0.DeviceBuffer && 0 < sizeof(mem) <= MAX_CHUNK
        pool_free(managed) && return
    end
    release(mem)
end


## sub-allocation

# Small device allocations are carved out of bigger ones, slabs. NEO backs every allocation
# with a buffer object of at least 64 KiB, and because kernels declare indirect access (see
# `link_kernel`), every live one is made resident for each submission, which on i915 costs
# ~0.45 µs per allocation and kernel launch. NEO's own pool doing this is disabled on Gen12LP
# and PVC.
#
# A freed chunk keeps the ownership state of the memory that used it, and is handed out
# again with that state: work on the same stream is ordered after the previous use anyway,
# and anything else waits for it first (see `take_ownership!`). Reuse thus never waits for
# the GPU, unlike freeing memory.

# Opt-out knob, initialized from ONEAPI_MEMORY_POOL in `__init__`.
const POOL = Ref{Bool}(true)

const SLAB_SIZE = 2^21
const MIN_CHUNK = 256
const MAX_CHUNK = 2^20

mutable struct Slab
    const buf::oneL0.DeviceBuffer
    const chunk::Int
    # chunks handed out, and the first one never handed out
    used::Int
    next::Int
end

struct FreeChunk
    slab::Slab
    ptr::ZePtr{Cvoid}
    owner::Union{Nothing, oneStream, ZeCommandQueue}
    dirty::Bool
end

struct DevicePool
    # per chunk size: slabs with chunks never handed out, and freed chunks
    fresh::Vector{Vector{Slab}}
    free::Vector{Vector{FreeChunk}}
    # all slabs, by address
    slabs::Dict{UInt, Slab}
end
const NCLASSES = trailing_zeros(MAX_CHUNK) - trailing_zeros(MIN_CHUNK) + 1
DevicePool() = DevicePool([Slab[] for _ in 1:NCLASSES],
                          [FreeChunk[] for _ in 1:NCLASSES], Dict{UInt, Slab}())

const pools = Dict{Tuple{ZeContext, ZeDevice}, DevicePool}()

# Protects the pools. Never held across anything that can yield, and with finalizers
# disabled, as chunks are freed from finalizers.
const pool_lock = Threads.SpinLock()

function with_pool_lock(f)
    GC.enable_finalizers(false)
    lock(pool_lock)
    try
        f()
    finally
        unlock(pool_lock)
        GC.enable_finalizers(true)
    end
end

chunk_class(chunk::Int) = trailing_zeros(chunk) - trailing_zeros(MIN_CHUNK) + 1

# how many freed chunks to consider, from the most recently freed one, looking for one that
# can be reused without waiting for another stream
const REUSE_WINDOW = 8

function take_chunk!(pool::DevicePool, class::Int, owner)
    free = pool.free[class]
    # a chunk last used by `owner`, or whose memory has no outstanding work
    for i in length(free):-1:max(1, length(free) - REUSE_WINDOW + 1)
        c = free[i]
        if c.owner === owner || !c.dirty
            deleteat!(free, i)
            c.slab.used += 1
            return c
        end
    end
    # a fresh chunk
    fresh = pool.fresh[class]
    if !isempty(fresh)
        slab = last(fresh)
        ptr = pointer(slab.buf) + slab.next * slab.chunk
        slab.next += 1
        slab.next * slab.chunk == SLAB_SIZE && pop!(fresh)
        slab.used += 1
        return FreeChunk(slab, ptr, nothing, false)
    end
    # any freed chunk
    isempty(free) && return nothing
    c = pop!(free)
    c.slab.used += 1
    return c
end

function pool_alloc(ctx, dev, bytes::Int, alignment::Int)
    bytes, alignment = atomic_allocation(bytes, alignment)
    chunk = max(MIN_CHUNK, nextpow(2, max(bytes, alignment)))
    class = chunk_class(chunk)
    owner = global_stream(ctx, dev)

    c = with_pool_lock() do
        pool = get!(DevicePool, pools, (ctx, dev))
        take_chunk!(pool, class, owner)
    end
    if c === nothing
        # `allocate` may collect garbage, freeing chunks, but those are only reused when
        # this slab runs out
        buf = allocate(oneL0.DeviceBuffer, ctx, dev, SLAB_SIZE, SLAB_SIZE)
        if UInt(pointer(buf)) % SLAB_SIZE != 0
            # chunks are mapped to their slab by address
            release(buf)
            return nothing
        end
        slab = Slab(buf, chunk, 1, 1)
        c = FreeChunk(slab, pointer(buf), nothing, false)
        with_pool_lock() do
            pool = pools[(ctx, dev)]
            pool.slabs[UInt(pointer(buf))] = slab
            push!(pool.fresh[class], slab)
        end
    end
    return Managed(oneL0.DeviceBuffer(c.ptr, bytes, ctx, dev), c.owner, c.dirty)
end

# Return a chunk to its pool, unless the memory wasn't sub-allocated.
function pool_free(managed::Managed{oneL0.DeviceBuffer})
    mem = managed.mem
    ptr = pointer(mem)
    owner = managed.owner
    dirty = Base.@atomic managed.dirty
    with_pool_lock() do
        pool = get(pools, (oneL0.context(mem), oneL0.device(mem)), nothing)
        pool === nothing && return false
        slab = get(pool.slabs, UInt(ptr) & ~UInt(SLAB_SIZE - 1), nothing)
        slab === nothing && return false
        slab.used -= 1
        push!(pool.free[chunk_class(slab.chunk)], FreeChunk(slab, ptr, owner, dirty))
        return true
    end
end

# Free slabs none of whose chunks are in use. Also run when the driver runs out of memory.
function trim_pools!()
    empty = with_pool_lock() do
        empty = Slab[]
        for pool in values(pools)
            for (base, slab) in collect(pool.slabs)
                slab.used == 0 || continue
                push!(empty, slab)
                delete!(pool.slabs, base)
                class = chunk_class(slab.chunk)
                filter!(s -> s !== slab, pool.fresh[class])
                filter!(c -> c.slab !== slab, pool.free[class])
            end
        end
        empty
    end
    # on the LTS stack, this waits for outstanding work on the memory (see `release`)
    foreach(slab -> release(slab.buf), empty)
    return
end
oneL0.register_reclaim_callback!(trim_pools!)

function release(buf::oneL0.AbstractBuffer)
    sizeof(buf) == 0 && return

    if buf isa oneL0.DeviceBuffer || buf isa oneL0.SharedBuffer
        Threads.atomic_sub!(_allocated_bytes, Int64(sizeof(buf)))
    end

    # XXX: is it necessary to evice memory if we are going to free it?
    #      this is racy, because eviction is not queue-ordered, and
    #      we don't want to synchronize inside what could have been a
    #      GC-driven finalizer. if we need to, port the stream/queue
    #      tracking from CUDA.jl so that we can synchronize only the
    #      queue that's associated with the buffer.
    #if buf isa oneL0.DeviceBuffer || buf isa oneL0.SharedBuffer
    #    ctx = oneL0.context(buf)
    #    dev = oneL0.device(buf)
    #    evict(ctx, dev, buf)
    #end

    if oneL0.LTS[]
        # NEO (at least the 25.18 LTS release) does not honor the BLOCKING_FREE/DEFER_FREE
        # policies of zeMemFreeExt: it advertises ZE_extension_memory_free_policies but
        # unmaps the allocation immediately, even with work in flight that references it.
        # That turns a GC-driven free of a dead array whose last kernel/copy hasn't retired
        # into a GPU pagefault, which gets the kernel context banned and makes every later
        # submission fail with ZE_RESULT_ERROR_UNKNOWN. Synchronize the queues that could
        # reference this buffer before freeing.
        synchronize_all_streams(oneL0.context(buf), oneL0.device(buf))
        free(buf; policy=oneL0.ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE)
    else
        # with kernels accessing memory indirectly, a blocking free waits for all work
        # submitted so far, which stalls the thread and deadlocks when that work waits for
        # a task on it (this is usually called from a finalizer). have the driver free the
        # memory once that work has completed instead. NEO doesn't report the free policies
        # it supports, so this relies on the rolling stack implementing DEFER_FREE.
        free(buf; policy=oneL0.ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE)
    end

    return
end


## upload staging

# A copy reads its source when it executes, not when it is appended, and NEO only stages
# pageable memory on some devices, and not reliably. An upload from host memory that may
# change as soon as it returns therefore either waits for the copy, or first copies its
# source to host USM. Uploads of up to `MAX_STAGED` bytes do the latter, through a ring
# buffer shared by all streams: each one occupies the bytes after the previous one, until
# the event its copy signals is done. The ring is freed up from its oldest upload on. An
# upload that finds it full, or all its events in use, synchronizes instead: the oldest
# upload may be another task's, held up by work this task shouldn't wait for.
#
# The ring isn't per stream, as every task has its own: a task doing a single upload would
# have to allocate and make resident a ring of its own, which stays around until the task
# is collected.
#
# Only integrated devices stage uploads. On discrete ones, NEO copies uploads of up to a few
# MB from pageable memory (and of up to 50 KB from host USM) on the CPU, through a mapping of
# device memory, after waiting for the queue itself. Staging can't avoid that wait for small
# uploads, and is slower than the CPU copy when the queue is idle.

const STAGING_SIZE = 2^21
const MAX_STAGED = 2^20
const MAX_STAGED_COPIES = 64

struct StagedCopy
    event::oneL0.ZeEvent
    start::Int
    stop::Int
end

struct Staging
    buf::oneL0.HostBuffer
    free::Vector{oneL0.ZeEvent}
    pending::Vector{StagedCopy}     # in submission order
end

function Staging(ctx::ZeContext, dev::ZeDevice)
    buf = allocate(oneL0.HostBuffer, ctx, dev, STAGING_SIZE, 64)
    pool = oneL0.ZeEventPool(ctx, MAX_STAGED_COPIES, dev;
                             flags = oneL0.ZE_EVENT_POOL_FLAG_HOST_VISIBLE)
    events = [oneL0.ZeEvent(pool, i; signal = oneL0.ZE_EVENT_SCOPE_FLAG_HOST,
                            wait = oneL0.ZE_EVENT_SCOPE_FLAG_HOST)
              for i in 1:MAX_STAGED_COPIES]
    return Staging(buf, events, StagedCopy[])
end

const stagings = Dict{Tuple{ZeContext, ZeDevice}, Staging}()
const staging_lock = ReentrantLock()
const staged_devices = Dict{ZeDevice, Bool}()

stages_uploads(dev::ZeDevice) =
    @lock staging_lock get!(() -> is_integrated(dev), staged_devices, dev)

function reclaim!(st::Staging)
    while !isempty(st.pending) && Base.isdone(first(st.pending).event)
        c = popfirst!(st.pending)
        reset(c.event)
        push!(st.free, c.event)
    end
    return
end

# where in the ring `bytes` fit, if they do
function ring_offset(st::Staging, bytes::Int)
    isempty(st.pending) && return 0
    head = last(st.pending).stop
    tail = first(st.pending).start
    if head > tail
        # in use from `tail` to `head`: fit after it, or wrap around
        head + bytes <= STAGING_SIZE && return head
        bytes <= tail && return 0
    else
        # in use from `tail` to the end, and from the start to `head`
        head + bytes <= tail && return head
    end
    return nothing
end

# copy `bytes` from `src` to `dst`, appending the copy to `list`, the immediate list of a
# stream on `dev`, through the ring so that the source can change as soon as this returns.
# returns `false` if the ring has no room.
function stage_upload!(list, ctx::ZeContext, dev::ZeDevice, dst::ZePtr, src::Ptr,
                       bytes::Int)
    padded = cld(bytes, 64) * 64
    @lock staging_lock begin
        st = get!(() -> Staging(ctx, dev), stagings, (ctx, dev))
        reclaim!(st)
        offset = ring_offset(st, padded)
        (offset === nothing || isempty(st.free)) && return false
        ptr = pointer(st.buf) + offset
        unsafe_copyto!(convert(Ptr{UInt8}, ptr), convert(Ptr{UInt8}, src), bytes)
        # if this fails, the copy may still execute and signal the event, so drop it
        event = pop!(st.free)
        oneL0.append_copy!(list, dst, ptr, bytes, event)
        push!(st.pending, StagedCopy(event, offset, offset + padded))
        return true
    end
end
