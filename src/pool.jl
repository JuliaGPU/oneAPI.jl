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

function _maybe_gc(dev, bytes)
    allocated = _allocated_bytes[]
    allocated <= 0 && return
    total_mem = _get_total_mem(dev)
    if allocated + bytes > total_mem * 0.8
        # Full GC to collect old-generation objects whose finalizers free GPU memory.
        # GC.gc(false) only does minor collection which won't reclaim promoted objects.
        GC.gc(true)
    elseif allocated + bytes > total_mem * 0.4
        GC.gc(false)
    end
end

function allocate(::Type{oneL0.DeviceBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.DeviceBuffer(ZE_NULL, bytes, ctx, dev)

    _maybe_gc(dev, bytes)
    buf = device_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)
    Threads.atomic_add!(_allocated_bytes, Int64(bytes))

    return buf
end

function allocate(::Type{oneL0.SharedBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.SharedBuffer(ZE_NULL, bytes, ctx, dev)

    # TODO: support cross-device shared buffers (by setting `dev=nothing`)

    _maybe_gc(dev, bytes)
    buf = shared_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)
    Threads.atomic_add!(_allocated_bytes, Int64(bytes))

    return buf
end

function allocate(::Type{oneL0.HostBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.HostBuffer(ZE_NULL, bytes, ctx)
    host_alloc(ctx, bytes, alignment)
end

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

    free(buf; policy=oneL0.ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE)

    # TODO: queue-ordered free from non-finalizer tasks once we have
    #       `zeMemFreeAsync(ptr, queue)`

    return
end
