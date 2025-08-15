function allocate(::Type{oneL0.DeviceBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.DeviceBuffer(ZE_NULL, bytes, ctx, dev)

    buf = device_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)

    return buf
end

function allocate(::Type{oneL0.SharedBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.SharedBuffer(ZE_NULL, bytes, ctx, dev)

    # TODO: support cross-device shared buffers (by setting `dev=nothing`)

    buf = shared_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)

    return buf
end

function allocate(::Type{oneL0.HostBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.HostBuffer(ZE_NULL, bytes, ctx)
    host_alloc(ctx, bytes, alignment)
end

function release(buf::oneL0.AbstractBuffer)
    sizeof(buf) == 0 && return

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
