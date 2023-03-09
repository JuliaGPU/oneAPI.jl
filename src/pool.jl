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

function release(buf::oneL0.AbstractBuffer)
    sizeof(buf) == 0 && return

    ctx = oneL0.context(buf)
    dev = oneL0.device(buf)

    evict(ctx, dev, buf)
    free(buf; policy=oneL0.ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE)

    # TODO: queue-ordered free from non-finalizer tasks once we have
    #       `zeMemFreeAsync(ptr, queue)`

    return
end
