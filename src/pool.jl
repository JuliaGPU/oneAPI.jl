function allocate(::Type{oneL0.DeviceBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return oneL0.DeviceBuffer(ZE_NULL, bytes, ctx, dev)

    buf = device_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)

    return buf
end

function release(buf::oneL0.AbstractBuffer)
    sizeof(buf) == 0 && return

    ctx = oneL0.context(buf)
    dev = oneL0.device(buf)

    evict(ctx, dev, buf)
    free(buf)

    return
end
