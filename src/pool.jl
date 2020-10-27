const allocated = Dict{Tuple{ZeContext,ZePtr},Tuple{oneL0.DeviceBuffer,Int}}()

function allocate(ctx, dev, bytes::Int, alignment::Int)
    # 0-byte allocations shouldn't hit the pool
    bytes == 0 && return ZE_NULL

    buf = device_alloc(ctx, dev, bytes, alignment)
    make_resident(ctx, dev, buf)

    ptr = convert(ZePtr{Nothing}, buf)
    @assert !haskey(allocated, (ctx,ptr))
    allocated[(ctx,ptr)] = buf, 1

    return buf
end

function alias(ctx, dev, ptr)
    # 0-byte allocations shouldn't hit the pool
    ptr == ZE_NULL && return

    buf, refcount = allocated[(ctx,ptr)]
    allocated[(ctx,ptr)] = buf, refcount+1

    return
end

function release(ctx, dev, ptr)
    # 0-byte allocations shouldn't hit the pool
    ptr == ZE_NULL && return

    buf, refcount = allocated[(ctx,ptr)]
    if refcount == 1
        delete!(allocated, (ctx,ptr))
    else
        allocated[(ctx,ptr)] = buf, refcount-1
        return
    end

    evict(ctx, dev, buf)
    free(buf)

    return
end
