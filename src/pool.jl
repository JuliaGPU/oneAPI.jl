export OutOfGPUMemoryError

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory.
"""
struct OutOfGPUMemoryError <: Exception
  sz::Int
  dev::ZeDevice

  function OutOfGPUMemoryError(sz::Integer=0, dev::ZeDevice=device())
    new(sz, dev)
  end
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
      print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    print(" on device $(properties(err.dev).name)")
    if length(memory_properties(err.dev)) == 1
        # XXX: how to handle multiple memories?
        print(" with $(Base.format_bytes(only(memory_properties(err.dev)).totalSize))")
    end
    return io
end

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

    if buf isa oneL0.DeviceBuffer || buf isa oneL0.SharedBuffer
        ctx = oneL0.context(buf)
        dev = oneL0.device(buf)
        evict(ctx, dev, buf)
    end

    free(buf; policy=oneL0.ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE)

    # TODO: queue-ordered free from non-finalizer tasks once we have
    #       `zeMemFreeAsync(ptr, queue)`

    return
end
