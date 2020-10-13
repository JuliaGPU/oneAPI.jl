export make_resident, evict


## memory

function make_resident(ctx::ZeContext, dev::ZeDevice, buf::AbstractBuffer, size=sizeof(buf))
    if pointer(buf) != ZE_NULL
        zeContextMakeMemoryResident(ctx, dev, buf, size)
    end
end

function evict(ctx::ZeContext, dev::ZeDevice, buf::AbstractBuffer, size=sizeof(buf))
    if pointer(buf) != ZE_NULL
        zeContextEvictMemory(ctx, dev, buf, size)
    end
end
