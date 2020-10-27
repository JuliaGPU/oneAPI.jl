export make_resident, evict


## memory

function make_resident(ctx::ZeContext, dev::ZeDevice, buf::AbstractBuffer, size=sizeof(buf))
    zeContextMakeMemoryResident(ctx, dev, buf, size)
end

function evict(ctx::ZeContext, dev::ZeDevice, buf::AbstractBuffer, size=sizeof(buf))
    zeContextEvictMemory(ctx, dev, buf, size)
end
