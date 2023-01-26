export ZeContext, status

mutable struct ZeContext
    handle::ze_context_handle_t

    driver::ZeDriver

    function ZeContext(drv::ZeDriver)
        desc_ref = Ref(ze_context_desc_t())
        handle_ref = Ref{ze_context_handle_t}()
        zeContextCreate(drv, desc_ref, handle_ref)
        obj = new(handle_ref[], drv)
        finalizer(obj) do obj
            zeContextDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_context_handle_t}, dev::ZeContext) = dev.handle

Base.:(==)(a::ZeContext, b::ZeContext) = a.handle == b.handle
Base.hash(e::ZeContext, h::UInt) = hash(e.handle, h)

status(ctx::ZeContext) = zeContextGetStatus(ctx)
