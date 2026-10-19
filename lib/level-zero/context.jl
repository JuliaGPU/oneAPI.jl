export ZeContext, status

mutable struct ZeContext
    handle::ze_context_handle_t

    driver::ZeDriver

    # the context object, and every object and allocation created in the context, holds a
    # reference to the handle, and the last one to go destroys it: Julia finalizes objects
    # that become unreachable together in no particular order, but memory allocated in a
    # context can't be freed anymore once the context is destroyed.
    @atomic refs::Int

    # the allocations holding a reference, as memory allocated otherwise can be freed too.
    # protected by `allocations_lock`, see `with_allocations`.
    const allocations::Set{Ptr{Cvoid}}
    const allocations_lock::Threads.SpinLock

    function ZeContext(drv::ZeDriver)
        desc_ref = Ref(ze_context_desc_t())
        handle_ref = Ref{ze_context_handle_t}()
        zeContextCreate(drv, desc_ref, handle_ref)
        obj = new(handle_ref[], drv, 1, Set{Ptr{Cvoid}}(), Threads.SpinLock())
        finalizer(release, obj)
        obj
    end
end

retain(ctx::ZeContext) = (@atomic ctx.refs += 1; return)

function release(ctx::ZeContext)
    if (@atomic ctx.refs -= 1) == 0
        zeContextDestroy(ctx)
    end
    return
end

# with finalizers disabled, as memory is freed from finalizers
function with_allocations(f, ctx::ZeContext)
    GC.enable_finalizers(false)
    lock(ctx.allocations_lock)
    try
        f(ctx.allocations)
    finally
        unlock(ctx.allocations_lock)
        GC.enable_finalizers(true)
    end
end

Base.unsafe_convert(::Type{ze_context_handle_t}, dev::ZeContext) = dev.handle

Base.:(==)(a::ZeContext, b::ZeContext) = a.handle == b.handle
Base.hash(e::ZeContext, h::UInt) = hash(e.handle, h)

status(ctx::ZeContext) = zeContextGetStatus(ctx)
