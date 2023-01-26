# pool

export ZeEventPool

mutable struct ZeEventPool
    handle::ze_event_pool_handle_t

    context::ZeContext

    function ZeEventPool(ctx::ZeContext, count::Integer, devs::ZeDevice...;
                         flags=0)
        desc_ref = Ref(ze_event_pool_desc_t(; flags, count))
        handle_ref = Ref{ze_event_pool_handle_t}()
        zeEventPoolCreate(ctx, desc_ref, length(devs), isempty(devs) ? C_NULL : [devs...], handle_ref)
        obj = new(handle_ref[], ctx)
        finalizer(obj) do obj
            zeEventPoolDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_event_pool_handle_t}, pool::ZeEventPool) = pool.handle

Base.:(==)(a::ZeEventPool, b::ZeEventPool) = a.handle == b.handle
Base.hash(e::ZeEventPool, h::UInt) = hash(e.handle, h)

Base.getindex(pool::ZeEventPool, i::Integer) = ZeEvent(pool, i)


# event

export ZeEvent, append_wait!, signal, append_signal!, append_reset!, kernel_timestamp

mutable struct ZeEvent
    handle::ze_event_handle_t
    pool::ZeEventPool

    function ZeEvent(pool, index::Integer)
        desc_ref = Ref(ze_event_desc_t(; index=index-1))
        handle_ref = Ref{ze_event_handle_t}()
        zeEventCreate(pool, desc_ref, handle_ref)
        obj = new(handle_ref[], pool)
        finalizer(obj) do obj
            zeEventDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_event_handle_t}, event::ZeEvent) = event.handle

Base.:(==)(a::ZeEvent, b::ZeEvent) = a.handle == b.handle
Base.hash(e::ZeEvent, h::UInt) = hash(e.handle, h)

signal(event::ZeEvent) = zeEventHostSignal(event)
append_signal!(list::ZeCommandList, event::ZeEvent) = zeCommandListAppendSignalEvent(list, event)

Base.wait(event::ZeEvent, timeout::Number=typemax(UInt64)) =
    zeEventHostSynchronize(event, timeout)
append_wait!(list::ZeCommandList, events::ZeEvent...) =
    zeCommandListAppendWaitOnEvents(list, length(events), [events...])

Base.reset(event::ZeEvent) = zeEventHostReset(event)
append_reset!(list::ZeCommandList, event::ZeEvent) = zeCommandListAppendEventReset(list, event)

function Base.isdone(event::ZeEvent)
    res = unsafe_zeEventQueryStatus(event)
    if res == RESULT_NOT_READY
        return false
    elseif res == RESULT_SUCCESS
        return true
    else
        throw_api_error(res)
    end
end

function kernel_timestamp(event)
    timestamp_ref = Ref{ze_kernel_timestamp_result_t}()
    zeEventQueryKernelTimestamp(event, timestamp_ref)

    # TODO: convert using ze_device_properties_t.timerResolution
    # TODO: mask by ze_device_properties_t.kernelTimestampValidBits
    # https://spec.oneapi.com/level-zero/latest/core/PROG.html#kernel-timestamp-events
    # but how to get the device?

    timestamp = timestamp_ref[]
    return (;
        :global => (
            start = timestamp._global.kernelStart == -1%UInt32 ? nothing : Int(timestamp._global.kernelStart),
            stop  = timestamp._global.kernelEnd == -1%UInt32 ?   nothing : Int(timestamp._global.kernelEnd)
        ),
        :context => (
            start = timestamp.context.kernelStart == -1%UInt32 ? nothing : Int(timestamp.context.kernelStart),
            stop  = timestamp.context.kernelEnd == -1%UInt32 ?   nothing : Int(timestamp.context.kernelEnd)
        )
    )
end
