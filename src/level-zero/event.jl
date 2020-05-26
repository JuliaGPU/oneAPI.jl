# pool

export ZeEventPool

mutable struct ZeEventPool
    handle::ze_event_pool_handle_t

    # https://github.com/intel/compute-runtime/issues/294
    ZeEventPool(drv::ZeDriver, size::Integer; kwargs...) =
        ZeEventPool(drv, size, devices(drv)...; kwargs...)

    function ZeEventPool(drv::ZeDriver, size::Integer, devs::ZeDevice...;
                         flags=ZE_EVENT_POOL_FLAG_DEFAULT)
        desc_ref = Ref(ze_event_pool_desc_t(
            ZE_EVENT_POOL_DESC_VERSION_CURRENT,
            flags,
            size
        ))
        handle_ref = Ref{ze_event_pool_handle_t}()
        zeEventPoolCreate(drv, desc_ref, length(devs), [devs...], handle_ref)
        obj = new(handle_ref[])
        finalizer(obj) do obj
            zeEventPoolDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_event_pool_handle_t}, pool::ZeEventPool) = pool.handle

Base.getindex(pool::ZeEventPool, i::Integer) = ZeEvent(pool, i)


# event

export ZeEvent, append_wait!, signal, append_signal!, append_reset!, query,
       global_time, context_time

mutable struct ZeEvent
    handle::ze_event_handle_t
    pool::ZeEventPool

    function ZeEvent(pool, index::Integer)
        desc_ref = Ref(ze_event_desc_t(
            ZE_EVENT_DESC_VERSION_CURRENT,
            index-1,
            ZE_EVENT_SCOPE_FLAG_NONE,
            ZE_EVENT_SCOPE_FLAG_HOST
        ))
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

signal(event::ZeEvent) = zeEventHostSignal(event)
append_signal!(list::ZeCommandList, event::ZeEvent) = zeCommandListAppendSignalEvent(list, event)

Base.wait(event::ZeEvent, timeout::Number=typemax(UInt32)) =
    zeEventHostSynchronize(event, timeout)
append_wait!(list::ZeCommandList, events::ZeEvent...) =
    zeCommandListAppendWaitOnEvents(list, length(events), [events...])

Base.reset(event::ZeEvent) = zeEventHostReset(event)
append_reset!(list::ZeCommandList, event::ZeEvent) = zeCommandListAppendEventReset(list, event)

function query(event::ZeEvent)
    res = unsafe_zeEventQueryStatus(event)
    if res == RESULT_NOT_READY
        return false
    elseif res == RESULT_SUCCESS
        return true
    else
        throw_api_error(res)
    end
end

function global_time(event)
    start_ref = Ref{UInt64}()
    zeEventGetTimestamp(event, ZE_EVENT_TIMESTAMP_GLOBAL_START, start_ref)
    stop_ref = Ref{UInt64}()
    zeEventGetTimestamp(event, ZE_EVENT_TIMESTAMP_GLOBAL_END, stop_ref)
    (start = start_ref[] == -1%UInt32 ? nothing : Int(start_ref[]),
     stop  = stop_ref[] == -1%UInt32 ?  nothing : Int(stop_ref[]))
end

function context_time(event)
    start_ref = Ref{UInt64}()
    zeEventGetTimestamp(event, ZE_EVENT_TIMESTAMP_CONTEXT_START, start_ref)
    stop_ref = Ref{UInt64}()
    zeEventGetTimestamp(event, ZE_EVENT_TIMESTAMP_CONTEXT_END, stop_ref)
    (start = start_ref[] == -1%UInt32 ? nothing : Int(start_ref[]),
     stop  = stop_ref[] == -1%UInt32 ?  nothing : Int(stop_ref[]))
end
