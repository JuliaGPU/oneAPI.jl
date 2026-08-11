# list

export ZeCommandList, ZeImmediateCommandList, AbstractZeCommandList, execute!

abstract type AbstractZeCommandList end

mutable struct ZeCommandList <: AbstractZeCommandList
    handle::ze_command_list_handle_t

    context::ZeContext
    device::ZeDevice

    function ZeCommandList(ctx::ZeContext, dev::ZeDevice, ordinal=1; flags=0)
        desc_ref = Ref(ze_command_list_desc_t(;
            commandQueueGroupOrdinal=ordinal-1, flags,
        ))
        handle_ref = Ref{ze_command_list_handle_t}()
        zeCommandListCreate(ctx, dev, desc_ref, handle_ref)
        obj = new(handle_ref[], ctx, dev)
        finalizer(obj) do obj
            zeCommandListDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_command_list_handle_t}, list::AbstractZeCommandList) = list.handle

Base.:(==)(a::AbstractZeCommandList, b::AbstractZeCommandList) = a.handle == b.handle
Base.hash(e::AbstractZeCommandList, h::UInt) = hash(e.handle, h)

Base.close(list::ZeCommandList) = zeCommandListClose(list)

Base.reset(list::ZeCommandList) = zeCommandListReset(list)

"""
    ZeCommandList(dev::ZeDevice, ...) do list
        append_...!(list)
    end

Create a command list for device `dev`, passing in a do block that appends operations.
The list is then closed and can be used immediately, e.g. for execution.

"""
function ZeCommandList(f::Base.Callable, args...; kwargs...)
    list = ZeCommandList(args...; kwargs...)
    f(list)
    close(list)
    return list
end

"""
    ZeImmediateCommandList(ctx::ZeContext, dev::ZeDevice, ordinal=1, index=1;
                           flags=0, mode=ZE_COMMAND_QUEUE_MODE_DEFAULT,
                           priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL)

Create an immediate command list: appended commands are submitted to the device as they
are appended, with no separate close/execute step and no per-submission list object.
The descriptor is a command *queue* descriptor; pass
`flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER` and `mode=ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS`
for an asynchronous stream with in-order semantics (requires Level Zero >= 1.9).
Synchronize with [`synchronize`](@ref).
"""
mutable struct ZeImmediateCommandList <: AbstractZeCommandList
    handle::ze_command_list_handle_t

    context::ZeContext
    device::ZeDevice
    ordinal::Int

    function ZeImmediateCommandList(ctx::ZeContext, dev::ZeDevice, ordinal=1, index=1;
                                    flags=0,
                                    mode::ze_command_queue_mode_t=ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                    priority::ze_command_queue_priority_t=ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
        desc_ref = Ref(ze_command_queue_desc_t(;
            ordinal=ordinal-1, index=index-1, flags, mode, priority
        ))
        handle_ref = Ref{ze_command_list_handle_t}()
        zeCommandListCreateImmediate(ctx, dev, desc_ref, handle_ref)
        obj = new(handle_ref[], ctx, dev, ordinal)
        finalizer(obj) do obj
            # unlike a regular list, an immediate list can have work in flight at
            # finalization on any stack, and destroying it then is illegal. Bounded
            # unchecked drain, leaking the list on timeout — same rationale as the
            # queue finalizer in cmdqueue.jl: an infinite wait on event-gated work
            # whose event is never signaled would hang GC and process exit.
            if unchecked_zeCommandListHostSynchronize(obj, FINALIZER_SYNC_TIMEOUT_NS) == RESULT_NOT_READY
                @warn "Leaking an immediate command list still busy after $(FINALIZER_SYNC_TIMEOUT_NS ÷ 1_000_000_000)s to avoid blocking finalization" maxlog = 1
                return
            end
            zeCommandListDestroy(obj)
            # mark destroyed: the stream registry can still reach this list, and
            # synchronizing a destroyed handle crashes in the driver
            obj.handle = ze_command_list_handle_t(C_NULL)
        end
        obj
    end
end

"""
    synchronize(list::ZeImmediateCommandList, timeout=typemax(UInt64))

Block the host until all commands appended to the immediate command list have completed.
"""
synchronize(list::ZeImmediateCommandList, timeout::Number=typemax(UInt64)) =
    zeCommandListHostSynchronize(list, timeout)

# Opt-in workaround for the Aurora LTS NEO stack (set ONEAPI_SYNC_EACH_SUBMISSION=1).
# Under heavy multi-process oversubscription of a single tile, a whole-queue
# `zeCommandQueueSynchronize` does not reliably retire the tail of an earlier,
# separately-submitted command list — producing silent "dropped tail" corruption (the
# last work-item of a kernel / last element of a copy is missing). See
# docs/src/lts.md. Synchronizing after *every* submission eliminates it, at a large
# throughput cost (~3x), so it is off by default and only enabled when correctness under
# oversubscription matters more than speed.
const SYNC_EACH_SUBMISSION = Ref{Bool}(false)

"""
    sync_each_submission() -> Bool

Whether [`execute!`](@ref) follows every command-list submission with a full
`zeCommandQueueSynchronize` (the Aurora LTS "dropped tail" workaround). See
[`sync_each_submission!`](@ref).
"""
sync_each_submission() = SYNC_EACH_SUBMISSION[]

"""
    sync_each_submission!(enable::Bool) -> Bool

Enable or disable synchronizing after every submission, returning the previous setting.
Initialized from the `ONEAPI_SYNC_EACH_SUBMISSION` environment variable.
"""
function sync_each_submission!(enable::Bool)
    old = SYNC_EACH_SUBMISSION[]
    SYNC_EACH_SUBMISSION[] = enable
    return old
end

"""
    sync_each_submission(f, enable::Bool)

Run `f()` with the workaround temporarily set to `enable`, restoring the previous setting
afterwards. Use this for submit-then-signal patterns that would otherwise deadlock, where a
synchronize is forced before the event gating the submitted work is signaled.
"""
function sync_each_submission(f::Base.Callable, enable::Bool)
    old = sync_each_submission!(enable)
    return try
        f()
    finally
        sync_each_submission!(old)
    end
end

function execute!(queue::ZeCommandQueue, lists::Vector{ZeCommandList}, fence = nothing)
    r = zeCommandQueueExecuteCommandLists(queue, length(lists), lists, something(fence, C_NULL))
    sync_each_submission() && synchronize(queue)
    return r
end

"""
    execute!(queue::ZeCommandQueue, ...) do list
        append_...!(list)
    end

Create a command list for the device that owns `queue`, passing in a do block that appends
operations. The list is then closed and executed on the queue.
"""
function execute!(f::Base.Callable, queue::ZeCommandQueue, fence=nothing; kwargs...)
    list = ZeCommandList(f, queue.context, queue.device, queue.ordinal; kwargs...)
    execute!(queue, [list], fence)
end

"""
    execute!(list::ZeImmediateCommandList) do list
        append_...!(list)
    end

Append operations to an immediate command list. Each append is submitted to the device
as it happens, so there is no separate close/execute step; the return value is that of
the do block.
"""
function execute!(f::Base.Callable, list::ZeImmediateCommandList)
    ret = f(list)
    sync_each_submission() && synchronize(list)
    return ret
end
