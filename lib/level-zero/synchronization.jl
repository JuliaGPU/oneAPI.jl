# cooperative synchronization
#
# `zeCommandListHostSynchronize`, `zeCommandQueueSynchronize` and `zeEventHostSynchronize`
# block the calling thread until the work has completed, so no other task can run on it in
# the meantime. Instead, wait using GPUToolbox's `cooperative_wait`: first poll, which keeps
# the latency of short operations low, and then block in the driver on a separate thread,
# while the calling task yields.

using GPUToolbox: cooperative_wait

export nonblocking_synchronize

const SyncObject = Union{ZeImmediateCommandList, ZeCommandQueue, ZeEvent}

# with a zero timeout, a synchronization is a query
function check_done(res::ze_result_t)
    if res == RESULT_NOT_READY
        return false
    elseif res == RESULT_SUCCESS
        return true
    else
        throw_api_error(res)
    end
end
Base.isdone(list::ZeImmediateCommandList) =
    check_done(unchecked_zeCommandListHostSynchronize(list, 0))
Base.isdone(queue::ZeCommandQueue) =
    check_done(unchecked_zeCommandQueueSynchronize(queue, 0))

# the blocking synchronization, marked GC-safe so that it doesn't keep the GC from running
gcsafe_synchronize(list::ZeImmediateCommandList) =
    @gcsafe_ccall libze_loader.zeCommandListHostSynchronize(
        list::ze_command_list_handle_t, typemax(UInt64)::UInt64)::ze_result_t
gcsafe_synchronize(queue::ZeCommandQueue) =
    @gcsafe_ccall libze_loader.zeCommandQueueSynchronize(
        queue::ze_command_queue_handle_t, typemax(UInt64)::UInt64)::ze_result_t
gcsafe_synchronize(event::ZeEvent) =
    @gcsafe_ccall libze_loader.zeEventHostSynchronize(
        event::ze_event_handle_t, typemax(UInt64)::UInt64)::ze_result_t

"""
    nonblocking_synchronize(list_or_queue)
    nonblocking_synchronize(event)

Wait for the work on an immediate command list or command queue to complete, or for an
event to be signaled, like [`synchronize`](@ref) or `wait`, but without blocking the calling
thread: other tasks keep running while this one waits.
"""
function nonblocking_synchronize(obj::SyncObject)
    # when polling found the work to be done, synchronize again to check for errors
    res = @something(cooperative_wait(gcsafe_synchronize, obj; isdone=Base.isdone),
                     gcsafe_synchronize(obj))
    res == RESULT_SUCCESS || throw_api_error(res)
    return
end
