# queue

export ZeCommandQueue, synchronize

mutable struct ZeCommandQueue
    handle::ze_command_queue_handle_t

    function ZeCommandQueue(dev::ZeDevice)
        desc_ref = Ref(ze_command_queue_desc_t(
            ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT,
            ZE_COMMAND_QUEUE_FLAG_NONE,
            ZE_COMMAND_QUEUE_MODE_DEFAULT,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
            0
        ))
        handle_ref = Ref{ze_command_queue_handle_t}()
        zeCommandQueueCreate(dev, desc_ref, handle_ref)
        obj = new(handle_ref[])
        finalizer(obj) do obj
            zeCommandQueueDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_command_queue_handle_t}, queue::ZeCommandQueue) = queue.handle

synchronize(queue::ZeCommandQueue, timeout::Number=typemax(UInt32)) =
    zeCommandQueueSynchronize(queue, timeout)
