# queue

export ZeCommandQueue, synchronize

mutable struct ZeCommandQueue
    handle::ze_command_queue_handle_t

    dev::ZeDevice
    ordinal::Int

    function ZeCommandQueue(dev::ZeDevice, ordinal=0;
                            flags=ZE_COMMAND_QUEUE_FLAG_NONE,
                            mode::ze_command_queue_mode_t=ZE_COMMAND_QUEUE_MODE_DEFAULT,
                            priority::ze_command_queue_priority_t=ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
        desc_ref = Ref(ze_command_queue_desc_t(
            ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT,
            flags, mode, priority, ordinal
        ))
        handle_ref = Ref{ze_command_queue_handle_t}()
        zeCommandQueueCreate(dev, desc_ref, handle_ref)
        obj = new(handle_ref[], dev, ordinal)
        finalizer(obj) do obj
            zeCommandQueueDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_command_queue_handle_t}, queue::ZeCommandQueue) = queue.handle

synchronize(queue::ZeCommandQueue, timeout::Number=typemax(UInt32)) =
    zeCommandQueueSynchronize(queue, timeout)
