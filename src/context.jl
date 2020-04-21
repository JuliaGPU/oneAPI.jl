# context management and global state

# to avoid CUDA-style implicit state, where operations can fail if they are accidentally
# executed in the wrong context, ownership should always be encoded in each object.
# the accessors below should only be used to determine initial ownership.

function driver()
    get!(task_local_storage(), :ZeDriver) do
        first(drivers())
    end
end

function driver!(drv::ZeDriver)
    task_local_storage(:ZeDriver, drv)
    delete!(task_local_storage(), :ZeDevice)
end

function device()
    get!(task_local_storage(), :ZeDevice) do
        first(devices(driver()))
    end
end

function device!(drv::ZeDevice)
    task_local_storage(:ZeDevice, drv)
end

# the global queue can be used as a default queue to execute operations on,
# guaranteeing expected semantics when using a device on a Julia task.

function global_queue(dev::ZeDevice)
    # NOTE: dev purposefully does not default to device() to stress that objects should
    #       track device ownership, and not rely on the currently active device.
    get!(task_local_storage(), (:ZeCommandQueue, dev)) do
        ZeCommandQueue(dev)
    end
end
