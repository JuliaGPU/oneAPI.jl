# context management and global state

# to avoid CUDA-style implicit state, where operations can fail if they are accidentally
# executed in the wrong context, ownership should always be encoded in each object.
# the accessors below should only be used to determine initial ownership.

export driver, driver!, device, device!, context, context!, global_queue, synchronize

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

const global_contexts = Dict{ZeDriver,ZeContext}()

function context()
    get!(task_local_storage(), :ZeContext) do
        get!(global_contexts, driver()) do
            ZeContext(driver())
        end
    end
end

function context!(ctx::ZeContext)
    task_local_storage(:ZeContext, ctx)
end

# the global queue can be used as a default queue to execute operations on,
# guaranteeing expected semantics when using a device on a Julia task.

function global_queue(ctx::ZeContext, dev::ZeDevice)
    # NOTE: dev purposefully does not default to context() or device() to stress that
    #       objects should track ownership, and not rely on implicit global state.
    get!(task_local_storage(), (:ZeCommandQueue, ctx, dev)) do
        ZeCommandQueue(ctx, dev)
    end
end

function oneL0.synchronize()
    oneL0.synchronize(global_queue(context(), device()))
end


function Adapt.adapt_storage(dev::ZeDevice, x)
    prev_dev = device()
    try
        device!(dev)
        Adapt.adapt_storage(oneArray, x)
    finally
        device!(prev_dev)
    end
end


# ToDo: implement Sys.total_memory(dev::ZeDevice)
# ToDo: implement Sys.free_memory(dev::ZeDevice)
