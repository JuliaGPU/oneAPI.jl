# context management and global state

# to avoid CUDA-style implicit state, where operations can fail if they are accidentally
# executed in the wrong context, ownership should always be encoded in each object.
# the functions below should only be used to determine initial ownership.

# XXX: rework this -- it doesn't work well when altering the state

export driver, driver!, device, device!, context, context!, global_queue, synchronize, is_integrated

"""
    driver() -> ZeDriver

Get the current Level Zero driver for the calling task. If no driver has been explicitly
set with [`driver!`](@ref), returns the first available driver.

The driver selection is task-local, allowing different Julia tasks to use different drivers.

# Examples
```julia
drv = driver()
println("Using driver: ", drv)
```

See also: `driver!`, `drivers`
"""
function driver()
    get!(task_local_storage(), :ZeDriver) do
        first(drivers())
    end
end

"""
    driver!(drv::ZeDriver)

Set the current Level Zero driver for the calling task. This also clears the current
device selection, as devices are associated with specific drivers.

The driver selection is task-local, allowing different Julia tasks to use different drivers.

# Arguments
- `drv::ZeDriver`: The driver to use for subsequent operations.

# Examples
```julia
drv = drivers()[2]  # Select second available driver
driver!(drv)
```

See also: `driver`, `drivers`
"""
function driver!(drv::ZeDriver)
    task_local_storage(:ZeDriver, drv)
    delete!(task_local_storage(), :ZeDevice)
end

"""
    device() -> ZeDevice

Get the current Level Zero device for the calling task. If no device has been explicitly
set with [`device!`](@ref), returns the first available device for the current driver.

The device selection is task-local, allowing different Julia tasks to use different devices.

# Examples
```julia
dev = device()
println("Using device: ", dev)
```

See also: `device!`, `devices`, `driver`
"""
function device()
    get!(task_local_storage(), :ZeDevice) do
        first(devices(driver()))
    end
end

"""
    device!(dev::ZeDevice)
    device!(i::Int)

Set the current Level Zero device for the calling task.

The device selection is task-local, allowing different Julia tasks to use different devices.

# Arguments
- `dev::ZeDevice`: The device to use for subsequent operations.
- `i::Int`: Device index (1-based) from the list of available devices for the current driver.

# Examples
```julia
# Select by device object
dev = devices()[2]
device!(dev)

# Select by index
device!(2)  # Select second device
```

See also: [`device`](@ref), [`devices`](@ref)
"""
function device!(drv::ZeDevice)
    task_local_storage(:ZeDevice, drv)
end
function device!(i::Int)
    devs = devices(driver())
    if i < 1 || i > length(devs)
        throw(ArgumentError("Invalid device index $i (must be between 1 and $(length(devs)))"))
    end
    return device!(devs[i])
end

"""
    is_integrated(dev::ZeDevice=device()) -> Bool

Check if the given device is an integrated GPU (i.e., integrated with the host processor).

Integrated GPUs share memory with the CPU and are typically found in laptop and desktop
processors with integrated graphics.

# Arguments
- `dev::ZeDevice`: The device to check. Defaults to the current device.

# Returns
- `true` if the device is integrated, `false` otherwise (e.g., discrete GPU).

# Examples
```julia
if is_integrated()
    println("Running on integrated graphics")
else
    println("Running on discrete GPU")
end

# Check a specific device
dev = devices()[1]
is_integrated(dev)
```

See also: [`device`](@ref), [`devices`](@ref)
"""
function is_integrated(dev::ZeDevice=device())
    props = oneL0.properties(dev)
    return (props.flags & oneL0.ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0
end

const global_contexts = Dict{ZeDriver,ZeContext}()

"""
    context() -> ZeContext

Get the current Level Zero context for the calling task. If no context has been explicitly
set with [`context!`](@ref), returns a global context for the current driver.

Contexts manage the lifetime of resources like memory allocations and command queues.
The context selection is task-local, but contexts themselves are cached globally per driver.

# Examples
```julia
ctx = context()
println("Using context: ", ctx)
```

See also: [`context!`](@ref), [`driver`](@ref)
"""
function context()
    get!(task_local_storage(), :ZeContext) do
        get!(global_contexts, driver()) do
            ZeContext(driver())
        end
    end
end

"""
    context!(ctx::ZeContext)

Set the current Level Zero context for the calling task.

The context selection is task-local, allowing different Julia tasks to use different contexts.

# Arguments
- `ctx::ZeContext`: The context to use for subsequent operations.

# Examples
```julia
ctx = ZeContext(driver())
context!(ctx)
```

See also: `context`, `ZeContext`
"""
function context!(ctx::ZeContext)
    task_local_storage(:ZeContext, ctx)
end

"""
    global_queue(ctx::ZeContext, dev::ZeDevice) -> ZeCommandQueue

Get the global command queue for the given context and device. This queue is used as the
default queue for executing operations, guaranteeing expected semantics when using a device
on a Julia task.

The queue is created with in-order execution flags, meaning commands are executed in the
order they are submitted. Queues are cached per task and (context, device) pair.

# Arguments
- `ctx::ZeContext`: The context for the command queue.
- `dev::ZeDevice`: The device for the command queue.

# Returns
- `ZeCommandQueue`: A cached command queue with in-order execution.

# Examples
```julia
ctx = context()
dev = device()
queue = global_queue(ctx, dev)
```

See also: `context`, `device`, `synchronize`
"""
function global_queue(ctx::ZeContext, dev::ZeDevice)
    # NOTE: dev purposefully does not default to context() or device() to stress that
    #       objects should track ownership, and not rely on implicit global state.
    get!(task_local_storage(), (:ZeCommandQueue, ctx, dev)) do
        queue = ZeCommandQueue(ctx, dev; flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER)
        if oneL0.LTS[]
            # disable finalizers while mutating the registry: a GC-driven finalizer on this
            # task could call back into `synchronize_all_queues` (the lock is reentrant) and
            # observe/mutate the registry mid-update.
            GC.enable_finalizers(false)
            try
                @lock queue_registry_lock begin
                    push!(get!(Vector{Tuple{WeakRef,ZeCommandQueue}}, queue_registry, (ctx, dev)),
                          (WeakRef(current_task()), queue))
                end
            finally
                GC.enable_finalizers(true)
            end
        end
        queue
    end
end

# Registry of all queues created through `global_queue`, across tasks. Buffers can be
# freed from any task (GC finalizers), so `release` needs to be able to find the queues
# that may still have work in flight referencing the buffer; queues themselves are
# cached task-locally and would otherwise be unreachable from the finalizing task.
#
# Entries reference the queue *strongly*: the GC clears WeakRefs to a dead queue in the
# same cycle that queues its finalizer, i.e., before the finalizer runs, so a WeakRef
# would hide the queue from `release` exactly when its in-flight work still references
# buffers about to be freed. The owning task is tracked weakly instead: queues are
# task-local, so once their task is dead no new work can reach them, and the entry can
# be dropped (allowing the queue to be finalized) after a final synchronize.
const queue_registry_lock = ReentrantLock()
const queue_registry = Dict{Tuple{ZeContext,ZeDevice},Vector{Tuple{WeakRef,ZeCommandQueue}}}()

# synchronize all known queues that target the given context (and device, if specified),
# i.e., all queues whose in-flight work could possibly reference an allocation that is
# about to be freed.
function synchronize_all_queues(ctx::ZeContext, dev::Union{ZeDevice,Nothing})
    # only the LTS stack populates the queue registry (see `global_queue`); on the
    # rolling stack this is a no-op and `release` frees directly.
    oneL0.LTS[] || return
    queues = ZeCommandQueue[]
    stale = Tuple{WeakRef,ZeCommandQueue}[]
    GC.enable_finalizers(false)
    try
        @lock queue_registry_lock begin
            for ((qctx, qdev), entries) in queue_registry
                qctx == ctx || continue
                (dev === nothing || qdev == dev) || continue
                for entry in entries
                    (task, queue) = entry
                    queue.handle == C_NULL && continue  # finalized, handle destroyed
                    push!(queues, queue)
                    # entries whose task was already dead at this point cannot
                    # receive new work, so they are safe to retire after the sync
                    if task.value === nothing || istaskdone(task.value::Task)
                        push!(stale, entry)
                    end
                end
            end
        end
        # synchronize outside the lock: this can block for as long as a kernel runs,
        # and finalizers running concurrently also need to take the lock. Keep
        # finalizers disabled so none of the collected queues can be destroyed
        # between collection and synchronization.
        for queue in queues
            oneL0.synchronize(queue)
        end
        # retire drained queues of dead tasks, allowing them to be finalized (the
        # finalizer synchronizes once more before destroying the queue, in case
        # the queue is dropped through other means).
        if !isempty(stale)
            @lock queue_registry_lock begin
                for ((qctx, qdev), entries) in queue_registry
                    qctx == ctx || continue
                    (dev === nothing || qdev == dev) || continue
                    filter!(entry -> !any(s -> s === entry, stale), entries)
                end
            end
        end
    finally
        GC.enable_finalizers(true)
    end
    return
end

"""
    synchronize()

Block the host thread until all operations on the global command queue for the current
context and device have completed.

This is useful for timing operations or ensuring that GPU work has finished before
accessing results on the CPU.

# Examples
```julia
x = oneArray(rand(1000))
y = x .+ 1
synchronize()  # Wait for GPU computation to complete
println("GPU work completed")
```

See also: [`global_queue`](@ref), [`context`](@ref), [`device`](@ref)
"""
function oneL0.synchronize()
    oneL0.synchronize(global_queue(context(), device()))
end

# re-export and augment parts of oneL0 to make driver and device selection easier
export drivers, devices

"""
    devices() -> Vector{ZeDevice}
    devices(drv::ZeDriver) -> Vector{ZeDevice}

Return a list of available Level Zero devices. Without arguments, returns devices for
the current driver. With a driver argument, returns devices for that specific driver.

# Examples
```julia
# Get devices for current driver
devs = devices()
println("Found ", length(devs), " devices")

# Get devices for specific driver
drv = drivers()[1]
devs = devices(drv)
```

See also: `device`, `device!`, `drivers`
"""
oneL0.devices() = devices(driver())


## SYCL state

# XXX: including objects in the TLS key is bad for performance

export sycl_platform, sycl_device, sycl_context, sycl_queue

function sycl_platform(drv=driver())
    get!(task_local_storage(), (:SYCLPlatform, drv)) do
        syclPlatform(drv)
    end
end

function sycl_device(dev=device())
    get!(task_local_storage(), (:SYCLDevice, dev)) do
        syclDevice(sycl_platform(), dev)
    end
end

function sycl_context(ctx=context(), dev=device())
    get!(task_local_storage(), (:SYCLContext, dev)) do
        syclContext([sycl_device(dev)], ctx)
    end
end

function sycl_queue(queue)
    get!(task_local_storage(), (:SYCLQueue, queue.context, queue.device)) do
        syclQueue(sycl_context(queue.context, queue.device),
                  sycl_device(queue.device),
                  global_queue(queue.context, queue.device))
    end
end
