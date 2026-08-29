# context management and global state

# to avoid CUDA-style implicit state, where operations can fail if they are accidentally
# executed in the wrong context, ownership should always be encoded in each object.
# the functions below should only be used to determine initial ownership.

# XXX: rework this -- it doesn't work well when altering the state

export driver, driver!, device, device!, context, context!, global_stream, global_queue,
       synchronize, is_integrated

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
const global_contexts_lock = ReentrantLock()

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
        drv = driver()
        Base.@lock global_contexts_lock begin
            get!(global_contexts, drv) do
                ZeContext(drv)
            end
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

# The per-task submission target. `list` is where Julia-side work (kernels, copies,
# fills) is appended — an in-order asynchronous immediate command list, so appends are
# submitted to the device as they happen and no per-dispatch driver objects exist.
# `queue` exists only for SYCL/oneMKL interop, which requires a real command-queue
# handle, and is created on first use. Work on the two executes independently, so
# ordering at the oneMKL boundary is restored explicitly: `mkl_boundary!` (Julia → MKL)
# drains the list before handing out the SYCL queue, and `mkl_dirty` makes the next
# Julia-side submission drain the queue (MKL → Julia, see `mkl_wait!`).
mutable struct oneStream
    const ctx::ZeContext
    const dev::ZeDevice
    const list::oneL0.ZeImmediateCommandList
    queue::Union{Nothing, ZeCommandQueue}
    mkl_dirty::Bool
    # high-water mark of per-thread spill (bytes) among kernels submitted to this
    # stream, maintained by the scratch hedge (see src/compiler/execution.jl)
    scratch_hwm::Int
    const priority::oneL0.ze_command_queue_priority_t
end

function create_stream(ctx::ZeContext, dev::ZeDevice,
                       priority::oneL0.ze_command_queue_priority_t =
                           oneL0.ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
    # In-order immediate command lists entered the spec in 1.9, but the reported API
    # version is a floor, not a feature inventory: the Aurora LTS driver reports 1.6
    # while implementing them (they are DPC++'s production submission path on PVC).
    # Probe by creating — a driver without support rejects the flag — since there is
    # deliberately no fallback submission path.
    list = try
        oneL0.ZeImmediateCommandList(ctx, dev;
                                     flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                                     mode = oneL0.ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                     priority)
    catch err
        err isa oneL0.ZeError || rethrow()
        error("oneAPI.jl requires driver support for in-order immediate command lists " *
              "(Level Zero >= 1.9, or a driver implementing them regardless of its " *
              "reported API version $(oneL0.api_version(ctx.driver))); creation failed " *
              "with $(err.code)")
    end
    s = oneStream(ctx, dev, list, nothing, false, 0, priority)
    return register_stream!(ctx, dev, s)
end

"""
    global_stream(ctx::ZeContext, dev::ZeDevice) -> oneStream

Get the stream all oneAPI.jl operations of the calling task target for the given context
and device: an in-order asynchronous immediate command list for kernels, copies and
fills, plus a lazily-created companion command queue for SYCL/oneMKL interop. Streams
are cached per task and (context, device) pair.

See also: [`global_queue`](@ref), [`synchronize`](@ref)
"""
function global_stream(ctx::ZeContext, dev::ZeDevice)
    # NOTE: dev purposefully does not default to context() or device() to stress that
    #       objects should track ownership, and not rely on implicit global state.
    get!(task_local_storage(), (:oneStream, ctx, dev)) do
        create_stream(ctx, dev)
    end::oneStream
end

# the companion command queue of a stream, created on first use. Only the SYCL/oneMKL
# interop path needs one; pure-Julia workloads never create a queue.
function stream_queue(s::oneStream)
    q = s.queue
    q === nothing || return q
    s.queue = ZeCommandQueue(s.ctx, s.dev; flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                             priority = s.priority)
    return s.queue::ZeCommandQueue
end

"""
    global_queue(ctx::ZeContext, dev::ZeDevice) -> ZeCommandQueue

Get the calling task's companion command queue for the given context and device — the
queue oneMKL work is enqueued on through SYCL interop. Julia-side kernels, copies and
fills do not use it; they are appended to the task's [`global_stream`](@ref) immediate
command list instead.
"""
global_queue(ctx::ZeContext, dev::ZeDevice) = stream_queue(global_stream(ctx, dev))

# Register `stream` as a stream targeting (ctx, dev) so `synchronize_all_streams`/
# `release` can find and drain it before freeing buffers whose in-flight work it may
# still reference. EVERY stream that becomes a task's active stream must go through here
# — not just the one `global_stream` creates but also the replacement `KA.priority!`
# installs — or the unregistered stream's in-flight work can outlive a freed buffer (a
# use-after-free that faults and bans the context on the LTS NEO stack). Only the LTS
# stack maintains the registry; on the rolling stack this is a no-op. Returns `stream`.
function register_stream!(ctx::ZeContext, dev::ZeDevice, stream::oneStream)
    oneL0.LTS[] || return stream
    # disable finalizers while mutating the registry: a GC-driven finalizer on this
    # task could call back into `synchronize_all_streams` (the lock is reentrant) and
    # observe/mutate the registry mid-update.
    GC.enable_finalizers(false)
    try
        @lock stream_registry_lock begin
            push!(
                get!(Vector{Tuple{WeakRef, oneStream}}, stream_registry, (ctx, dev)),
                (WeakRef(current_task()), stream)
            )
        end
    finally
        GC.enable_finalizers(true)
    end
    return stream
end

# Registry of all streams created through `global_stream`, across tasks. Buffers can be
# freed from any task (GC finalizers), so `release` needs to be able to find the streams
# that may still have work in flight referencing the buffer; streams themselves are
# cached task-locally and would otherwise be unreachable from the finalizing task.
#
# Entries reference the stream *strongly*: the GC clears WeakRefs to a dead stream in
# the same cycle that queues its members' finalizers, i.e., before they run, so a
# WeakRef would hide the stream from `release` exactly when its in-flight work still
# references buffers about to be freed. The owning task is tracked weakly instead:
# streams are task-local, so once their task is dead no new work can reach them, and
# the entry can be dropped (allowing list and queue to be finalized) after a final
# synchronize.
const stream_registry_lock = ReentrantLock()
const stream_registry = Dict{Tuple{ZeContext, ZeDevice}, Vector{Tuple{WeakRef, oneStream}}}()

# synchronize all known streams that target the given context (and device, if
# specified), i.e., all streams whose in-flight work could possibly reference an
# allocation that is about to be freed. Drains both each stream's immediate command
# list and its companion queue (oneMKL work).
function synchronize_all_streams(ctx::ZeContext, dev::Union{ZeDevice, Nothing})
    # only the LTS stack populates the stream registry (see `global_stream`); on the
    # rolling stack this is a no-op and `release` frees directly.
    oneL0.LTS[] || return
    streams = oneStream[]
    stale = Tuple{WeakRef, oneStream}[]
    GC.enable_finalizers(false)
    try
        @lock stream_registry_lock begin
            for ((sctx, sdev), entries) in stream_registry
                sctx == ctx || continue
                (dev === nothing || sdev == dev) || continue
                for entry in entries
                    (task, stream) = entry
                    push!(streams, stream)
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
        # finalizers disabled so no stream member can be destroyed between collection
        # and synchronization; the null-handle checks are defense in depth against
        # lists/queues finalized before their stream was registered stale.
        for stream in streams
            stream.list.handle == C_NULL || oneL0.synchronize(stream.list)
            q = stream.queue
            (q === nothing || q.handle == C_NULL) || oneL0.synchronize(q)
        end
        # retire drained streams of dead tasks, allowing their list and queue to be
        # finalized (the finalizers synchronize once more before destroying, in case
        # the stream is dropped through other means).
        if !isempty(stale)
            @lock stream_registry_lock begin
                for ((sctx, sdev), entries) in stream_registry
                    sctx == ctx || continue
                    (dev === nothing || sdev == dev) || continue
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
    synchronize(stream::oneStream)

Block the host thread until all operations on the calling task's stream for the current
context and device have completed: work appended to the immediate command list as well
as oneMKL work on the companion queue.

This is useful for timing operations or ensuring that GPU work has finished before
accessing results on the CPU.

# Examples
```julia
x = oneArray(rand(1000))
y = x .+ 1
synchronize()  # Wait for GPU computation to complete
println("GPU work completed")
```

See also: [`global_stream`](@ref), [`context`](@ref), [`device`](@ref)
"""
function oneL0.synchronize(s::oneStream)
    oneL0.synchronize(s.list)
    q = s.queue
    if q !== nothing
        oneL0.synchronize(q)
        s.mkl_dirty = false
    end
    # every user-facing synchronization funnels through here, so this is where a device
    # exception surfaces (src/exceptions.jl); `synchronize_all_streams` deliberately does
    # not check, it runs from finalizers
    check_exceptions(s.ctx, s.dev)
    return
end

function oneL0.synchronize()
    oneL0.synchronize(global_stream(context(), device()))
end

# Julia → MKL ordering: everything Julia appended to the task's immediate list must be
# visible before oneMKL work is enqueued on the companion queue, which is a separate
# stream from the driver's point of view. Runs on every `sycl_queue` access; oneMKL
# wrappers must evaluate `sycl_queue(...)` only after all device-side argument
# preparation (temporaries, conversions), which holds today because the queue is the
# first ccall argument and Julia evaluates arguments left to right.
function mkl_boundary!(s::oneStream = global_stream(context(), device()))
    oneL0.synchronize(s.list)
    s.mkl_dirty = true
    return
end

# MKL → Julia ordering: consumed at the head of every Julia-side submission. One Bool
# load on the fast path; only a preceding oneMKL call makes it synchronize. The queue
# can be absent with the flag set: an FFT plan executing on this task runs on the queue
# it captured at construction, which need not be this stream's companion queue (whose
# creation the flag does not force).
@inline function mkl_wait!(s::oneStream)
    if s.mkl_dirty
        q = s.queue
        q === nothing || oneL0.synchronize(q)
        s.mkl_dirty = false
    end
    return
end

"""
    execute!(stream::oneStream) do list
        append_...!(list)
    end

Append operations to the stream's immediate command list, after waiting for any
outstanding oneMKL work. Appends are submitted to the device as they happen.
"""
@inline function oneL0.execute!(f::Base.Callable, s::oneStream)
    mkl_wait!(s)
    oneL0.execute!(f, s.list)
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

# Hands out the task's SYCL queue (wrapping the stream's companion command queue) for
# an imminent oneMKL call, which is why this is also the Julia → MKL ordering boundary:
# `mkl_boundary!` drains the immediate command list on EVERY access, so device work the
# wrappers prepared beforehand is complete before oneMKL work is enqueued.
function sycl_queue(queue)
    s = global_stream(queue.context, queue.device)
    mkl_boundary!(s)
    get!(task_local_storage(), (:SYCLQueue, queue.context, queue.device)) do
        syclQueue(sycl_context(queue.context, queue.device),
                  sycl_device(queue.device),
                  stream_queue(s))
    end
end
