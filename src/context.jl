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
    end::ZeDriver
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
    end::ZeDevice
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
    end::ZeContext
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

## command batching
#
# Each submission to an immediate command list is a system call, costing ~9 µs per kernel
# launch on i915. Kernel launches are therefore recorded in a regular command list, a
# batch, which is submitted to the stream's immediate list once it holds `BATCH_SIZE`
# launches, or before anything else needs the stream's work: other submissions,
# synchronization, host accesses to memory the stream uses (see `drain`), and oneMKL
# calls. The first launch after the stream was synchronized is submitted directly, so that
# a lone kernel doesn't wait for one of those. The memory the kernels of a batch use is
# kept alive until the batch has executed.

# initialized from ONEAPI_COMMAND_BATCHING in `__init__`
const BATCHING = Ref{Bool}(true)

const BATCH_SIZE = 32
const MAX_BATCHES = 8

mutable struct Batch
    const list::oneL0.ZeCommandList
    const event::oneL0.ZeEvent
    # copies of the references to the memory the batch's kernels use, by allocation
    const roots::Vector{DataRef}
    const rooted::Base.IdSet{Any}
    launches::Int
end

# whether the driver can submit regular command lists to an immediate one, which is an
# experimental extension, and executes in-order regular lists
const batching_support = Dict{Tuple{ZeContext, ZeDevice}, Bool}()
const batching_support_lock = ReentrantLock()
function batching_supported(ctx::ZeContext, dev::ZeDevice)
    @lock batching_support_lock get!(batching_support, (ctx, dev)) do
        try
            list = oneL0.ZeCommandList(ctx, dev; flags = oneL0.ZE_COMMAND_LIST_FLAG_IN_ORDER)
            oneL0.append_barrier!(list)
            close(list)
            immediate = oneL0.ZeImmediateCommandList(ctx, dev;
                flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                mode = oneL0.ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS)
            GC.@preserve list begin
                oneL0.zeCommandListImmediateAppendCommandListsExp(immediate, 1,
                                                                 Ref(list.handle),
                                                                 C_NULL, 0, C_NULL)
                oneL0.synchronize(immediate)
            end
            true
        catch err
            # the driver rejects the flag or the function, or the loader doesn't have it
            err isa Union{oneL0.ZeError, ErrorException} || rethrow()
            false
        end
    end
end

# The per-task submission target. `immediate` is where Julia-side work (kernels, copies,
# fills) is submitted: an in-order asynchronous immediate command list. Kernel launches are
# mostly recorded in batches first, see above.
# `queue` exists only for SYCL/oneMKL interop, which requires a real command-queue
# handle, and is created on first use. Work on the two executes independently, so
# ordering at the oneMKL boundary is restored explicitly: `mkl_boundary!` (Julia → MKL)
# drains the list before handing out the SYCL queue, and `mkl_dirty` makes the next
# Julia-side submission drain the queue (MKL → Julia, see `mkl_wait!`).
mutable struct oneStream
    const ctx::ZeContext
    const dev::ZeDevice
    # only append to this through `execute!` or `launch!`, which hold `lock`
    const immediate::oneL0.ZeImmediateCommandList
    const lock::ReentrantLock

    # command batching (see above), protected by `lock`
    const batching::Bool
    batch::Union{Nothing, Batch}    # the batch being recorded
    const submitted::Vector{Batch}  # in submission order
    const spare::Vector{Batch}
    nbatches::Int
    events::Union{Nothing, oneL0.ZeEventPool}
    idle::Bool                      # submit the next launch directly
    recorder::Union{Nothing, Task}  # the last task to record a launch, if registered
    const retired::Vector{DataRef}  # roots of executed batches, still to be released
    const failed::Vector{Batch}     # possibly submitted, so never released
    # a failed submission, reported by the next synchronization
    error::Union{Nothing, Exception}

    # atomic because other tasks read it when waiting for work on memory this stream owns
    Base.@atomic queue::Union{Nothing, ZeCommandQueue}
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
    batching = BATCHING[] && batching_supported(ctx, dev)
    s = oneStream(ctx, dev, list, ReentrantLock(),
                  batching, nothing, Batch[], Batch[], 0, nothing, true, nothing, DataRef[],
                  Batch[], nothing, nothing, false, 0, priority)
    return register_stream!(ctx, dev, s)
end

# The functions below expect the caller to hold `s.lock`.

check_error(s::oneStream) = (s.error === nothing || throw(s.error); nothing)

function new_batch!(s::oneStream)
    if s.events === nothing
        s.events = oneL0.ZeEventPool(s.ctx, MAX_BATCHES, s.dev;
                                     flags = oneL0.ZE_EVENT_POOL_FLAG_HOST_VISIBLE)
    end
    s.nbatches += 1
    list = oneL0.ZeCommandList(s.ctx, s.dev; flags = oneL0.ZE_COMMAND_LIST_FLAG_IN_ORDER)
    event = oneL0.ZeEvent(s.events, s.nbatches;
                          signal = oneL0.ZE_EVENT_SCOPE_FLAG_HOST,
                          wait = oneL0.ZE_EVENT_SCOPE_FLAG_HOST)
    return Batch(list, event, DataRef[], Base.IdSet{Any}(), 0)
end

# recycle the batches that have executed. their roots are released by `release_retired!`,
# not here, as that can free memory, which waits for the GPU on some stacks.
function retire!(s::oneStream)
    while !isempty(s.submitted) && Base.isdone(first(s.submitted).event)
        b = popfirst!(s.submitted)
        isempty(b.roots) || queue_release!(s)
        append!(s.retired, b.roots)
        empty!(b.roots)
        empty!(b.rooted)
        reset(b.list)
        reset(b.event)
        b.launches = 0
        push!(s.spare, b)
    end
    return
end

# the batch to record a launch in, or `nothing` if it should be submitted directly
function open_batch!(s::oneStream)
    b = s.batch
    b === nothing || return b
    retire!(s)
    if isempty(s.spare)
        # with all batches executing, submit directly rather than waiting for one: the
        # caller holds the kernel's lock, which another task may need to make progress
        s.nbatches < MAX_BATCHES || return nothing
        push!(s.spare, new_batch!(s))
    end
    b = pop!(s.spare)
    s.batch = b
    return b
end

function flush_batch!(s::oneStream)
    b = s.batch
    b === nothing && return
    s.batch = nothing
    try
        close(b.list)
        oneL0.zeCommandListImmediateAppendCommandListsExp(s.immediate, 1, Ref(b.list.handle),
                                                         b.event, 0, C_NULL)
    catch err
        # the batch may have been submitted, so keep it and the memory it uses alive
        push!(s.failed, b)
        s.error = err
        rethrow()
    end
    push!(s.submitted, b)
    return
end

# keep `roots`, the references to the memory a launch uses, alive with the batch
function root!(b::Batch, roots)
    roots === nothing && return
    for ref in roots
        managed = ref[]
        managed in b.rooted && continue
        push!(b.rooted, managed)
        push!(b.roots, copy(ref))
    end
    return
end

"""
    flush!(stream::oneStream = global_stream(context(), device()))

Submit the kernel launches the stream has recorded but not yet submitted to the device.
Launches are submitted in batches, so call this before doing significant work on the host
that should overlap with the GPU executing those kernels. Synchronizing flushes too.
"""
function flush!(s::oneStream = global_stream(context(), device()))
    @lock s.lock begin
        check_error(s)
        flush_batch!(s)
    end
    return
end

# streams with roots to release, which happens at `release_pending!`
const pending_release = Base.IdSet{oneStream}()
const pending_release_lock = ReentrantLock()
const npending_release = Threads.Atomic{Int}(0)

function queue_release!(s::oneStream)
    @lock pending_release_lock push!(pending_release, s)
    npending_release[] = 1
    return
end

# release the memory of executed batches. the caller must not hold any lock, as freeing
# memory can wait for the GPU on some stacks.
function release_pending!()
    npending_release[] == 0 && return
    streams = @lock pending_release_lock begin
        npending_release[] = 0
        streams = collect(pending_release)
        empty!(pending_release)
        streams
    end
    foreach(release_retired!, streams)
    return
end

function release_retired!(s::oneStream)
    roots = @lock s.lock begin
        isempty(s.retired) && return
        roots = copy(s.retired)
        empty!(s.retired)
        roots
    end
    foreach(unsafe_free!, roots)
    return
end

# streams that recorded launches since they were last synchronized, so that `synchronize`
# can submit what tasks recorded before finishing. the stream is referenced strongly, so
# that it doesn't get destroyed with work recorded. a stream's lock is taken before this
# one, and the stream's `recorder` is the task last to record on it.
const batching_streams = Base.IdSet{oneStream}()
const batching_streams_lock = ReentrantLock()

function register_batching!(s::oneStream)
    @lock batching_streams_lock push!(batching_streams, s)
    s.recorder = current_task()
    return
end

# forget a stream once nothing it recorded is still to execute. streams with batches that
# failed to submit, which may still execute, are kept around.
function maybe_unregister_batching!(s::oneStream)
    if s.recorder !== nothing && s.batch === nothing && isempty(s.submitted) &&
       isempty(s.failed)
        @lock batching_streams_lock delete!(batching_streams, s)
        s.recorder = nothing
    end
    return
end

# wait for the streams on which tasks that have finished were the last to record. other
# tasks' streams are left alone: their tasks synchronize them.
function adopt_orphans!(ctx::ZeContext, dev::ZeDevice)
    streams = @lock batching_streams_lock collect(batching_streams)
    for s in streams
        (s.ctx == ctx && s.dev == dev) || continue
        orphaned = @lock s.lock begin
            task = s.recorder
            task !== nothing && istaskdone(task)
        end
        orphaned || continue
        err = nothing
        try
            drain(s)
        catch e
            err = e
        end
        @lock s.lock begin
            # reported here, as the task that would have isn't around anymore
            err === nothing || (s.error = nothing)
            maybe_unregister_batching!(s)
        end
        err === nothing || throw(err)
    end
    return
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
    q = Base.@atomic s.queue
    q === nothing || return q
    q = ZeCommandQueue(s.ctx, s.dev; flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
                       priority = s.priority)
    Base.@atomic s.queue = q
    return q
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
# — not just the one `global_stream` creates but also the replacement `KI.priority!`
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
            stream.immediate.handle == C_NULL || oneL0.synchronize(stream.immediate)
            q = Base.@atomic stream.queue
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
    synchronize(; blocking=false)
    synchronize(stream::oneStream; blocking=false)

Block the calling task until all operations on its stream for the current context and
device have completed: work appended to the immediate command list as well as oneMKL work
on the companion queue.

Unless `blocking` is set, other tasks keep running while waiting: the host thread is only
blocked in the driver when the work is already done.

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
function oneL0.synchronize(s::oneStream; blocking::Bool=false)
    err = synchronize_stream!(s; blocking)
    release_pending!()
    err === nothing || throw(err)
    return
end

# synchronize a stream, returning the error of a failed submission
function synchronize_stream!(s::oneStream; blocking::Bool=false)
    sync = blocking ? oneL0.synchronize : oneL0.nonblocking_synchronize
    # a failed submission is recorded in `s.error`, and reported below
    @lock s.lock try
        flush_batch!(s)
    catch
    end
    sync(s.immediate)
    q = Base.@atomic s.queue
    if q !== nothing
        sync(q)
        s.mkl_dirty = false
    end
    err = @lock s.lock begin
        retire!(s)
        s.idle = true
        maybe_unregister_batching!(s)
        err = s.error
        s.error = nothing
        err
    end
    return err
end

function oneL0.synchronize(; blocking::Bool=false)
    ctx = context()
    dev = device()
    oneL0.synchronize(global_stream(ctx, dev); blocking)
    adopt_orphans!(ctx, dev)
    release_pending!()
end

# Julia → MKL ordering: everything Julia appended to the task's immediate list must be
# visible before oneMKL work is enqueued on the companion queue, which is a separate
# stream from the driver's point of view. Runs on every `sycl_queue` access; oneMKL
# wrappers must evaluate `sycl_queue(...)` only after all device-side argument
# preparation (temporaries, conversions), which holds today because the queue is the
# first ccall argument and Julia evaluates arguments left to right.
function mkl_boundary!(s::oneStream = global_stream(context(), device()))
    @lock s.lock begin
        check_error(s)
        flush_batch!(s)
    end
    oneL0.synchronize(s.immediate)
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
        q = Base.@atomic s.queue
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
    @lock s.lock begin
        check_error(s)
        flush_batch!(s)
        oneL0.execute!(f, s.immediate)
    end
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
