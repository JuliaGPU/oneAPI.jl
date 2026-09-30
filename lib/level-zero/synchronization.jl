# cooperative synchronization
#
# `zeCommandListHostSynchronize`, `zeCommandQueueSynchronize` and `zeEventHostSynchronize`
# block the calling thread until the work has completed, so no other task can run on it in the meantime. As CUDA.jl
# does, first busy-wait on a non-blocking query, which keeps the latency of short
# operations low, and then block in the driver on a separate thread, while the calling
# task waits for that thread without blocking the scheduler.

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

# once the work has completed, synchronizing a list or queue doesn't block anymore; a
# signaled event needs nothing more
finish_synchronization(obj::Union{ZeImmediateCommandList, ZeCommandQueue}) = synchronize(obj)
finish_synchronization(::ZeEvent) = nothing


## bidirectional channel

# custom, unbuffered channel that supports returning a value to the sender
# without the need for a second channel
struct BidirectionalChannel{I,O} <: AbstractChannel{I}
    cond_take::Threads.Condition                 # waiting for data to become available
    cond_put::Threads.Condition                  # waiting for a writeable slot
    cond_ret::Threads.Condition                  # waiting for a data to be returned

    function BidirectionalChannel{I,O}() where {I,O}
        lock = ReentrantLock()
        cond_put = Threads.Condition(lock)
        cond_take = Threads.Condition(lock)
        cond_ret = Threads.Condition(lock)
        return new(cond_take, cond_put, cond_ret)
    end
end

Base.put!(c::BidirectionalChannel{I}, v) where {I} = put!(c, convert(I, v))
function Base.put!(c::BidirectionalChannel{I,O}, v::I) where {I,O}
    lock(c)
    try
        # wait for a slot to be available
        while isempty(c.cond_take)
            Base.wait(c.cond_put)
        end

        # pass a value to the consumer
        notify(c.cond_take, v, false, false)

        # wait for a return value to be produced
        Base.wait(c.cond_ret)::O
    finally
        unlock(c)
    end
end

function Base.take!(f::Base.Callable, c::BidirectionalChannel{I,O}) where {I,O}
    lock(c)
    try
        # notify the producer that we're ready to accept a value
        notify(c.cond_put, nothing, false, false)

        # receive a value from the producer
        v = Base.wait(c.cond_take)::I

        # return a value to the producer
        ret = f(v)::O
        notify(c.cond_ret, ret, false, false)
    finally
        unlock(c)
    end
end

Base.lock(c::BidirectionalChannel) = lock(c.cond_take)
Base.unlock(c::BidirectionalChannel) = unlock(c.cond_take)


## fast path

# before blocking on a separate thread, which has some overhead, busy-wait on a query of
# the object to synchronize. when this returns true, the object still has to be
# synchronized, but that won't block anymore.
function spinning_synchronization(f, obj)
    # fast path
    f(obj) && return true

    # minimize latency of short operations by busy-waiting,
    # initially without even yielding to other tasks
    spins = 0
    while spins < 256
        if spins < 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        f(obj) && return true
        spins += 1
    end

    return false
end


## slow path: synchronize on a separate thread

const MAX_SYNC_THREADS = 4
const sync_channels = Array{BidirectionalChannel{SyncObject,ze_result_t}}(undef, MAX_SYNC_THREADS)
const sync_channel_cursor = Threads.Atomic{UInt32}(1)
const sync_channel_lock = Base.ReentrantLock()

function synchronization_worker(data)
    i = Int(data)
    chan = sync_channels[i]

    while true
        # wait for work
        take!(gcsafe_synchronize, chan)
    end
end

@noinline function create_synchronization_worker(i)
    lock(sync_channel_lock) do
        # test and test-and-set
        if isassigned(sync_channels, i)
            return
        end

        # should be safe to assign before threads are running;
        # any user will just submit work that makes it block
        sync_channels[i] = BidirectionalChannel{SyncObject,ze_result_t}()

        # we don't know what the size of uv_thread_t is, so reserve enough space
        tid = Ref{NTuple{32, UInt8}}(ntuple(i -> 0, 32))

        cb = @cfunction(synchronization_worker, Cvoid, (Ptr{Cvoid},))
        err = @ccall uv_thread_create(tid::Ptr{Cvoid}, cb::Ptr{Cvoid}, Ptr{Cvoid}(i)::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_create", err)
        err = @ccall uv_thread_detach(tid::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_detach", err)
    end

    return
end

"""
    nonblocking_synchronize(list_or_queue)
    nonblocking_synchronize(event)

Wait for the work on an immediate command list or command queue to complete, or for an
event to be signaled, like [`synchronize`](@ref) or `wait`, but without blocking the Julia
scheduler: other tasks keep running while this one waits.
"""
function nonblocking_synchronize(obj::SyncObject)
    if spinning_synchronization(Base.isdone, obj)
        # done, so this doesn't block
        finish_synchronization(obj)
        return
    end

    # pick a worker channel: sticky per task, so repeated synchronizations from the
    # same task always hit the same, already running worker thread.
    tls = task_local_storage()
    i = get!(tls, :ZeSyncChannel) do
        mod1(Threads.atomic_add!(sync_channel_cursor, UInt32(1)), MAX_SYNC_THREADS)
    end::Int
    if !isassigned(sync_channels, i)
        create_synchronization_worker(i)
    end
    chan = @inbounds sync_channels[i]

    # submit the object to synchronize; unlike with regular channels, this `put!` blocks
    # until the worker has synchronized it and returned the result
    res = put!(chan, obj)
    res == RESULT_SUCCESS || throw_api_error(res)

    return
end
