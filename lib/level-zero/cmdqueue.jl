# queue

export ZeCommandQueue, synchronize

# Bound on how long the LTS drain-before-destroy finalizer waits for in-flight work before
# giving up and leaking the queue (see the finalizer below). 10 s comfortably covers any
# kernel still legitimately running at finalization while keeping a never-signaled event
# from hanging GC or process exit forever.
const FINALIZER_SYNC_TIMEOUT_NS = UInt64(10_000_000_000)

mutable struct ZeCommandQueue
    handle::ze_command_queue_handle_t

    context::ZeContext
    device::ZeDevice
    ordinal::Int

    # high-water mark of per-thread spill (bytes) among kernels submitted to this queue,
    # maintained by the scratch hedge (see `scratch_hedge!` in src/compiler/execution.jl)
    scratch_hwm::Int

    function ZeCommandQueue(ctx::ZeContext, dev::ZeDevice, ordinal=1, index=1;
                            flags=0,
                            mode::ze_command_queue_mode_t=ZE_COMMAND_QUEUE_MODE_DEFAULT,
                            priority::ze_command_queue_priority_t=ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
        desc_ref = Ref(ze_command_queue_desc_t(;
            ordinal=ordinal-1, index=index-1, flags, mode, priority
        ))
        handle_ref = Ref{ze_command_queue_handle_t}()
        zeCommandQueueCreate(ctx, dev, desc_ref, handle_ref)
        obj = new(handle_ref[], ctx, dev, ordinal, 0)
        finalizer(obj) do obj
            if LTS[]
                # the queue may still have work in flight (nothing requires a task to
                # synchronize before dying), and zeCommandQueueDestroy does not wait for
                # it: on the LTS NEO stack the still-running work then faults as soon as
                # a referenced allocation is freed, getting the context banned. drain the
                # queue first; unchecked, as sync on a banned context returns an error.
                #
                # Bounded wait, not typemax(UInt64): event-gated work whose event is never
                # signaled (a task that submits work with a wait event and dies before
                # signaling it) would make an infinite wait hang the finalizer forever, and
                # with it GC and process exit. On timeout, leak the queue deliberately —
                # destroying it now would trigger the very fault+ban this drain prevents.
                if unchecked_zeCommandQueueSynchronize(obj, FINALIZER_SYNC_TIMEOUT_NS) == RESULT_NOT_READY
                    @warn "Leaking a command queue still busy after $(FINALIZER_SYNC_TIMEOUT_NS ÷ 1_000_000_000)s to avoid blocking finalization (event-gated work whose event was never signaled?)" maxlog = 1
                    return
                end
            end
            zeCommandQueueDestroy(obj)
            if LTS[]
                # mark the queue as destroyed: it can still be weakly reachable (e.g. from
                # the stream registry used by `synchronize_all_streams`), and synchronizing a
                # destroyed handle crashes in the driver.
                obj.handle = ze_command_queue_handle_t(C_NULL)
            end
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_command_queue_handle_t}, queue::ZeCommandQueue) = queue.handle

Base.:(==)(a::ZeCommandQueue, b::ZeCommandQueue) = a.handle == b.handle
Base.hash(e::ZeCommandQueue, h::UInt) = hash(e.handle, h)

synchronize(queue::ZeCommandQueue, timeout::Number=typemax(UInt64)) =
    zeCommandQueueSynchronize(queue, timeout)


## groups

export command_queue_groups, compute_groups

struct ZeCommandQueueGroups
    device::ZeDevice
end

command_queue_groups(dev::ZeDevice) = ZeCommandQueueGroups(dev)

Base.eltype(::ZeCommandQueueGroups) = ZeCommandQueueGroup

function Base.iterate(groups::ZeCommandQueueGroups, i=1)
    i >= length(groups) + 1 ? nothing : (ZeCommandQueueGroup(groups, i), i+1)
end

Base.length(groups::ZeCommandQueueGroups) = length(properties(groups))

function properties(groups::ZeCommandQueueGroups)
    count_ref = Ref{UInt32}(0)
    zeDeviceGetCommandQueueGroupProperties(groups.device, count_ref, C_NULL)

    all_props = fill(ze_command_queue_group_properties_t(), count_ref[])
    zeDeviceGetCommandQueueGroupProperties(groups.device, count_ref, all_props)

    return [(flags=props.flags,
             maxMemoryFillPatternSize=UInt(props.maxMemoryFillPatternSize),
             numQueues=Int(props.numQueues),
             ) for props in all_props[1:count_ref[]]]
end

Base.IteratorSize(::ZeCommandQueueGroups) = Base.HasLength()

struct ZeCommandQueueGroup
    groups::ZeCommandQueueGroups
    ordinal::Int
end

properties(group::ZeCommandQueueGroup) = properties(group.groups)[group.ordinal]

# short-hands
compute_groups(dev::ZeDevice) = filter(collect(command_queue_groups(dev))) do group
    properties(group).flags & oneL0.ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE != 0
end
