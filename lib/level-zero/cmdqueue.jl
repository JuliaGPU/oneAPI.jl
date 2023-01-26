# queue

export ZeCommandQueue, synchronize

mutable struct ZeCommandQueue
    handle::ze_command_queue_handle_t

    context::ZeContext
    device::ZeDevice
    ordinal::Int

    function ZeCommandQueue(ctx::ZeContext, dev::ZeDevice, ordinal=1, index=1;
                            flags=0,
                            mode::ze_command_queue_mode_t=ZE_COMMAND_QUEUE_MODE_DEFAULT,
                            priority::ze_command_queue_priority_t=ZE_COMMAND_QUEUE_PRIORITY_NORMAL)
        desc_ref = Ref(ze_command_queue_desc_t(;
            ordinal=ordinal-1, index=index-1, flags, mode, priority
        ))
        handle_ref = Ref{ze_command_queue_handle_t}()
        zeCommandQueueCreate(ctx, dev, desc_ref, handle_ref)
        obj = new(handle_ref[], ctx, dev, ordinal)
        finalizer(obj) do obj
            zeCommandQueueDestroy(obj)
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
