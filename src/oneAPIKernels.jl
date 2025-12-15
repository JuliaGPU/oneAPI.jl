module oneAPIKernels

using ..oneAPI
using ..oneAPI: @device_override, SPIRVIntrinsics, method_table

import KernelAbstractions as KA

import StaticArrays

import Adapt


## Back-end Definition

export oneAPIBackend

struct oneAPIBackend <: KA.GPU
    prefer_blocks::Bool
    always_inline::Bool
end

oneAPIBackend(; prefer_blocks = false, always_inline = false) = oneAPIBackend(prefer_blocks, always_inline)

@inline KA.allocate(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims)
@inline KA.zeros(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = fill!(oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims), zero(T))
@inline KA.ones(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = fill!(oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims), one(T))

KA.get_backend(::oneArray) = oneAPIBackend()
# TODO should be non-blocking
KA.synchronize(::oneAPIBackend) = oneAPI.oneL0.synchronize()
KA.supports_float64(::oneAPIBackend) = false  # TODO: Check if this is device dependent
KA.supports_unified(::oneAPIBackend) = true

KA.functional(::oneAPIBackend) = oneAPI.functional()

Adapt.adapt_storage(::oneAPIBackend, a::AbstractArray) = Adapt.adapt(oneArray, a)
Adapt.adapt_storage(::oneAPIBackend, a::oneArray) = a
Adapt.adapt_storage(::KA.CPU, a::oneArray) = convert(Array, a)


## Memory Operations

function KA.copyto!(::oneAPIBackend, A, B)
    copyto!(A, B)
    # TODO: Address device to host copies in jl being synchronizing
end


## Device Operations

function KA.ndevices(::oneAPIBackend)
    return length(oneAPI.devices())
end

function KA.device(::oneAPIBackend)::Int
    dev = oneAPI.device()
    devs = oneAPI.devices()
    idx = findfirst(==(dev), devs)
    return idx === nothing ? 1 : idx
end

function KA.device!(backend::oneAPIBackend, id::Int)
    return oneAPI.device!(id)
end


## Kernel Launch

function KA.mkcontext(kernel::KA.Kernel{oneAPIBackend}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end
function KA.mkcontext(kernel::KA.Kernel{oneAPIBackend}, I, _ndrange, iterspace,
                      ::Dynamic) where Dynamic
    KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

function KA.launch_config(kernel::KA.Kernel{oneAPIBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        KA.partition(kernel, ndrange, ndrange)
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function (obj::KA.Kernel{oneAPIBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    backend = KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KA.workgroupsize(obj) <: KA.StaticSize
        # TODO: maxthreads
        # maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        # maxthreads = nothing
    end

    kernel = @oneapi launch = false always_inline = backend.always_inline obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        items = oneAPI.launch_configuration(kernel)

        if backend.prefer_blocks
            # Prefer blocks over threads:
            # Reducing the workgroup size (items) increases the number of workgroups (blocks).
            # We use a simple heuristic here since we lack full occupancy info (max_blocks) from launch_configuration.

            # If the total range is large enough, full workgroups are fine.
            # If the range is small, we might want to reduce 'items' to create more blocks to fill the GPU.
            # (Simplified logic compared to CUDA.jl which uses explicit occupancy calculators)
            total_items = prod(ndrange)
            if total_items < items * 16 # Heuristic factor
                # Force at least a few blocks if possible by reducing items per block
                target_blocks = 16 # Target at least 16 blocks
                items = max(1, min(items, cld(total_items, target_blocks)))
            end
        end

        workgroupsize = threads_to_workgroupsize(items, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    groups = length(KA.blocks(iterspace))
    items = length(KA.workitems(iterspace))

    if groups == 0
        return nothing
    end

    # Launch kernel
    kernel(ctx, args...; items, groups)

    return nothing
end


## Indexing Functions

@device_override @inline function KA.__index_Local_Linear(ctx)
    return get_local_id()
end

@device_override @inline function KA.__index_Group_Linear(ctx)
    return get_group_id()
end

@device_override @inline function KA.__index_Global_Linear(ctx)
    return get_global_id()
end

@device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[get_local_id()]
end

@device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[get_group_id()]
end

@device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(), get_local_id())
end

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(), get_local_id())
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end


## Shared and Scratch Memory

@device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = oneAPI.emit_localmemory(T, Val(prod(Dims)))
    oneDeviceArray(Dims, ptr)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end


## Synchronization and Printing

@device_override @inline function KA.__synchronize()
    barrier(0)
end

@device_override @inline function KA.__print(args...)
    oneAPI._print(args...)
end


## Other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::oneDeviceArray) = Base.Experimental.Const(a)

KA.argconvert(::KA.Kernel{oneAPIBackend}, arg) = kernel_convert(arg)

function KA.priority!(::oneAPIBackend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end

    priority_enum = if prio == :high
        oneAPI.oneL0.ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH
    elseif prio == :low
        oneAPI.oneL0.ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW
    else
        oneAPI.oneL0.ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    end

    ctx = oneAPI.context()
    dev = oneAPI.device()

    # Update the cached queue
    # We synchronize the current queue first to ensure safety
    current_queue = oneAPI.global_queue(ctx, dev)
    oneAPI.oneL0.synchronize(current_queue)

    # Replace the queue in task_local_storage
    # The key used by global_queue is (:ZeCommandQueue, ctx, dev)

    new_queue = oneAPI.oneL0.ZeCommandQueue(
        ctx, dev;
        flags = oneAPI.oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
        priority = priority_enum
    )

    task_local_storage((:ZeCommandQueue, ctx, dev), new_queue)

    return nothing
end

end
