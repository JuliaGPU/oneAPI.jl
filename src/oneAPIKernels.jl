module oneAPIInterface

using ..oneAPI
using ..oneAPI: @device_override, SPIRVIntrinsics, method_table, kernel_convert, zefunction

import KernelInterface as KI

import StaticArrays

import Adapt


## Back-end Definition

export oneAPIBackend

struct oneAPIBackend <: KA.GPU
    prefer_blocks::Bool
    always_inline::Bool
end

KI.versioninfo(io::IO, ::oneAPIBackend) = oneAPI.versioninfo(io)

oneAPIBackend(; prefer_blocks = false, always_inline = false) = oneAPIBackend(prefer_blocks, always_inline)

@inline KI.allocate(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims)
@inline KI.zeros(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = fill!(oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims), zero(T))
@inline KI.ones(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = fill!(oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims), one(T))

KI.get_backend(::oneArray) = oneAPIBackend()
# TODO should be non-blocking
KI.synchronize(::oneAPIBackend) = oneAPI.oneL0.synchronize()
KI.supports_float64(::oneAPIBackend) = false  # TODO: Check if this is device dependent
KI.supports_unified(::oneAPIBackend) = true

KI.functional(::oneAPIBackend) = oneAPI.functional()

Adapt.adapt_storage(::oneAPIBackend, a::AbstractArray) = Adapt.adapt(oneArray, a)
Adapt.adapt_storage(::oneAPIBackend, a::oneArray) = a


## Memory Operations

function KI.copyto!(::oneAPIBackend, A, B)
    copyto!(A, B)
    # TODO: Address device to host copies in jl being synchronizing
end


## Device Operations

function KI.ndevices(::oneAPIBackend)
    return length(oneAPI.devices())
end

function KI.device(::oneAPIBackend)::Int
    dev = oneAPI.device()
    devs = oneAPI.devices()
    idx = findfirst(==(dev), devs)
    return idx === nothing ? 1 : idx
end

function KI.device!(backend::oneAPIBackend, id::Int)
    return oneAPI.device!(id)
end


## Kernel Launch

KI.argconvert(::oneAPIBackend, arg) = kernel_convert(arg)

function KI.kernel_function(::oneAPIBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    kern = zefunction(f, tt; name, kwargs...)
    KI.Kernel{oneAPIBackend, typeof(kern)}(oneAPIBackend(), kern)
end

function (obj::KI.Kernel{oneAPIBackend})(args...; numworkgroups=(), workgroupsize=(), ndrange=(), max_work_group_size=typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)
    prod(ndrange) == 0 && return nothing

    numworkgroups, workgroupsize = KI.auto_launch_sizes(obj, numworkgroups, workgroupsize, ndrange, max_work_group_size)
    items = (workgroupsize..., ntuple(_ -> 1, 3 - length(workgroupsize))...)
    groups = (numworkgroups..., ntuple(_ -> 1, 3 - length(numworkgroups))...)

    obj.kern(args...; items, groups)
    return nothing
end

function (obj::KI.Kernel{oneAPIBackend})(args...; numworkgroups = 1, workgroupsize = 1)
    KI.check_launch_args(numworkgroups, workgroupsize)

    items = (workgroupsize..., ntuple(_ -> 1, 3 - length(workgroupsize))...)

    groups = (numworkgroups..., ntuple(_ -> 1, 3 - length(numworkgroups))...)

    obj.kern(args...; items, groups)
    return nothing
end

function KI.kernel_max_work_group_size(kernel::KI.Kernel{<:oneAPIBackend}; max_work_items::Int=typemax(Int))::Int
    group_size = oneAPI.launch_configuration(kernel.kern)
    Int(min(group_size, max_work_items))
end
function KI.max_work_group_size(::oneAPIBackend)::Int
    oneAPI.oneL0.compute_properties(device()).maxTotalGroupSize
end
function KI.sub_group_size(::oneAPIBackend)::Int
    sg_sizes = oneAPI.oneL0.compute_properties(device()).subGroupSizes
    if 32 in sg_sizes
        return 32
    elseif 64 in sg_sizes
        return 64
    elseif 16 in sg_sizes
        return 16
    else
        return 1
    end
end
function KI.multiprocessor_count(::oneAPIBackend)::Int
    oneAPI.oneL0.properties(device()).numSlices
end

function KI.shfl_down_types(::oneAPIBackend)
    res = copy(SPIRVIntrinsics.gentypes)

    res = setdiff(res, [Float64])

    return res
end

## Indexing Functions
## COV_EXCL_START
@device_override @inline function KI.get_local_id()
    return (; x = Int(get_local_id(1)), y = Int(get_local_id(2)), z = Int(get_local_id(3)))
end

@device_override @inline function KI.get_group_id()
    return (; x = Int(get_group_id(1)), y = Int(get_group_id(2)), z = Int(get_group_id(3)))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int(get_global_id(1)), y = Int(get_global_id(2)), z = Int(get_global_id(3)))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int(get_local_size(1)), y = Int(get_local_size(2)), z = Int(get_local_size(3)))
end

@device_override @inline function KI.get_num_groups()
    return (; x = Int(get_num_groups(1)), y = Int(get_num_groups(2)), z = Int(get_num_groups(3)))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(get_global_size(1)), y = Int(get_global_size(2)), z = Int(get_global_size(3)))
end

@device_override KI.get_sub_group_size() = get_sub_group_size()

@device_override KI.get_max_sub_group_size() = get_max_sub_group_size()

@device_override KI.get_num_sub_groups() = get_num_sub_groups()

@device_override KI.get_sub_group_id() = get_sub_group_id()

@device_override KI.get_sub_group_local_id() = get_sub_group_local_id()

## Shared and Scratch Memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ptr = oneAPI.emit_localmemory(T, Val(prod(Dims)))
    oneDeviceArray(Dims, ptr)
end

## Synchronization and Printing

@device_override @inline function KI.barrier()
    # Fence both local and global memory across the workgroup barrier, matching CUDA
    # `__syncthreads` semantics. `barrier(0)` lowers to `OpControlBarrier` with
    # `SequentiallyConsistent` but WITHOUT any storage-class bit, which the SPIR-V spec
    # treats as ordering *no* memory — so shared-local or global writes are not guaranteed
    # visible to other work-items after the barrier. `LOCAL_MEM_FENCE | GLOBAL_MEM_FENCE`
    # ORs in the WorkgroupMemory/CrossWorkgroupMemory fence bits.
    barrier(SPIRVIntrinsics.LOCAL_MEM_FENCE | SPIRVIntrinsics.GLOBAL_MEM_FENCE)
end

@device_override @inline function KI.sub_group_barrier()
    sub_group_barrier(SPIRVIntrinsics.LOCAL_MEM_FENCE | SPIRVIntrinsics.GLOBAL_MEM_FENCE)
end

@device_override function KI.shfl_down(val::T, offset::Integer) where T
    sub_group_shuffle(val, get_sub_group_local_id() + offset)
end

@device_override @inline function KI._print(args...)
    oneAPI._print(args...)
end

## COV_EXCL_STOP

## Other

function KI.priority!(::oneAPIBackend, prio::Symbol)
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

    # drain the task's current stream before swapping it out, so operations submitted
    # to the new stream cannot overtake in-flight work on the old one
    oneAPI.oneL0.synchronize(oneAPI.global_stream(ctx, dev))

    # Replace the stream in task_local_storage. `create_stream` registers the
    # replacement so `synchronize_all_streams`/`release` can drain it before freeing a
    # buffer whose in-flight work it references; otherwise all work after `priority!`
    # runs on an unregistered stream and a freed buffer can be reused while its kernel
    # is still running (use-after-free → banned context on the LTS NEO stack). The old
    # stream stays registered until its task dies, like replaced queues before it.
    new_stream = oneAPI.create_stream(ctx, dev, priority_enum)
    task_local_storage((:oneStream, ctx, dev), new_stream)

    # the cached SYCL queue wraps the old stream's companion queue; drop it so the next
    # oneMKL call recreates it against the new stream (the old one was just drained)
    delete!(task_local_storage(), (:SYCLQueue, ctx, dev))

    return nothing
end

end
