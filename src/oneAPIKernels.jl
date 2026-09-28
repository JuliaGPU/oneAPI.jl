module oneAPIInterface

using ..oneAPI
using ..oneAPI: @device_override, SPIRVIntrinsics, method_table, kernel_convert, zefunction

import KernelInterface as KI

import StaticArrays

## Back-end Definition

export oneAPIBackend

struct oneAPIBackend <: KI.Backend
    prefer_blocks::Bool
    always_inline::Bool
end

KI.versioninfo(io::IO, ::oneAPIBackend) = oneAPI.versioninfo(io)

oneAPIBackend(; prefer_blocks = false, always_inline = false) = oneAPIBackend(prefer_blocks, always_inline)

@inline KI.allocate(::oneAPIBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = oneArray{T, length(dims), unified ? oneAPI.oneL0.SharedBuffer : oneAPI.oneL0.DeviceBuffer}(undef, dims)

KI.get_backend(::oneArray) = oneAPIBackend()
KI.synchronize(::oneAPIBackend) = oneAPI.oneL0.synchronize()
KI.supports_float64(::oneAPIBackend) = device_limits().supports_float64
KI.supports_unified(::oneAPIBackend) = true
KI.supports_atomics(::oneAPIBackend) = true

KI.functional(::oneAPIBackend) = oneAPI.functional()

# sparse arrays (oneMKL is only available on Linux)
@static if Sys.islinux()
    KI.get_backend(::oneAPI.oneMKL.oneAbstractSparseMatrix) = oneAPIBackend()
end


## Memory Operations

function KI.copyto!(::oneAPIBackend, A, B)
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    copyto!(A, B)
    # TODO: Address device to host copies in jl being synchronizing
    return A
end

KI.unsafe_free!(A::oneArray) = oneAPI.unsafe_free!(A)


## Device Operations

function KI.ndevices(::oneAPIBackend)
    return length(oneAPI.devices())
end

function device_index(dev)::Int
    devs = oneAPI.devices()
    idx = findfirst(==(dev), devs)
    return idx === nothing ? 1 : idx
end
KI.device(::oneAPIBackend)::Int = device_index(oneAPI.device())
KI.device(::oneAPIBackend, A::oneArray)::Int = device_index(oneAPI.device(A))

function KI.device!(backend::oneAPIBackend, id::Int)
    oneAPI.device!(id)
    return
end


## Kernel Launch

KI.argconvert(::oneAPIBackend, arg) = kernel_convert(arg)

function KI.kernel_function(backend::oneAPIBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    # compile for the sub-group width that `KI.sub_group_size` promises
    sub_group_size = KI.sub_group_size(backend)
    kern = if sub_group_size > 0
        zefunction(f, tt; name, backend.always_inline, sub_group_size, kwargs...)
    else
        zefunction(f, tt; name, backend.always_inline, kwargs...)
    end
    KI.Kernel{oneAPIBackend, typeof(kern)}(backend, kern)
end

function KI.launch(obj::KI.Kernel{oneAPIBackend}, groups::Dims{3}, items::Dims{3}, args::Vararg{Any, N}; kwargs...) where {N}
    # kernels are compiled for a device, and launched on the task's stream of the active one
    obj.kern.fun.mod.device == device() ||
        throw(ArgumentError("Cannot launch a kernel compiled for another device than the active one"))
    obj.kern(args...; items, groups, kwargs...)
    return
end

function KI.max_work_group_size(kernel::KI.Kernel{oneAPIBackend})::Int
    fun = kernel.kern.fun
    max_group_size = oneAPI.oneL0.max_group_size(fun)
    # without the MAX_GROUP_SIZE extension, the device limit is all we know
    return coalesce(max_group_size, device_limits(fun.mod.device).max_work_group_size)
end
function KI.launch_configuration(kernel::KI.Kernel{oneAPIBackend}; max_work_group_size::Integer = typemax(Int))
    group_size = oneAPI.launch_configuration(kernel.kern)
    return (; workgroupsize = Int(min(group_size, max_work_group_size)))
end
# querying the device allocates, so cache what every launch needs
const DeviceLimits = @NamedTuple{
    max_work_group_size::Int, max_work_group_dims::NTuple{3, Int}, max_num_groups::NTuple{3, Int},
    sub_group_size::Int, supports_float16::Bool, supports_float64::Bool,
}
function device_limits(dev::oneAPI.oneL0.ZeDevice = device())
    limits = get!(task_local_storage(), :oneAPIDeviceLimits) do
        Dict{oneAPI.oneL0.ZeDevice, DeviceLimits}()
    end::Dict{oneAPI.oneL0.ZeDevice, DeviceLimits}
    get!(limits, dev) do
        props = oneAPI.oneL0.compute_properties(dev)
        module_props = oneAPI.oneL0.module_properties(dev)
        # the sub-group width that `kernel_function` compiles for: the width `@oneapi` defaults
        # to if the device supports it, and 0 if the device has no sub-groups
        sg_sizes = props.subGroupSizes
        sub_group_size = 32 in sg_sizes ? 32 : maximum(sg_sizes; init = 0)
        (; max_work_group_size = props.maxTotalGroupSize,
           max_work_group_dims = (props.maxGroupSizeX, props.maxGroupSizeY, props.maxGroupSizeZ),
           max_num_groups = (props.maxGroupCountX, props.maxGroupCountY, props.maxGroupCountZ),
           sub_group_size,
           supports_float16 = module_props.flags & oneAPI.oneL0.ZE_DEVICE_MODULE_FLAG_FP16 != 0,
           supports_float64 = module_props.flags & oneAPI.oneL0.ZE_DEVICE_MODULE_FLAG_FP64 != 0)
    end
end
KI.max_work_group_size(::oneAPIBackend)::Int = device_limits().max_work_group_size
KI.max_work_group_dims(::oneAPIBackend)::NTuple{3, Int} = device_limits().max_work_group_dims
KI.max_num_groups(::oneAPIBackend)::NTuple{3, Int} = device_limits().max_num_groups
KI.sub_group_size(::oneAPIBackend)::Int = device_limits().sub_group_size
function KI.multiprocessor_count(::oneAPIBackend)::Int
    oneAPI.oneL0.properties(device()).numSlices
end

KI.supports_subgroups(::oneAPIBackend) = device_limits().sub_group_size > 0
function KI.supports_shuffle(::oneAPIBackend, ::Type{T}) where {T}
    T in SPIRVIntrinsics.gentypes || return false
    T === Float64 && return device_limits().supports_float64
    T === Float16 && return device_limits().supports_float16
    return true
end

## Indexing Functions
## COV_EXCL_START

# computed with `% T`, which unlike `T(x)` has no error path

@device_override @inline function KI.get_local_id(::Type{T}) where {T}
    return (; x = get_local_id(1) % T, y = get_local_id(2) % T, z = get_local_id(3) % T)
end

@device_override @inline function KI.get_group_id(::Type{T}) where {T}
    return (; x = get_group_id(1) % T, y = get_group_id(2) % T, z = get_group_id(3) % T)
end

@device_override @inline function KI.get_local_size(::Type{T}) where {T}
    return (; x = get_local_size(1) % T, y = get_local_size(2) % T, z = get_local_size(3) % T)
end

@device_override @inline function KI.get_num_groups(::Type{T}) where {T}
    return (; x = get_num_groups(1) % T, y = get_num_groups(2) % T, z = get_num_groups(3) % T)
end

@device_override KI.get_sub_group_size(::Type{T}) where {T} = get_sub_group_size() % T

@device_override KI.get_max_sub_group_size(::Type{T}) where {T} = get_max_sub_group_size() % T

@device_override KI.get_num_sub_groups(::Type{T}) where {T} = get_num_sub_groups() % T

@device_override KI.get_sub_group_id(::Type{T}) where {T} = get_sub_group_id() % T

@device_override KI.get_sub_group_local_id(::Type{T}) where {T} = get_sub_group_local_id() % T

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
