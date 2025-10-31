module oneAPIKernels

using ..oneAPI
using ..oneAPI: @device_override, SPIRVIntrinsics, method_table

import KernelAbstractions as KA

import StaticArrays

import Adapt


## Back-end Definition

export oneAPIBackend

struct oneAPIBackend <: KA.GPU
end

KA.allocate(::oneAPIBackend, ::Type{T}, dims::Tuple) where T = oneArray{T}(undef, dims)
KA.zeros(::oneAPIBackend, ::Type{T}, dims::Tuple) where T = oneAPI.zeros(T, dims)
KA.ones(::oneAPIBackend, ::Type{T}, dims::Tuple) where T = oneAPI.ones(T, dims)

KA.get_backend(::oneArray) = oneAPIBackend()
# TODO should be non-blocking
KA.synchronize(::oneAPIBackend) = oneL0.synchronize()
KA.supports_float64(::oneAPIBackend) = false  # TODO: Check if this is device dependent

Adapt.adapt_storage(::oneAPIBackend, a::Array) = Adapt.adapt(oneArray, a)
Adapt.adapt_storage(::oneAPIBackend, a::oneArray) = a
Adapt.adapt_storage(::KA.CPU, a::oneArray) = convert(Array, a)


## Memory Operations

function KA.copyto!(::oneAPIBackend, A, B)
    copyto!(A, B)
    # TODO: Address device to host copies in jl being synchronizing
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
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel = @oneapi launch=false obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        items = oneAPI.launch_configuration(kernel)
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

KA.argconvert(::KA.Kernel{oneAPIBackend}, arg) = kernel_convert(arg)

end
