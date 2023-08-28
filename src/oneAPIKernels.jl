module oneAPIKernels

import KernelAbstractions
import oneAPI
import oneAPI: oneL0, @device_override
import GPUCompiler

import UnsafeAtomicsLLVM

struct oneAPIBackend <: KernelAbstractions.GPU
end
export oneAPIBackend

KernelAbstractions.allocate(::oneAPIBackend, ::Type{T}, dims::Tuple) where T = oneAPI.oneArray{T}(undef, dims)
KernelAbstractions.zeros(::oneAPIBackend, ::Type{T}, dims::Tuple) where T = oneAPI.zeros(T, dims)
KernelAbstractions.ones(::oneAPIBackend, ::Type{T}, dims::Tuple) where T = oneAPI.ones(T, dims)

# Import through parent
import KernelAbstractions: StaticArrays, Adapt
import .StaticArrays: MArray

KernelAbstractions.get_backend(::oneAPI.oneArray) = oneAPIBackend()
# TODO should be non-blocking
KernelAbstractions.synchronize(::oneAPIBackend) = oneL0.synchronize()
KernelAbstractions.supports_float64(::oneAPIBackend) = false # TODO is this device dependent?

Adapt.adapt_storage(::oneAPIBackend, a::Array) = Adapt.adapt(oneAPI.oneArray, a)
Adapt.adapt_storage(::oneAPIBackend, a::oneAPI.oneArray) = a
Adapt.adapt_storage(::KernelAbstractions.CPU, a::oneAPI.oneArray) = convert(Array, a)

##
# copyto!
##


function KernelAbstractions.copyto!(::oneAPIBackend, A, B)
    copyto!(A, B)
    # TODO device to host copies in oneAPI.jl are synchronizing.
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{oneAPIBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KernelAbstractions.workgroupsize(kernel) <: DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        partition(kernel, ndrange, ndrange)
    else
        partition(kernel, ndrange, workgroupsize)
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

function (obj::Kernel{oneAPIBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)
    kernel = oneAPI.@oneapi launch=false obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        items = oneAPI.suggest_groupsize(kernel.fun, prod(ndrange)).x
        # XXX: the z dimension of the suggested group size is often non-zero. use this?
        workgroupsize = threads_to_workgroupsize(items, ndrange)
        iterspace, dynamic = partition(obj, ndrange, workgroupsize)
        ctx = mkcontext(obj, ndrange, iterspace)
    end

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    if nblocks == 0
        return nothing
    end

    # Launch kernel
    kernel(ctx, args...; items=threads, groups=nblocks)

    return nothing
end

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{oneAPIBackend}, _ndrange, iterspace)
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end
function mkcontext(kernel::Kernel{oneAPIBackend}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

@device_override @inline function __index_Local_Linear(ctx)
    return oneAPI.get_local_id(0)
end

@device_override @inline function __index_Group_Linear(ctx)
    return oneAPI.get_group_id(0)
end

@device_override @inline function __index_Global_Linear(ctx)
    I =  @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(0), oneAPI.get_local_id(0))
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@device_override @inline function __index_Local_Cartesian(ctx)
    @inbounds workitems(__iterspace(ctx))[oneAPI.get_local_id(0)]
end

@device_override @inline function __index_Group_Cartesian(ctx)
    @inbounds blocks(__iterspace(ctx))[oneAPI.get_group_id(0)]
end

@device_override @inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(0), oneAPI.get_local_id(0))
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(0), oneAPI.get_local_id(0))
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace
import KernelAbstractions: SharedMemory, Scratchpad, __synchronize, __size

###
# GPU implementation of shared memory
###
@device_override @inline function SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = oneAPI.emit_localmemory(T, Val(prod(Dims)))
    oneAPI.oneDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@device_override @inline function Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{__size(Dims), T}(undef)
end

@device_override @inline function __synchronize()
    oneAPI.barrier()
end

@device_override @inline function __print(args...)
    oneAPI._print(args...)
end

KernelAbstractions.argconvert(::Kernel{oneAPIBackend}, arg) = oneAPI.kernel_convert(arg)

end
