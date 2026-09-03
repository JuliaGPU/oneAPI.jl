module KernelAbstractionsExt

using ..oneAPI
using ..oneAPI: @device_override, SPIRVIntrinsics, method_table

import KernelAbstractions as KA

import StaticArrays

import Adapt


Adapt.adapt_storage(::KA.CPU, a::oneArray) = convert(Array, a)

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
        # (clamped to 1, since an empty ndrange cannot serve as a workgroup size)
        KA.partition(kernel, ndrange, max.(ndrange, 1))
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = max(1, min(div(threads, total), n))
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

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(), get_local_id())
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end


## Scratch Memory

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

## Other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::oneDeviceArray) = Base.Experimental.Const(a)

KA.argconvert(::KA.Kernel{oneAPIBackend}, arg) = kernel_convert(arg)

end
