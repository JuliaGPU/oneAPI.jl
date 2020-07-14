# GPUArrays.jl interface


#
# Device functionality
#

## device properties

GPUArrays.threads(dev::ZeDevice) = compute_properties(dev).maxTotalGroupSize


## execution

struct oneArrayBackend <: AbstractGPUBackend end

struct oneKernelContext <: AbstractKernelContext end

function GPUArrays.gpu_call(::oneArrayBackend, f, args, total_threads::Int;
                            name::Union{String,Nothing})
    function configurator(kernel)
        items = suggest_groupsize(kernel.fun, total_threads).x
        groups = cld(total_threads, items)

        return (items=items, groups=groups)
    end

    @oneapi config=configurator name=name f(oneKernelContext(), args...)
end

function GPUArrays.gpu_call(::oneArrayBackend, f, args, threads::Int, blocks::Int;
                            name::Union{String,Nothing})
    @oneapi items=threads groups=blocks name=name f(oneKernelContext(), args...)
end


## on-device

# indexing

GPUArrays.blockidx(ctx::oneKernelContext) = oneAPI.get_group_id(0)
GPUArrays.blockdim(ctx::oneKernelContext) = oneAPI.get_local_size(0)
GPUArrays.threadidx(ctx::oneKernelContext) = oneAPI.get_local_id(0)
GPUArrays.griddim(ctx::oneKernelContext) = CUDA.get_num_groups(0)



#
# Host abstractions
#

GPUArrays.device(A::oneArray) = A.dev

GPUArrays.backend(::Type{<:oneArray}) = oneArrayBackend()

# TODO: ownership
# TODO: make this `dims::Dims{N}`
GPUArrays.unsafe_reinterpret(::Type{T}, A::oneArray, size::NTuple{N, Integer}) where {T, N} =
  oneArray{T,N}(A.buf, size, A.dev)
