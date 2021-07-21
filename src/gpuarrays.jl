# GPUArrays.jl interface


#
# Device functionality
#


## execution

struct oneArrayBackend <: AbstractGPUBackend end

struct oneKernelContext <: AbstractKernelContext end

@inline function GPUArrays.launch_heuristic(::oneArrayBackend, f::F, args::Vararg{Any,N};
                                            maximize_blocksize=false) where {F,N}
    # HACK: oneAPI needs the global size to compute a group size,
    #       so forward the kernel to launch_configuration
    # TODO: re-work this API in GPUArrays.jl
    kernel = @oneapi launch=false f(oneKernelContext(), args...)
    return kernel
end

@inline function GPUArrays.launch_configuration(::oneArrayBackend, kernel::AbstractKernel, total_threads::Int, elements_per_thread::Int=1)
    # TODO: respect maximize_blocksize? currently max groupsize seems to be 256
    items = suggest_groupsize(kernel.fun, total_threads).x
    groups = cld(total_threads, items)
    return (threads=items, blocks=groups, elements_per_thread=1)
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
GPUArrays.griddim(ctx::oneKernelContext) = oneAPI.get_num_groups(0)

# math

@inline GPUArrays.cos(ctx::oneKernelContext, x) = oneAPI.cos(x)
@inline GPUArrays.sin(ctx::oneKernelContext, x) = oneAPI.sin(x)
@inline GPUArrays.sqrt(ctx::oneKernelContext, x) = oneAPI.sqrt(x)
@inline GPUArrays.log(ctx::oneKernelContext, x) = oneAPI.log(x)

# memory

@inline function GPUArrays.LocalMemory(::oneKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = oneAPI.emit_localmemory(Val(id), T, Val(prod(dims)))
    oneDeviceArray(dims, LLVMPtr{T, onePI.AS.Local}(ptr))
end

# synchronization

@inline GPUArrays.synchronize_threads(::oneKernelContext) = oneAPI.barrier()



#
# Host abstractions
#

GPUArrays.backend(::Type{<:oneArray}) = oneArrayBackend()

const GLOBAL_RNGs = Dict{ZeDevice,GPUArrays.RNG}()
function GPUArrays.default_rng(::Type{<:oneArray})
    dev = device()
    get!(GLOBAL_RNGs, dev) do
        N = oneL0.compute_properties(dev).maxTotalGroupSize
        state = oneArray{NTuple{4, UInt32}}(undef, N)
        rng = GPUArrays.RNG(state)
        Random.seed!(rng)
        rng
    end
end
