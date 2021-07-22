# GPUArrays.jl interface


#
# Device functionality
#


## execution

struct oneArrayBackend <: AbstractGPUBackend end

struct oneKernelContext <: AbstractKernelContext end

@inline function GPUArrays.launch_heuristic(::oneArrayBackend, f::F, args::Vararg{Any,N};
                                             elements::Int, elements_per_thread::Int) where {F,N}
    kernel = @oneapi launch=false f(oneKernelContext(), args...)

    items = suggest_groupsize(kernel.fun, elements).x
    groups = cld(elements, items)
    return (threads=items, blocks=groups)
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
