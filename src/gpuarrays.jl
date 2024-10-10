# GPUArrays.jl interface

import KernelAbstractions
import KernelAbstractions: Backend

#
# Device functionality
#


## execution

@inline function GPUArrays.launch_heuristic(::oneAPIBackend, obj::O, args::Vararg{Any,N};
                                            elements::Int, elements_per_thread::Int) where {O,N}
    ndrange = ceil(Int, elements / elements_per_thread)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange,
                                                                  nothing)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    kernel = @oneapi launch=false obj.f(ctx, args...)

    items = launch_configuration(kernel)
    # XXX: how many groups is a good number? the API doesn't tell us.
    #      measured on a low-end IGP, 32 blocks seems like a good sweet spot.
    #      note that this only matters for grid-stride kernels, like broadcast.
    return (threads=items, blocks=32)
end

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
