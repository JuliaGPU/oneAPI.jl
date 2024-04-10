# GPUArrays.jl interface

import KernelAbstractions
import KernelAbstractions: Backend

#
# Device functionality
#


## execution

struct oneArrayBackend <: Backend end

@inline function GPUArrays.launch_heuristic(::oneArrayBackend, f::F, args::Vararg{Any,N};
                                             elements::Int, elements_per_thread::Int) where {F,N}
    kernel = @oneapi launch=false f(oneKernelContext(), args...)

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
