# GPUArrays.jl interface

#
# Device functionality
#


## execution

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
