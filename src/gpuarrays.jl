# GPUArrays.jl interface

function GPUArrays.default_rng(::Type{<:oneArray})
    dev = device()
    rngs = get!(task_local_storage(), :oneAPI_GLOBAL_RNGs) do
        Dict{ZeDevice,GPUArrays.RNG}()
    end
    get!(rngs, dev) do
        N = oneL0.compute_properties(dev).maxTotalGroupSize
        state = oneArray{NTuple{4, UInt32}}(undef, N)
        rng = GPUArrays.RNG(state)
        Random.seed!(rng)
        rng
    end
end
