using Random

# one GPUArrays RNG per task and device, so that streams of random numbers are independent
function gpuarrays_rng()
    dev = device()
    rngs = get!(task_local_storage(), :oneAPI_GLOBAL_RNGs) do
        Dict{ZeDevice,GPUArrays.RNG{oneArray}}()
    end
    get!(() -> GPUArrays.RNG{oneArray}(), rngs, dev)
end

# GPUArrays in-place
Random.rand!(A::oneWrappedArray) = Random.rand!(gpuarrays_rng(), A)
Random.randn!(A::oneWrappedArray) = Random.randn!(gpuarrays_rng(), A)

# GPUArrays out-of-place
rand(T::Type, dims::Dims) = Random.rand!(oneArray{T}(undef, dims...))
randn(T::Type, dims::Dims; kwargs...) = Random.randn!(oneArray{T}(undef, dims...); kwargs...)

# support all dimension specifications
rand(T::Type, dim1::Integer, dims::Integer...) = Random.rand!(oneArray{T}(undef, dim1, dims...))
randn(T::Type, dim1::Integer, dims::Integer...; kwargs...) = Random.randn!(oneArray{T}(undef, dim1, dims...); kwargs...)

# untyped out-of-place
rand(dim1::Integer, dims::Integer...) = Random.rand!(oneArray{Float32}(undef, dim1, dims...))
randn(dim1::Integer, dims::Integer...; kwargs...) = Random.randn!(oneArray{Float32}(undef, dim1, dims...); kwargs...)

# seeding
seed!(seed=Base.rand(UInt64)) = Random.seed!(gpuarrays_rng(), seed)
