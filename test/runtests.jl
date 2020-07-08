using Test, oneAPI

using Random
Random.seed!(1)

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))

testf(f, xs...; kwargs...) = TestSuite.compare(f, oneArray, xs...; kwargs...)

@testset "oneAPI" begin

oneAPI.allowscalar(false)

include("level-zero.jl")
include("array.jl")

end
