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

oneAPI.versioninfo()

oneAPI.allowscalar(false)

include("util.jl")

include("level-zero.jl")
include("execution.jl")
include("array.jl")

@testset "GPUArrays.jl" begin
TestSuite.test(oneArray)
end

include("examples.jl")

end
