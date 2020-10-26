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

include("util.jl")

include("level-zero.jl")
include("execution.jl")
include("array.jl")

@testset "GPUArrays.jl" begin

tests = [
  "base",
  "broadcasting",
  "constructors",
  "conversions",
  "indexing multidimensional",
  "indexing scalar",
  "input output",
  "interface",
  "iterator constructors",
# "linear algebra",             # 128-bit mulwide
  "mapreduce essentials",
# "mapreduce derivatives",      # 128-bit mulwide
  "math",
  "random",
# "uniformscaling",             # gpu_malloc missing
  "value constructors",
]

for test in tests
    @testset "$test" begin
        TestSuite.tests[test](oneArray)
    end
end

end

include("examples.jl")

end
