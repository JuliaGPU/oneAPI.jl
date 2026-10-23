import KernelAbstractions
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
])
Testsuite.testsuite(oneAPIBackend, "oneAPI", oneAPI, oneArray, oneDeviceArray; skip_tests)

using LinearAlgebra
using KernelAbstractions: @kernel, @index

@kernel function linear_index_kernel!(out)
    i = @index(Global, Linear)
    @inbounds out[i] = i
end

# When the work-group size does not divide a dimension of `ndrange`, KernelAbstractions pads
# the range: the hardware global id then differs from the linear index into `ndrange`
# (JuliaGPU/oneAPI.jl#509).
@testset "padded ndrange (#509)" begin
    @testset "@index(Global, Linear)" begin
        backend = oneAPIBackend()
        cases = [
            # ndrange => explicit work-group sizes (`nothing` lets the back-end choose)
            (513,) => (nothing, (32,), (256,)),
            (513, 3) => (nothing, (32, 1), (256, 2)),
            (3, 513) => (nothing, (1, 32), (2, 256)),
            (64, 4, 500) => (nothing, (256, 4, 1), (32, 1, 2), (7, 3, 5)),
            (64, 4, 500, 3) => (nothing, (64, 4, 1, 1), (16, 1, 7, 2)),
        ]
        for (ndrange, workgroupsizes) in cases, workgroupsize in workgroupsizes
            out = oneAPI.zeros(Int, prod(ndrange))
            linear_index_kernel!(backend)(out; ndrange, workgroupsize)
            KernelAbstractions.synchronize(backend)
            @test Array(out) == 1:prod(ndrange)
        end
    end

    @testset "getindex" begin
        x = randn(Float32, 513, 3)
        dx = oneArray(x)
        @test Array(dx[:, 1:2]) == x[:, 1:2]

        x = rand(Float32, 64, 4, 500, 3)
        dx = oneArray(x)
        @test Array(dx[:, 2, :, :]) == x[:, 2, :, :]
        @test Array(dx[2, :, :, :]) == x[2, :, :, :]
        @test Array(dx[:, :, 2, :]) == x[:, :, 2, :]
        @test Array(dx[:, :, :, 2]) == x[:, :, :, 2]
    end

    @testset "permutedims" begin
        x = rand(Float32, 64, 4, 500, 3)
        dx = oneArray(x)
        @test Array(permutedims(dx, (3, 1, 4, 2))) == permutedims(x, (3, 1, 4, 2))
        @test Array(permutedims(dx, (2, 1, 3, 4))) == permutedims(x, (2, 1, 3, 4))
    end

    @testset "rmul!" begin
        x = rand(ComplexF32, 64, 1000, 16)
        dx = oneArray(x)
        rmul!(dx, 2.0f0)
        @test Array(dx) == x .* 2.0f0
    end
end
