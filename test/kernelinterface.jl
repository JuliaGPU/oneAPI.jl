import KernelInterface
import KernelInterface as KI

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(oneAPIBackend(), oneArray)

function ki_fill!(A)
    i = KI.get_global_id().x
    if i <= length(A)
        @inbounds A[i] = i
    end
    return
end

@testset "launch keywords" begin
    A = oneAPI.zeros(Int, 4)
    kernel = KI.@launch oneAPIBackend() launch=false ki_fill!(A)

    # oneAPI's launch options are passed on
    kernel(A; ndrange=4, queue=oneAPI.global_stream(oneAPI.context(), oneAPI.device()))
    @test Array(A) == 1:4

    # but not ones that would override the launch geometry
    @test_throws ArgumentError kernel(A; ndrange=4, items=8)
    @test_throws ArgumentError kernel(A; ndrange=4, groups=2)
end
