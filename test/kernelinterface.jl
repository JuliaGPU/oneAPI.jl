import KernelInterface
import KernelInterface as KI

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(oneAPIBackend(), oneArray)

function ki_subgroup_kernel(num, sizes, max, id, lane)
    l = KI.get_local_id()
    s = KI.get_local_size()
    i = l.x + (l.y - 1) * s.x
    @inbounds begin
        num[i] = KI.get_num_sub_groups()
        sizes[i] = KI.get_sub_group_size()
        max[i] = KI.get_max_sub_group_size()
        id[i] = KI.get_sub_group_id()
        lane[i] = KI.get_sub_group_local_id()
    end
    return
end

# KernelInterface leaves the formation of sub-groups unspecified; Intel GPUs form them from
# consecutive linear work-item indices
@testset "partial sub-groups" begin
    backend = oneAPIBackend()
    @test KI.supports_subgroups(backend)
    width = KI.sub_group_size(backend)

    # a (width+1)x2 work-group is made up of 3 sub-groups, the last one only partially filled
    workgroupsize = (width + 1, 2)
    n = prod(workgroupsize)
    num, sizes, max, id, lane = (oneArray{UInt32}(undef, n) for _ in 1:5)
    KI.@launch backend workgroupsize=workgroupsize ki_subgroup_kernel(num, sizes, max, id, lane)
    @test all(==(3), Array(num))
    @test all(==(width), Array(max))
    @test Array(sizes) == [i < 2width ? width : 2 for i in 0:n-1]
    @test Array(id) == [div(i, width) + 1 for i in 0:n-1]
    @test Array(lane) == [rem(i, width) + 1 for i in 0:n-1]

    # the width is fixed
    @test_throws ArgumentError KI.@launch backend launch=false sub_group_size=(width == 16 ? 8 : 16) ki_subgroup_kernel(num, sizes, max, id, lane)
end

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

@testset "versioninfo" begin
    @test occursin("oneAPI.jl", sprint(KI.versioninfo, oneAPIBackend()))
end
