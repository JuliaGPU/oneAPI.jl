import KernelInterface
import KernelInterface as KI

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

# synchronizing every submission makes a launch wait for the device, so the timing check that
# an event wait is ordered after earlier work can't distinguish a missing wait
skip_tests = oneL0.sync_each_submission() ? Set(["Events"]) : Set{String}()
Testsuite.testsuite(oneAPIBackend(), oneArray; skip_tests)

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
    # kernels that don't request a width are compiled for the first one offered
    width = first(KI.sub_group_sizes(backend))

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
end

@testset "requested sub-group sizes" begin
    backend = oneAPIBackend()
    sizes = KI.sub_group_sizes(backend)
    @test !isempty(sizes)
    @test all(<=(64), sizes)
    @test allunique(sizes)
    @test issubset(sizes, oneL0.compute_properties(device()).subGroupSizes)

    # every width offered is the one the kernel runs at
    for width in sizes
        n = 2 * width
        num, sg_sizes, max, id, lane = (oneArray{UInt32}(undef, n) for _ in 1:5)
        KI.@launch backend workgroupsize=n sub_group_size=width ki_subgroup_kernel(num, sg_sizes, max, id, lane)
        @test all(==(width), Array(max))
        @test Array(lane) == [rem(i, width) + 1 for i in 0:n-1]
    end

    # requesting the width that kernels are compiled for anyway reuses the kernel
    args = Tuple(oneArray{UInt32}(undef, 1) for _ in 1:5)
    default = KI.@launch backend launch=false ki_subgroup_kernel(args...)
    requested = KI.@launch backend launch=false sub_group_size=Int32(first(sizes)) ki_subgroup_kernel(args...)
    @test requested.kern.host.fun === default.kern.host.fun

    # widths the device doesn't offer are rejected
    for width in (1, 3, 128)
        @test_throws ArgumentError KI.@launch backend launch=false sub_group_size=width ki_subgroup_kernel(args...)
    end
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
