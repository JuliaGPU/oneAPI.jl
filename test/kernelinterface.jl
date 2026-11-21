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

function ki_dynamic_localmem!(out)
    n = KI.get_local_size().x
    i = KI.get_local_id().x
    static = KI.localmemory(Int32, 3)
    x = KI.dynamic_localmemory(Int32, n)
    @inbounds begin
        x[i] = i
        static[1] = n
    end
    KI.barrier()
    @inbounds if i == 1
        out[1] = x[n] + static[1]
        out[2] = KI.dynamic_localmemory_size()
        out[3] = reinterpret(UInt, pointer(x)) % 16
    end
    return
end

@testset "dynamic local memory" begin
    backend = oneAPIBackend()
    @test KI.supports_dynamic_localmemory(backend)
    max_local_memory = oneL0.compute_properties(oneAPI.device()).maxSharedLocalMemory

    out = oneAPI.zeros(Int, 3)
    kernel = KI.@launch backend launch=false ki_dynamic_localmem!(out)
    fun = kernel.kern.host.fun
    @test fun.dynamic_localmemory

    # the kernel's static local memory, and what aligning the block takes, come out of what
    # the device has
    static = oneL0.properties(fun).localMemSize
    @test static > 0
    limit = KI.max_dynamic_localmemory(kernel)
    @test limit == max_local_memory - static - oneAPI.DYNAMIC_LOCALMEMORY_ALIGNMENT

    # the size is the one that was asked for, up to the limit, and the block is aligned
    # whatever static local memory the kernel has besides it
    n = 8
    for localmem in (n * sizeof(Int32), n * sizeof(Int32) + 1, limit)
        fill!(out, -1)
        kernel(out; workgroupsize=n, localmem)
        @test Array(out) == [2n, localmem, 0]
    end
    @test_throws ArgumentError kernel(out; workgroupsize=n, localmem=limit + 1)

    # launches that set the size don't change what a kernel reports as static
    @test oneL0.properties(fun).localMemSize == static

    # kernels that don't use the block don't have the arguments for it, and ignore the size
    A = oneAPI.zeros(Int, 4)
    plain = KI.@launch backend launch=false ki_fill!(A)
    @test !plain.kern.host.fun.dynamic_localmemory
    @test oneL0.properties(plain.kern.host.fun).numKernelArgs + 2 ==
        oneL0.properties(KI.@launch(backend, launch=false, ki_dynamic_localmem!(A)).kern.host.fun).numKernelArgs
    plain(A; ndrange=4, localmem=64)
    @test Array(A) == 1:4

    # oneAPI's own launches take the size as well
    fill!(out, -1)
    @oneapi items=n localmem=64 ki_dynamic_localmem!(out)
    @test Array(out) == [2n, 64, 0]
    @test_throws ArgumentError @oneapi items=n localmem=-1 ki_dynamic_localmem!(out)
end

@testset "versioninfo" begin
    @test occursin("oneAPI.jl", sprint(KI.versioninfo, oneAPIBackend()))
end
