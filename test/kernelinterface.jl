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

@testset "versioninfo" begin
    @test occursin("oneAPI.jl", sprint(KI.versioninfo, oneAPIBackend()))
end

using oneAPI: SPIRVIntrinsics
using .SPIRVIntrinsics: UnsafeAtomics

# Kernels that use any of these can tell the sub-group width they run at, so they have to be
# compiled for the width that `KI.sub_group_size` promises
const sub_group_ops = [
    "get_sub_group_size"            => (x, p) -> x + KI.get_sub_group_size(),
    "get_max_sub_group_size"        => (x, p) -> x + KI.get_max_sub_group_size(),
    "get_num_sub_groups"            => (x, p) -> x + KI.get_num_sub_groups(),
    "get_sub_group_id"              => (x, p) -> x + KI.get_sub_group_id(),
    "get_sub_group_local_id"        => (x, p) -> x + KI.get_sub_group_local_id(),
    "sub_group_barrier()"           => (x, p) -> (KI.sub_group_barrier(); x),
    "sub_group_barrier(width)"      => (x, p) -> (KI.sub_group_barrier(8); x),
    "shfl"                          => (x, p) -> KI.shfl(x, 1),
    "shfl_down"                     => (x, p) -> KI.shfl_down(x, 1),
    "shfl_up"                       => (x, p) -> KI.shfl_up(x, 1),
    "shfl_xor"                      => (x, p) -> KI.shfl_xor(x, 1),
    "sub_group_any"                 => (x, p) -> x + KI.sub_group_any(x > UInt32(3)),
    "sub_group_all"                 => (x, p) -> x + KI.sub_group_all(x > UInt32(3)),
    "sub_group_ballot"              => (x, p) -> x + KI.sub_group_ballot(x > UInt32(3)) % UInt32,
    "sub_group_any(width)"          => (x, p) -> x + KI.sub_group_any(x > UInt32(3), 8),
    "sub_group_ballot(width)"       => (x, p) -> x + KI.sub_group_ballot(x > UInt32(3), 8) % UInt32,
    "sub_group_reduce(+)"           => (x, p) -> KI.sub_group_reduce(+, x),
    "sub_group_reduce(max)"         => (x, p) -> KI.sub_group_reduce(max, x),
    "sub_group_reduce(*)"           => (x, p) -> KI.sub_group_reduce(*, x),
    "sub_group_reduce(+, width)"    => (x, p) -> KI.sub_group_reduce(+, x, 8),
    "sub_group_scan(+)"             => (x, p) -> KI.sub_group_scan(+, x),
    "sub_group_scan(+, width)"      => (x, p) -> KI.sub_group_scan(+, x, 8),
    "sub_group_exclusive_scan(+)"   => (x, p) -> KI.sub_group_exclusive_scan(+, x, UInt32(0)),
    "sub-group fence"               => (x, p) -> (SPIRVIntrinsics.atomic_work_item_fence(
                                           SPIRVIntrinsics.GLOBAL_MEM_FENCE,
                                           SPIRVIntrinsics.memory_order_acq_rel,
                                           SPIRVIntrinsics.memory_scope_sub_group); x),
    "sub-group atomic"              => (x, p) -> first(UnsafeAtomics.Internal.llvm_modify!(
                                           p, +, x, Val(:monotonic), Val(:subgroup), Val(false),
                                           Val(4), Val(()))),
]
# while these behave the same at any width
const work_group_ops = [
    "arithmetic"                    => (x, p) -> x * UInt32(3) + UInt32(1),
    "work-item indices"             => (x, p) -> x + KI.get_local_id().x + KI.get_group_id().x,
    "barrier"                       => (x, p) -> (KI.barrier(); x),
    "atomic"                        => (x, p) -> SPIRVIntrinsics.atomic_add!(p, x),
]

function ki_op_kernel(op, out)
    i = KI.get_global_id().x
    @inbounds out[i] = op(UInt32(i), pointer(out))
    return
end

@testset "sub-group width detection" begin
    width = KI.sub_group_size(oneAPIBackend())
    DevVec = oneDeviceVector{UInt32, oneAPI.AS.CrossWorkgroup}
    function observes(op)
        # as compiled for the promised width, which adds the `SubgroupDispatch` capability
        kwargs = width > 0 ? (; sub_group_size = width) : (;)
        config = oneAPI.compiler_config(device(); kernel = true, kwargs...)
        job = oneAPI.CompilerJob(oneAPI.methodinstance(typeof(ki_op_kernel),
                                                       Tuple{typeof(op), DevVec}), config)
        return oneAPI.observes_sub_group_width(oneAPI.compile_to_obj(job).image)
    end
    @testset "$name" for (name, op) in sub_group_ops
        @test observes(op)
    end
    @testset "$name" for (name, op) in work_group_ops
        @test !observes(op)
    end

    # anything that can't be parsed counts
    spirv(instructions...) = collect(reinterpret(UInt8, UInt32[0x07230203, 0x00010500, 0, 16, 0,
                                                                instructions...]))
    @test !oneAPI.observes_sub_group_width(spirv(0x00020011, 6))    # OpCapability Kernel
    @test oneAPI.observes_sub_group_width(spirv(0x00020011, 18))    # OpCapability Groups
    @test oneAPI.observes_sub_group_width(spirv(0x00010011))        # OpCapability without one
    @test oneAPI.observes_sub_group_width(spirv(0x00050011, 6))     # truncated instruction
    @test oneAPI.observes_sub_group_width(spirv(0x00000011))        # zero word count
    @test oneAPI.observes_sub_group_width(spirv(0x00020011, 6)[1:end-1])
    @test oneAPI.observes_sub_group_width(UInt8[])
end

@inline spill_step(acc::NTuple{N}, x) where {N} =
    ntuple(k -> muladd(acc[k], x, acc[mod1(k + 1, N)]), Val(N))

# keeps `N` values live across the loop, which spills at SIMD32 on Xe-LP and Xe-HPG for N=32
function ki_spill_kernel(out, widths, x, ::Val{N}, ::Val{observe}) where {N, observe}
    i = KI.get_global_id().x
    acc = ntuple(k -> @inbounds(x[k]) * i, Val(N))
    for j in eachindex(x)
        acc = spill_step(acc, @inbounds(x[j]))
    end
    @inbounds out[i] = sum(acc)
    if observe
        @inbounds widths[i] = KI.get_max_sub_group_size()
    end
    return
end

@testset "narrower sub-groups for spilling kernels" begin
    backend = oneAPIBackend()
    width = KI.sub_group_size(backend)
    x = oneArray(rand(Float32, 64) ./ 2)
    out = oneArray{Float32}(undef, 256)
    widths = oneArray{Int}(undef, 256)
    simd(kernel::oneAPI.HostKernel) = oneL0.properties(kernel.fun).maxSubgroupSize
    simd(kernel::KI.Kernel) = simd(kernel.kern.host)
    spill(kernel::oneAPI.HostKernel) = oneL0.spill_mem_size(kernel.fun)
    launch(kernel, args...) = kernel(args...; ndrange = length(out))

    # the narrower kernel isn't used if it would be limited to smaller work-groups
    @test oneAPI.narrow_kernel_fits(1024, 1024)
    @test oneAPI.narrow_kernel_fits(512, 1024)
    @test !oneAPI.narrow_kernel_fits(1024, 512)
    @test !oneAPI.narrow_kernel_fits(missing, 1024)
    @test !oneAPI.narrow_kernel_fits(1024, missing)

    if width > 16 && 16 in oneL0.compute_properties(device()).subGroupSizes
        args = (out, widths, x, Val(32), Val(false))
        pinned = @oneapi launch=false sub_group_size=width ki_spill_kernel(args...)
        if spill(pinned) > 0
            # a kernel that can't tell its width is compiled for 16 when it spills
            kernel = KI.@launch backend launch=false ki_spill_kernel(args...)
            @test simd(kernel) == 16
            launch(kernel, args...)
            want = Array(out)
            pinned(args...; items = 32, groups = length(out) ÷ 32)
            @test Array(out) ≈ want
            # which doesn't affect kernels compiled for an explicit width
            @test simd(@oneapi launch=false sub_group_size=width ki_spill_kernel(args...)) == width

            # a kernel that can tell keeps the promised width, although it spills too
            args = (out, widths, x, Val(32), Val(true))
            @test spill(@oneapi launch=false sub_group_size=width ki_spill_kernel(args...)) > 0
            kernel = KI.@launch backend launch=false ki_spill_kernel(args...)
            @test simd(kernel) == width
            launch(kernel, args...)
            @test all(==(width), Array(widths))
            @test Array(out) ≈ want
        else
            @test_skip "ki_spill_kernel doesn't spill at a sub-group size of $width"
        end

        # as does a kernel that doesn't spill
        args = (out, widths, x, Val(2), Val(false))
        @test spill(@oneapi launch=false sub_group_size=width ki_spill_kernel(args...)) == 0
        @test simd(KI.@launch backend launch=false ki_spill_kernel(args...)) == width
    else
        @test_skip "the device has no sub-group size of 16 narrower than $width"
    end
end
