import KernelAbstractions
import KernelAbstractions as KA
import KernelInterface as KI
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
    "Random", # oneAPI doesn't support Random's default RNG in kernels yet
    # these run kernels on KernelAbstractions' POCL-based CPU back-end
    "CPU synchronization",
    "fallback test: callable types",
])
Testsuite.testsuite(()->oneAPIBackend(), "oneAPI", oneAPI, oneArray, oneDeviceArray; skip_tests)

KA.@kernel function store_global_linear!(A)
    I = KA.@index(Global, Linear)
    @inbounds A[I] = I
end

KA.@kernel function store_last_index!(A)
    I = KA.@index(Global, Linear)
    if I == prod(KA.@ndrange())
        @inbounds A[1] = I
        @inbounds A[2] = KA.@index(Global, Cartesian)[2]
    end
end

@testset "launch configuration" begin
    backend = oneAPIBackend()
    function select(kernel, ndrange, workgroupsize=nothing)
        ndrange, workgroupsize, iterspace, _ = KA.launch_config(kernel, ndrange, workgroupsize)
        KA.select_launch(kernel, workgroupsize, iterspace)
    end

    # kernels are launched on an N-d grid, computing indices in 32 bits
    kernel = store_global_linear!(backend)
    @test select(kernel, (64, 32, 16)) === KA.NDLaunch{Int32}()
    @test select(kernel, (4, 4, 4, 4)) === KA.LinearLaunch{Int32}()

    # which doesn't need divisions to compute the index of a dynamic N-d range
    A = oneAPI.zeros(Int, 64, 32, 16)
    ir = sprint(io -> oneAPI.@device_code_llvm io=io kernel(A; ndrange=size(A)))
    @test !occursin(r"\b[us](div|rem) ", ir)
    @test Array(A) == LinearIndices(A)

    # tuning for more work-groups (the testsuite above uses the default)
    Testsuite.launch_testsuite(()->oneAPIBackend(; prefer_blocks=true), oneArray)

    # iteration spaces that don't fit 32 bits use 64-bit indices
    kernel = store_last_index!(backend)
    A = oneAPI.zeros(Int, 2)
    for (dims, launch) in (((2^16 + 1, 2^15), KA.NDLaunch{Int}()),
                           ((2^11 + 1, 2^10, 2^10, 1), KA.LinearLaunch{Int}()))
        @test select(kernel, dims) === launch
        kernel(A; ndrange=dims)
        @test Array(A) == [prod(dims), dims[2]]
    end
end

function ki_store_index!(A)
    i = KI.get_global_id().x
    if i <= length(A)
        @inbounds A[i] = i
    end
    return
end

@testset "tuning" begin
    # tuning receives the number of work-items; prefer_blocks launches more, smaller groups
    A = oneAPI.zeros(Int, 1024)
    kernel = KI.@launch oneAPIBackend() launch=false ki_store_index!(A)
    items = KI.launch_configuration(kernel; nitems=1024).workgroupsize
    @test items <= KI.max_work_group_size(kernel)
    kernel = KI.@launch oneAPIBackend(; prefer_blocks=true) launch=false ki_store_index!(A)
    fewer = KI.launch_configuration(kernel; nitems=1024).workgroupsize
    @test fewer < items
    # ... but not for a bound on the work-group size alone
    @test KI.launch_configuration(kernel; max_work_group_size=1024).workgroupsize == items
    KI.@launch oneAPIBackend(; prefer_blocks=true) ndrange=length(A) ki_store_index!(A)
    @test Array(A) == 1:1024
end
