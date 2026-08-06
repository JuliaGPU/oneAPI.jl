using ParallelTestRunner
using oneAPI

oneAPI.functional() || error("oneAPI.jl is not functional on this system")

@info "System information:\n" * sprint(io->oneAPI.versioninfo(io))

if Sys.islinux()
    @info "Using oneAPI support library at " * oneAPI.Support.liboneapi_support
end


# choose tests
testsuite = find_tests(@__DIR__)
## GPUArrays test suite
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
gpuarrays_testsuite = joinpath(gpuarrays_root, "test", "testsuite.jl")
include(gpuarrays_testsuite)
for name in keys(TestSuite.tests)
    testsuite["gpuarrays/$name"] = :(TestSuite.tests[$name](oneArray))
end

args = parse_args(ARGS)

# Optional: spread test workers across all available GPUs, one worker per device
# (round-robin), by pinning each worker *process* to a device with ZE_AFFINITY_MASK.
# `device()` is task-local and Malt runs each test in a fresh task, so a `device!` in
# `init_worker_code` would not stick — pinning the process via the driver is the robust
# way to make every task on a worker use the same GPU.
#
# Enabled with ONEAPI_TEST_SPREAD_GPUS=1. When unset (the default) every worker stays on
# the first device, which oversubscribes a single tile — useful for surfacing
# contention/oversubscription bugs.
const spread_gpus = lowercase(get(ENV, "ONEAPI_TEST_SPREAD_GPUS", "")) in ("1", "true", "yes")
worker_env = Vector{Pair{String, String}}()

# Workers share one device but cannot see each other's allocations, so give each an equal
# share of device memory: the pool's proactive GC then kicks in before the workers
# collectively exhaust the card (important on 8GB consumer GPUs).
njobs = something(args.jobs, Some(ParallelTestRunner.default_njobs()))[]
push!(worker_env, "ONEAPI_MEMORY_LIMIT" => "$(max(1, 100 ÷ njobs))%")

device_claim_code = :()
if spread_gpus
    ndev = length(oneAPI.devices())
    # shared, node-local directory used as an atomic round-robin counter (mkdir is atomic)
    devdir = mktempdir(; prefix = "oneapi_test_gpus_")
    push!(worker_env, "ONEAPI_TEST_DEVDIR" => devdir)
    push!(worker_env, "ONEAPI_TEST_NDEV" => string(ndev))
    @info "Spreading test workers across $ndev GPU(s) via ZE_AFFINITY_MASK (ONEAPI_TEST_SPREAD_GPUS=1)"
    # NOTE: runs on the worker as the very first thing, before `using oneAPI` — so the
    # Level Zero driver picks up ZE_AFFINITY_MASK at init and the process sees only its tile.
    device_claim_code = quote
        let dir = ENV["ONEAPI_TEST_DEVDIR"], ndev = parse(Int, ENV["ONEAPI_TEST_NDEV"])
            i = 0
            while true
                try
                    mkdir(joinpath(dir, string(i)))
                    break
                catch
                    i += 1
                end
            end
            ENV["ZE_AFFINITY_MASK"] = string(i % ndev)
        end
    end
end

init_worker_code = quote
    $device_claim_code
    using oneAPI, Adapt
    # BFloat16s.jl supplies the host-side numeric support (conversions, rand sampling)
    # that Core.BFloat16 lacks on its own; the testsuite's CPU reference path needs it
    # when BFloat16 is part of the tested element types.
    using BFloat16s

    import GPUArrays
    include($gpuarrays_testsuite)
    testf(f, xs...; kwargs...) = TestSuite.compare(f, oneArray, xs...; kwargs...)

    const eltypes = [Int16, Int32, Int64,
                    Complex{Int16}, Complex{Int32}, Complex{Int64},
                    Float16, Float32,
                    ComplexF32]

    const float16_supported = oneL0.module_properties(device()).fp16flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP16 == oneL0.ZE_DEVICE_MODULE_FLAG_FP16
    if float16_supported
        append!(eltypes, [#=Float16,=# ComplexF16])
    end
    const float64_supported = oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
    if float64_supported
        append!(eltypes, [Float64, ComplexF64])
    end
    @static if isdefined(Core, :BFloat16)
        # Exercising BFloat16 as a generic element type needs (1) hardware support,
        # (2) a non-LTS stack — the Aurora LTS SPIR-V stack (Khronos translator +
        # NEO/IGC) cannot codegen native `bfloat` in generic kernels: any kernel that
        # keeps a bfloat value (e.g. clamp!) fails with InvalidIRError, and declaring
        # SPV_KHR_bfloat16 crashes the LTS runtime — and (3) a working CPU reference
        # path, which requires more host-side BFloat16 support than BFloat16s.jl
        # currently implements (e.g. the div/rem family throws "rem not defined for
        # BFloat16"). Probe the operations the testsuite relies on, so the element
        # type enables itself automatically once BFloat16s.jl catches up. Device-side
        # BFloat16 (oneMKL gemm/gemv, the bfloat16.jl example) is tested regardless.
        bfloat16_host_support() = try
            x, y = BFloat16(3), BFloat16(2)
            mod(x, y); fld(x, y); rand(BFloat16)
            true
        catch
            false
        end
        const bfloat16_supported = oneAPI._device_supports_bfloat16() &&
            !oneL0.LTS[] && bfloat16_host_support()
        if bfloat16_supported
            push!(eltypes, Core.BFloat16)
        end
    else
        const bfloat16_supported = false
    end
    TestSuite.supported_eltypes(::Type{<:oneArray}) = eltypes


    const validation_layer = parse(Bool, get(ENV, "ZE_ENABLE_VALIDATION_LAYER", "false"))
    const parameter_validation = parse(Bool, get(ENV, "ZE_ENABLE_PARAMETER_VALIDATION", "false"))

    # NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
    macro grab_output(ex)
        quote
            mktemp() do fname, fout
                ret = nothing
                open(fname, "w") do fout
                    redirect_stdout(fout) do
                                                ret = $(esc(ex))
                    end
                end
                ret, read(fname, String)
            end
        end
    end

    # Run some code on-device
    macro on_device(ex...)
        code = ex[end]
        kwargs = ex[1:end-1]

        @gensym kernel
        esc(quote
            let
                function $kernel()
                    $code
                    return
                end

                oneAPI.@sync @oneapi $(kwargs...) $kernel()
            end
        end)
    end

    # helper function for sinking a value to prevent the callee from getting optimized away
    @inline sink(i::Int32) =
        Base.llvmcall("""%slot = alloca i32
                        store volatile i32 %0, i32* %slot
                        %value = load volatile i32, i32* %slot
                        ret i32 %value""", Int32, Tuple{Int32}, i)
    @inline sink(i::Int64) =
        Base.llvmcall("""%slot = alloca i64
                        store volatile i64 %0, i64* %slot
                        %value = load volatile i64, i64* %slot
                        ret i64 %value""", Int64, Tuple{Int64}, i)
end

init_code = quote
    using oneAPI, Adapt

    import ..TestSuite, ..testf
    import ..eltypes, ..float16_supported, ..float64_supported,
           ..validation_layer, ..parameter_validation,
           ..@grab_output, ..@on_device, ..sink
end

# On memory-pressured GPUs (e.g. 8GB cards) a failing test can leave the worker — or even
# the driver — in a state where every subsequent allocation fails, so recycle workers on
# failure and give failed files one exclusive retry on an otherwise-idle device.
# These options are not available in every ParallelTestRunner release, so detect them
# rather than gating on a version number.
runtests_kwargs = Set(Iterators.flatten(Base.kwarg_decl.(methods(ParallelTestRunner.runtests))))
failure_handling = if :recycle_on_failure in runtests_kwargs
    (; recycle_on_failure = true, retries = 1)
else
    (;)
end

runtests(oneAPI, args; testsuite, init_code, init_worker_code, env = worker_env,
         failure_handling...)
