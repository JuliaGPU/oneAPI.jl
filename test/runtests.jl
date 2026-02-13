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

init_worker_code = quote
    using oneAPI, Adapt

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

runtests(oneAPI, args; testsuite, init_code, init_worker_code)
