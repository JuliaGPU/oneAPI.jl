using Distributed, Test, oneAPI

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, oneArray, xs...; kwargs...)

using Random


## entry point

function runtests(f, name)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    Test.TESTSET_PRINT_ENABLE[] = false

    try
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random, oneAPI)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        ex = quote
            Random.seed!(1)
            oneAPI.allowscalar(false)

            @timed @testset $"$name" begin
                $f()
            end
        end
        data = Core.eval(mod, ex)
        #data[1] is the testset

        # process results
        cpu_rss = Sys.maxrss()
        passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
            Test.get_test_counts(data[1])
        if data[1].anynonpass == false
            data = ((passes+c_passes,broken+c_broken),
                    data[2],
                    data[3],
                    data[4],
                    data[5])
        end
        res = vcat(collect(data), cpu_rss)

        res
    finally
        Test.TESTSET_PRINT_ENABLE[] = old_print_setting
    end
end


## auxiliary stuff

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

# Run some code on-device, returning captured standard output
macro on_device(ex)
    @gensym kernel
    esc(quote
        let
            function $kernel()
                $ex
                return
            end

            oneAPI.@sync @oneapi $kernel()
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

nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
