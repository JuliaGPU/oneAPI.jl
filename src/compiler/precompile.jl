using PrecompileTools: @setup_workload, @compile_workload

# Warm up the GPUCompiler -> SPIR-V pipeline during precompilation so the first real
# `zefunction` call is cheap. This doesn't need a GPU: SPIR-V codegen is device-independent
# (the target only carries version/extension/capability knobs), and the workload never
# launches anything. It is gated on the SPIR-V tool for the selected codegen path being
# available so the package still precompiles on platforms/toolchains without it.
#
# The workload mirrors the runtime backend choice (see `_compiler_config`): the LTS stack
# compiles through the Khronos translator, the rolling stack through the LLVM SPIR-V
# back-end. `oneL0.__init__` has not run at precompile time, so resolve ONEAPI_LTS
# directly from the environment with the same parser.
const _precompile_lts = oneL0.parse_env_bool("ONEAPI_LTS", false)
if _precompile_lts ? SPIRV_LLVM_Translator_jll.is_available() :
        SPIRV_LLVM_Backend_jll.is_available()
    @compile_workload begin
        let
            function _precompile_kernel(a)
                @inbounds a[1] += 1.0f0
                return
            end

            # Build a device-independent compiler config. `_compiler_config` normally derives
            # these knobs from the device; here we use conservative, portable defaults (the
            # workload only exercises the pipeline, it does not target a specific device).
            # Cover both Float64 settings: each gives inference a different method table
            # type (see `method_tables`).
            for supports_fp64 in (true, false)
                target = SPIRVCompilerTarget(;
                    backend = (_precompile_lts ? :khronos : :llvm),
                    extensions = "", supports_fp16 = true,
                    supports_fp64, supports_bfloat16 = false, driver = :intel,
                )
                params = oneAPICompilerParams()
                config = CompilerConfig(target, params; kernel=true, name=nothing,
                                        always_inline=false)

                tt = Tuple{oneDeviceArray{Float32,1,AS.CrossWorkgroup}}
                source = methodinstance(typeof(_precompile_kernel), tt)
                job = CompilerJob(source, config)

                # On Julia < 1.12, driving GPU compilation during precompilation can leak
                # foreign MethodInstances into host native compilation; only run the full
                # compile on 1.12+.
                @static if VERSION >= v"1.12-"
                    # Exercise the launch-side cache path and serialize its portable SPIR-V
                    # image.
                    compile_or_lookup(job)
                end
            end
        end
    end
end

# Exercise the runtime when a GPU can be used while precompiling: allocating, copying,
# launching and displaying all take seconds to compile on first use otherwise.
function precompile_runtime_kernel(a)
    @inbounds a[1] = 1
    return
end
function precompile_runtime()
    a = oneArray(Float32[1, 2, 3])
    Array(a)
    show(IOContext(IOBuffer(), :limit => true), MIME"text/plain"(), a)

    # these compile kernels, which has the same issue on Julia < 1.12 as above
    @static if VERSION >= v"1.12-"
        @oneapi precompile_runtime_kernel(a)
        Array(a .+ 1f0)
        Array(oneArray([1]) .+ 1)
        Array(rand(Float32, 2))
    end
    synchronize()
    return
end
# streams are task-local, so run the workload in a task of its own, whose storage can be
# collected along with the streams before the state they are registered in is reset. (from
# a function, as top-level code keeps the values it computes, like the task, alive)
function run_precompile_runtime()
    wait(@async precompile_runtime())
    return
end

@setup_workload begin
    if can_precompile()
        @compile_workload run_precompile_runtime()

        # the state the workload created refers to driver objects of this process. first
        # collect the streams, whose finalizers need that state: free the workload's arrays,
        # then drop the pools, whose freed chunks keep the streams that used them alive.
        Base.@lock stream_registry_lock empty!(stream_registry)
        Base.@lock pending_release_lock empty!(pending_release)
        Base.@lock pending_streams_lock empty!(pending_streams)
        GC.gc(true)
        trim_pools!()
        with_pool_lock(() -> empty!(pools))
        GC.gc(true)

        Base.@lock global_contexts_lock empty!(global_contexts)
        Base.@lock batching_support_lock empty!(batching_support)
        empty!(stagings)
        empty!(staged_devices)
        empty!(eltype_support_cache)
        empty!(free_report_slots)
        empty!(_compiler_configs)
        empty!(oneL0.extension_properties_cache)
        Base.@atomic oneAPIKernels.device_limits_cache.limits = oneAPIKernels.DeviceLimitsDict()
        empty!(oneAPIKernels.pending_events)
        _allocated_bytes[] = 0
        _total_mem_cache[] = 0
        _memory_limit_cache[] = -1
    end
end

# whatever the probe found, keep it and the driver state out of the package image
precompile_probe[] = nothing
oneL0.functional[] = false
