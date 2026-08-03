using PrecompileTools: @compile_workload

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
            target = SPIRVCompilerTarget(;
                backend = (_precompile_lts ? :khronos : :llvm),
                extensions = "", supports_fp16 = true,
                supports_fp64 = true, supports_bfloat16 = false,
            )
            params = oneAPICompilerParams()
            config = CompilerConfig(target, params; kernel=true, name=nothing,
                                    always_inline=false)

            tt = Tuple{oneDeviceArray{Float32,1,AS.CrossWorkgroup}}
            source = methodinstance(typeof(_precompile_kernel), tt)
            job = CompilerJob(source, config)

            # On Julia < 1.12, driving GPU compilation during precompilation can leak foreign
            # MethodInstances into host native compilation; only run the full compile on 1.12+.
            @static if VERSION >= v"1.12-"
                # Exercise the launch-side cache path and serialize its portable SPIR-V image.
                compile_or_lookup(job)
            end
        end
    end
end
