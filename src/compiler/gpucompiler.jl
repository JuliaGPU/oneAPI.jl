const ci_cache = GPUCompiler.CodeCache()

struct oneAPICompilerParams <: AbstractCompilerParams end

oneAPICompilerJob = CompilerJob{SPIRVCompilerTarget,oneAPICompilerParams}

GPUCompiler.runtime_module(::oneAPICompilerJob) = oneAPI

# TODO: eagerly lower these using the translator API
GPUCompiler.isintrinsic(job::oneAPICompilerJob, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(fn)},
           job, fn) ||
    in(fn, opencl_builtins)

function GPUCompiler.finish_module!(job::oneAPICompilerJob, mod::LLVM.Module)
    invoke(GPUCompiler.finish_module!,
           Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(mod)},
           job, mod)
    ctx = LLVM.context(mod)

    # OpenCL 2.0
    push!(metadata(mod)["opencl.ocl.version"],
          MDNode([ConstantInt(Int32(2); ctx),
                  ConstantInt(Int32(0); ctx)]; ctx))

    # SPIR-V 1.5
    push!(metadata(mod)["opencl.spirv.version"],
          MDNode([ConstantInt(Int32(1); ctx),
                  ConstantInt(Int32(5); ctx)]; ctx))
end

GPUCompiler.ci_cache(::oneAPICompilerJob) = ci_cache

GPUCompiler.method_table(::oneAPICompilerJob) = method_table

