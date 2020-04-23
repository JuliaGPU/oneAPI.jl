## target

struct oneAPICompilerTarget <: CompositeCompilerTarget
    parent::SPIRVCompilerTarget

    oneAPICompilerTarget() = new(SPIRVCompilerTarget())
end

Base.parent(target::oneAPICompilerTarget) = target.parent

# TODO: eagerly lower these using the translator API
GPUCompiler.isintrinsic(target::oneAPICompilerTarget, fn::String) =
    GPUCompiler.isintrinsic(target.parent, fn) || in(fn, opencl_builtins)

GPUCompiler.runtime_module(target::oneAPICompilerTarget) = oneAPI


## job

struct oneAPICompilerJob <: CompositeCompilerJob
    parent::SPIRVCompilerJob
end

oneAPICompilerJob(target::AbstractCompilerTarget, source::FunctionSpec) =
    oneAPICompilerJob(SPIRVCompilerJob(target, source))

Base.similar(job::oneAPICompilerJob, source::FunctionSpec) =
    oneAPICompilerJob(similar(job.parent, source))

Base.parent(job::oneAPICompilerJob) = job.parent

function GPUCompiler.process_module!(job::oneAPICompilerJob, mod::LLVM.Module)
    GPUCompiler.process_module!(job.parent, mod)

    # OpenCL 2.0
    push!(metadata(mod), "opencl.ocl.version",
         MDNode([ConstantInt(Int32(2), JuliaContext()),
                 ConstantInt(Int32(0), JuliaContext())]))

    # SPIR-V 1.5
    push!(metadata(mod), "opencl.spirv.version",
         MDNode([ConstantInt(Int32(1), JuliaContext()),
                 ConstantInt(Int32(5), JuliaContext())]))
end

GPUCompiler.mcgen(job::oneAPICompilerJob, mod::LLVM.Module, f::LLVM.Function, format) =
    GPUCompiler.mcgen(job.parent, mod, f, format)
