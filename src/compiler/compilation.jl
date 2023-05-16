## gpucompiler interface implementation

struct oneAPICompilerParams <: AbstractCompilerParams end
const oneAPICompilerConfig = CompilerConfig{SPIRVCompilerTarget, oneAPICompilerParams}
const oneAPICompilerJob = CompilerJob{SPIRVCompilerTarget,oneAPICompilerParams}

GPUCompiler.runtime_module(::oneAPICompilerJob) = oneAPI

const ci_cache = GPUCompiler.CodeCache()
GPUCompiler.ci_cache(::oneAPICompilerJob) = ci_cache

GPUCompiler.method_table(::oneAPICompilerJob) = method_table

# filter out OpenCL built-ins
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


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per device
const _compiler_caches = Dict{ZeDevice, Dict{Any, Any}}()
function compiler_cache(ctx::ZeDevice)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any, Any}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, oneAPICompilerConfig}()
function compiler_config(dev; kwargs...)
    h = hash(dev, hash(kwargs))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    supports_fp16 = oneL0.module_properties(device()).fp16flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP16 == oneL0.ZE_DEVICE_MODULE_FLAG_FP16
    supports_fp64 = oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64

    # TODO: emit printf format strings in constant memory
    extensions = String["SPV_EXT_relaxed_printf_string_address_space"]

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; extensions, supports_fp16, supports_fp64, kwargs...)
    params = oneAPICompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # TODO: on 1.9, this actually creates a context. cache those.
    JuliaContext() do ctx
        compile(job, ctx)
    end
end
function compile(@nospecialize(job::CompilerJob), ctx)
    asm, meta = GPUCompiler.compile(:obj, job; ctx)

    (image=asm, entry=LLVM.name(meta.entry))
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    ctx = context()
    dev = device()
    mod = ZeModule(ctx, dev, compiled.image)
    kernels(mod)[compiled.entry]
end
