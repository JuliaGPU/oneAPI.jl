## gpucompiler interface implementation

struct oneAPICompilerParams <: AbstractCompilerParams end
const oneAPICompilerConfig = CompilerConfig{SPIRVCompilerTarget, oneAPICompilerParams}
const oneAPICompilerJob = CompilerJob{SPIRVCompilerTarget,oneAPICompilerParams}

"""
    oneAPIResults

Cached compilation results for a oneAPI kernel job, managed by
`GPUCompiler.cached_results`. Fields are populated through the compile pipeline:
`image` (SPIR-V bytes) + `entry` after codegen, and `kernels` after the session-local
link onto a Level Zero module. The first two are session-portable (cached through
precompilation); `kernels` is session-local and never populated during precompilation.
`image === nothing` identifies a job that has not been compiled yet.

`kernels` is a small linear cache of `(ZeContext, ZeDevice, ZeKernel)` tuples. The cache
partition already covers everything that affects codegen via `GPUCompiler.cache_owner`, so
the only runtime-visible dimensions left are the Level Zero context and device that own the
linked `ZeKernel` (a `ZeModule` is built for a specific `(context, device)` pair). A linear
scan with `===`/`==` is fastest in the common case (n=1) and stays cheap for the rare
workload that bounces between a handful of contexts or devices.
"""
mutable struct oneAPIResults
    image::Union{Nothing, Vector{UInt8}}                    # SPIR-V binary
    entry::Union{Nothing, String}
    kernels::Vector{Tuple{ZeContext, ZeDevice, ZeKernel}}   # session-local; linear-scanned
    oneAPIResults() = new(nothing, nothing, Tuple{ZeContext, ZeDevice, ZeKernel}[])
end

GPUCompiler.runtime_module(::oneAPICompilerJob) = oneAPI

GPUCompiler.method_table_view(job::oneAPICompilerJob) =
    GPUCompiler.StackedMethodTable(job.world, method_table, SPIRVIntrinsics.method_table)

# filter out OpenCL built-ins
# TODO: eagerly lower these using the translator API
GPUCompiler.isintrinsic(job::oneAPICompilerJob, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(fn)},
           job, fn) ||
    in(fn, known_intrinsics) ||
    contains(fn, "__spirv_")

function GPUCompiler.finish_module!(job::oneAPICompilerJob, mod::LLVM.Module,
                                    entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_module!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(mod), typeof(entry)},
                   job, mod, entry)

    # OpenCL 2.0
    push!(metadata(mod)["opencl.ocl.version"],
          MDNode([ConstantInt(Int32(2)),
                  ConstantInt(Int32(0))]))

    # SPIR-V 1.5
    push!(metadata(mod)["opencl.spirv.version"],
          MDNode([ConstantInt(Int32(1)),
                  ConstantInt(Int32(5))]))

    return entry
end

# finish_ir! runs later in the pipeline, after optimizations that create nested insertvalue
function GPUCompiler.finish_ir!(job::oneAPICompilerJob, mod::LLVM.Module,
                                entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_ir!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(mod), typeof(entry)},
                   job, mod, entry)

    # FIX: Flatten nested insertvalue instructions to work around SPIR-V bug
    # See: https://github.com/JuliaGPU/oneAPI.jl/issues/259
    # Intel's SPIR-V runtime has a bug where OpCompositeInsert with nested
    # indices (e.g., "1 0") corrupts adjacent struct fields.
    flatten_nested_insertvalue!(mod)

    return entry
end

# Flatten nested insertvalue instructions
# This works around a bug in Intel's SPIR-V runtime where OpCompositeInsert
# with nested array indices corrupts adjacent struct fields.
function flatten_nested_insertvalue!(mod::LLVM.Module)
    changed = false
    count = 0

    for f in functions(mod)
        isempty(blocks(f)) && continue

        for bb in blocks(f)
            # Collect instructions to process (can't modify while iterating)
            to_process = LLVM.Instruction[]

            for inst in instructions(bb)
                # Check if this is an insertvalue with nested indices
                if LLVM.API.LLVMGetInstructionOpcode(inst) == LLVM.API.LLVMInsertValue
                    num_indices = LLVM.API.LLVMGetNumIndices(inst)
                    if num_indices > 1
                        push!(to_process, inst)
                    end
                end
            end

            # Flatten each nested insertvalue
            for inst in to_process
                try
                    flatten_insert!(inst)
                    changed = true
                    count += 1
                catch e
                    @warn "Failed to flatten nested insertvalue" exception=(e, catch_backtrace())
                end
            end
        end
    end

    return changed
end

function flatten_insert!(inst::LLVM.Instruction)
    # Transform: insertvalue %base, %val, i, j, k...
    # Into:      extractvalue %base, i
    #            insertvalue %extracted, %val, j, k...
    #            insertvalue %base, %modified, i

    composite = LLVM.operands(inst)[1]
    value = LLVM.operands(inst)[2]

    num_indices = LLVM.API.LLVMGetNumIndices(inst)
    idx_ptr = LLVM.API.LLVMGetIndices(inst)
    indices = unsafe_wrap(Array, idx_ptr, num_indices)

    builder = LLVM.IRBuilder()
    LLVM.position!(builder, inst)

    # Strategy: Recursively extract and insert for each nesting level
    # For insertvalue %base, %val, i, j, k
    # Do: %tmp1 = extractvalue %base, i
    #     %tmp2 = extractvalue %tmp1, j
    #     %tmp3 = insertvalue %tmp2, %val, k
    #     %tmp4 = insertvalue %tmp1, %tmp3, j
    #     %result = insertvalue %base, %tmp4, i

    # But that's complex. Simpler approach for 2-3 levels:
    # Just do one level of flattening at a time
    first_idx = indices[1]
    rest_indices = indices[2:end]

    # Extract the first level
    extracted = LLVM.extract_value!(builder, composite, first_idx)

    # Now insert into the extracted value using remaining indices
    # The LLVM IR builder will handle this correctly
    inserted = extracted
    if length(rest_indices) == 1
        # Simple case: just one more level
        inserted = LLVM.insert_value!(builder, extracted, value, rest_indices[1])
    else
        # Multiple levels: need to extract down, insert, then insert back up
        # For now, recursively extract to the deepest level
        temps = [extracted]
        for i in 1:(length(rest_indices)-1)
            temp = LLVM.extract_value!(builder, temps[end], rest_indices[i])
            push!(temps, temp)
        end

        # Insert the value at the deepest level
        inserted = LLVM.insert_value!(builder, temps[end], value, rest_indices[end])

        # Insert back up the chain
        for i in (length(rest_indices)-1):-1:1
            inserted = LLVM.insert_value!(builder, temps[i], inserted, rest_indices[i])
        end
    end

    # Insert the modified structure back into the original
    result = LLVM.insert_value!(builder, composite, inserted, first_idx)

    LLVM.replace_uses!(inst, result)
    LLVM.API.LLVMInstructionEraseFromParent(inst)
    LLVM.dispose(builder)
end


## compiler implementation (configure, compile, and link)

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, oneAPICompilerConfig}()
function compiler_config(dev; kwargs...)
    h = hash(dev.driver, hash(dev, hash(kwargs)))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    properties = oneL0.module_properties(dev)
    supports_fp16 = properties.fp16flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP16 == oneL0.ZE_DEVICE_MODULE_FLAG_FP16
    supports_fp64 = properties.fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64

    # SPIR-V codegen path. The Aurora LTS NEO/IGC runtime only accepts SPIR-V from the
    # Khronos translator and needs these extensions declared explicitly; the rolling stack
    # uses the LLVM SPIR-V back-end (which handles the extensions itself). GPUCompiler picks
    # the tool from the target's `backend` field and loads the JLL lazily, so both can be
    # listed as deps and the choice is made here at compile time.
    if oneL0.LTS[]
        backend = :khronos
        # TODO: emit printf format strings in constant memory
        extensions = String[
            "SPV_EXT_relaxed_printf_string_address_space",
            "SPV_EXT_shader_atomic_float_add"
        ]
    else
        backend = :llvm
        extensions = String[]
    end
    extensions_str = join(map(ext -> "+$ext", extensions), ",")

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; backend, extensions=extensions_str, supports_fp16, supports_fp64, kwargs...)
    params = oneAPICompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# run inference + LLVM codegen + SPIR-V emission. returns `(image, entry)`, both
# session-portable so they survive precompilation when stored on a cached `CodeInstance`.
function compile_to_obj(@nospecialize(job::CompilerJob))
    # TODO: on 1.9, this actually creates a context. cache those.
    asm, meta = JuliaContext() do ctx
        GPUCompiler.compile(:obj, job)
    end

    (image=asm, entry=LLVM.name(meta.entry))
end

# link the SPIR-V bytes into a session-local `ZeKernel` on the given context and device.
function link_kernel(image::Vector{UInt8}, entry::String, ctx::ZeContext, dev::ZeDevice)
    mod = ZeModule(ctx, dev, image)
    kernels(mod)[entry]
end
