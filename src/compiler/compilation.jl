## gpucompiler interface implementation

struct oneAPICompilerParams <: AbstractCompilerParams end
const oneAPICompilerConfig = CompilerConfig{SPIRVCompilerTarget, oneAPICompilerParams}
const oneAPICompilerJob = CompilerJob{SPIRVCompilerTarget,oneAPICompilerParams}

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

    # When the device supports BFloat16 but the SPIR-V runtime doesn't accept
    # SPV_KHR_bfloat16, lower all bfloat types to i16 so the translator can
    # handle the module without the extension.
    if @static(isdefined(Core, :BFloat16) && isdefined(LLVM, :BFloatType)) &&
       _device_supports_bfloat16() && !_driver_supports_bfloat16_spirv()
        lower_bfloat_to_i16!(mod)
    end

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


# Lower bfloat types to i16 in the LLVM IR.
# This is needed when the device supports BFloat16 but the SPIR-V runtime/translator
# doesn't support SPV_KHR_bfloat16. Since sizeof(bfloat)==sizeof(i16)==2, the memory
# layout is identical.
#
# TODO: Julia 1.12's Core.BFloat16 is a bare primitive (no Float32 conversion, no
# arithmetic), so fptrunc/fpext instructions never appear in practice. If Julia adds
# BFloat16 conversion methods in the future, this pass should be extended to handle
# fptrunc float→bfloat and fpext bfloat→float, either via inline RNE bit manipulation
# or calls to __devicelib_ConvertFToBF16INTEL / __devicelib_ConvertBF16ToFINTEL.
function lower_bfloat_to_i16!(mod::LLVM.Module)
    T_bf16 = LLVM.BFloatType()
    T_i16 = LLVM.Int16Type()

    # Phase 1: Eliminate all bitcasts between i16 and bfloat (same bit width).
    eliminate_bf16_bitcasts!(mod, T_bf16, T_i16)

    # Phase 2: Replace remaining bfloat GEPs, loads, and stores with i16 equivalents.
    for f in functions(mod)
        isempty(blocks(f)) && continue
        for bb in blocks(f)
            to_replace = LLVM.Instruction[]
            for inst in instructions(bb)
                opcode = LLVM.API.LLVMGetInstructionOpcode(inst)
                if opcode == LLVM.API.LLVMGetElementPtr
                    src_ty = LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    src_ty == T_bf16 && push!(to_replace, inst)
                elseif opcode == LLVM.API.LLVMLoad
                    value_type(inst) == T_bf16 && push!(to_replace, inst)
                elseif opcode == LLVM.API.LLVMStore
                    value_type(LLVM.operands(inst)[1]) == T_bf16 && push!(to_replace, inst)
                end
            end

            for inst in to_replace
                opcode = LLVM.API.LLVMGetInstructionOpcode(inst)
                builder = LLVM.IRBuilder()
                LLVM.position!(builder, inst)

                if opcode == LLVM.API.LLVMGetElementPtr
                    ptr = LLVM.operands(inst)[1]
                    indices = LLVM.Value[LLVM.operands(inst)[i] for i in 2:length(LLVM.operands(inst))]
                    new_gep = if LLVM.API.LLVMIsInBounds(inst) != 0
                        LLVM.inbounds_gep!(builder, T_i16, ptr, indices)
                    else
                        LLVM.gep!(builder, T_i16, ptr, indices)
                    end
                    LLVM.replace_uses!(inst, new_gep)
                elseif opcode == LLVM.API.LLVMLoad
                    ptr = LLVM.operands(inst)[1]
                    new_load = LLVM.load!(builder, T_i16, ptr)
                    LLVM.replace_uses!(inst, new_load)
                elseif opcode == LLVM.API.LLVMStore
                    val = LLVM.operands(inst)[1]
                    ptr = LLVM.operands(inst)[2]
                    LLVM.store!(builder, val, ptr)
                end

                LLVM.API.LLVMInstructionEraseFromParent(inst)
                LLVM.dispose(builder)
            end
        end
    end

    return true
end

# Iteratively eliminate bitcasts between i16 and bfloat (same bit representation).
function eliminate_bf16_bitcasts!(mod::LLVM.Module, T_bf16::LLVMType, T_i16::LLVMType)
    changed = true
    while changed
        changed = false
        for f in functions(mod)
            isempty(blocks(f)) && continue
            for bb in blocks(f)
                to_delete = LLVM.Instruction[]
                for inst in instructions(bb)
                    if LLVM.API.LLVMGetInstructionOpcode(inst) == LLVM.API.LLVMBitCast
                        src = LLVM.operands(inst)[1]
                        src_ty = value_type(src)
                        dst_ty = value_type(inst)
                        if (src_ty == T_i16 && dst_ty == T_bf16) ||
                           (src_ty == T_bf16 && dst_ty == T_i16) ||
                           (src_ty == dst_ty)
                            LLVM.replace_uses!(inst, src)
                            push!(to_delete, inst)
                            changed = true
                        end
                    end
                end
                for inst in to_delete
                    LLVM.API.LLVMInstructionEraseFromParent(inst)
                end
            end
        end
    end
end


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per device
const _compiler_caches = Dict{ZeDevice, Dict{Any, Any}}()
function compiler_cache(dev::ZeDevice)
    cache = get(_compiler_caches, dev, nothing)
    if cache === nothing
        cache = Dict{Any, Any}()
        _compiler_caches[dev] = cache
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
# Whether the driver's SPIR-V runtime accepts the SPV_KHR_bfloat16 extension.
function _driver_supports_bfloat16_spirv()
    @static if isdefined(Core, :BFloat16)
        haskey(oneL0.extension_properties(driver()),
               oneL0.ZE_BFLOAT16_CONVERSIONS_EXT_NAME)
    else
        false
    end
end

@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    supports_fp16 = oneL0.module_properties(device()).fp16flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP16 == oneL0.ZE_DEVICE_MODULE_FLAG_FP16
    supports_fp64 = oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 == oneL0.ZE_DEVICE_MODULE_FLAG_FP64
    # Allow BFloat16 in IR if the device supports it (even if the SPIR-V runtime doesn't
    # advertise the extension). We lower bfloat→i16 in finish_ir! when needed.
    supports_bfloat16 = _device_supports_bfloat16()

    # TODO: emit printf format strings in constant memory
    extensions = String[
        "SPV_EXT_relaxed_printf_string_address_space",
        "SPV_EXT_shader_atomic_float_add"
    ]
    # Only add the SPIR-V extension if the runtime actually supports it
    if _driver_supports_bfloat16_spirv()
        push!(extensions, "SPV_KHR_bfloat16")
    end

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; extensions, supports_fp16, supports_fp64, supports_bfloat16, kwargs...)
    params = oneAPICompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # TODO: on 1.9, this actually creates a context. cache those.
    asm, meta = JuliaContext() do ctx
        GPUCompiler.compile(:obj, job)
    end

    (image=asm, entry=LLVM.name(meta.entry))
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    ctx = context()
    dev = device()
    mod = ZeModule(ctx, dev, compiled.image)
    kernels(mod)[compiled.entry]
end
