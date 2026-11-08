## gpucompiler interface implementation

Base.@kwdef struct oneAPICompilerParams <: AbstractCompilerParams
    sub_group_size::Union{Nothing,Int} = nothing
end

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

GPUCompiler.kernel_state_type(::oneAPICompilerJob) = KernelState

# SPIR-V doesn't support 128-bit integers, so keep 64-bit integer math (e.g. `Rational`
# comparisons) that Base computes with them out of Int128
function GPUCompiler.method_tables(job::oneAPICompilerJob)
    if job.config.target.supports_fp64
        (method_table, SPIRVIntrinsics.method_table, GPUToolbox.Overlays.int128_overrides)
    else
        # keep single-precision math that Base computes in Float64 out of double precision
        (method_table, SPIRVIntrinsics.method_table, GPUToolbox.Overlays.float64_overrides,
         GPUToolbox.Overlays.int128_overrides)
    end
end

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

    # require a sub-group size, instead of letting the compiler choose one
    if job.config.params.sub_group_size !== nothing
        entry.metadata["intel_reqd_sub_group_size"] = MDNode([ConstantInt(Int32(job.config.params.sub_group_size))])
    end

    # OpenCL 2.0
    push!(
        get!(mod.metadata, "opencl.ocl.version").operands,
          MDNode([ConstantInt(Int32(2)),
                  ConstantInt(Int32(0))]))

    # SPIR-V 1.5
    push!(
        get!(mod.metadata, "opencl.spirv.version").operands,
          MDNode([ConstantInt(Int32(1)),
                  ConstantInt(Int32(5))]))

    return entry
end

function GPUCompiler.finish_ir!(job::oneAPICompilerJob, mod::LLVM.Module,
                                entry::LLVM.Function)
    # Initialize the heap before SPIR-V lowering converts the state to a reference.
    job.config.kernel && add_heap!(mod, entry)

    entry = invoke(GPUCompiler.finish_ir!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(mod), typeof(entry)},
                   job, mod, entry)

    return entry
end

# Reserve a private arena only when device code reads the heap pointer. At this point,
# GPUCompiler has threaded the state through callees as a leading by-value argument.
function add_heap!(mod::LLVM.Module, entry::LLVM.Function)
    T_state = convert(LLVMType, KernelState)
    heap_field = Base.fieldindex(KernelState, :heap) - 1
    uses_heap(mod, T_state, heap_field) || return false

    params = entry.parameters
    if isempty(params) || params[1].value_type != T_state
        error("kernel `$(entry.name)` allocates but has no kernel state to hold the heap")
    end
    state = params[1]
    # collect the users now, before the insertvalue below becomes one of them
    users = collect(state.users)

    T_size = convert(LLVMType, Csize_t)
    T_heap = LLVM.StructType([T_size, T_size, LLVM.ArrayType(LLVM.Int8Type(), HEAP_SIZE)])
    T_ptr = convert(LLVMType, fieldtype(KernelState, :heap))
    @dispose builder = IRBuilder() begin
        position!(builder, LLVM.at_begin(entry.entry))

        heap = alloca!(builder, T_heap, "heap"; align = HEAP_ALIGNMENT)
        store!(builder, ConstantInt(T_size, 0), struct_gep!(builder, T_heap, heap, 0))
        store!(builder, ConstantInt(T_size, HEAP_SIZE), struct_gep!(builder, T_heap, heap, 1))

        # Replace the original uses, excluding the insertvalue that constructs the state.
        ptr = pointercast!(builder, heap, T_ptr)
        new_state = insert_value!(builder, state, ptr, heap_field, "state")
        for u in users
            replace!(u.operands, state => new_state)
        end
    end

    return true
end

# Inspect field reads rather than calls to malloc, which may already have been inlined.
function uses_heap(mod::LLVM.Module, T_state::LLVMType, heap_field::Integer)
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        inst isa LLVM.ExtractValueInst || continue
        inst.operands[1].value_type == T_state || continue
        first(inst.indices) == heap_field && return true
    end
    return false
end

## compiler implementation (configure, compile, and link)

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, oneAPICompilerConfig}()
const compiler_config_lock = ReentrantLock()
function compiler_config(dev; kwargs...)
    # (as a `NamedTuple`: hashing the `Pairs` boxes each value, on every launch)
    h = hash(dev.driver, hash(dev, hash(values(kwargs))))
    Base.@lock compiler_config_lock begin
        get!(_compiler_configs, h) do
            _compiler_config(dev; kwargs...)
        end
    end
end
# Whether the driver's SPIR-V runtime accepts the SPV_KHR_bfloat16 extension.
#
# XXX: this mirrors IGC's platform table, because IGC terminates the process when a module
#      uses an extension it doesn't accept on the device, and Level Zero has no query for
#      the accepted extensions (intel/intel-graphics-compiler#452). IGC 2.41 accepts
#      SPV_KHR_bfloat16 on Xe-HPG and newer cores except Meteor Lake (`SPIRVExtensions.td`;
#      2.34 behaves the same), which corresponds to graphics IP 12.55 and up, excluding 12.70
#      and 12.71 (MTL, and the ARL-U/S parts that share its IP). Re-check with every IGC bump.
#      ZE_extension_bfloat16_conversions is no substitute: it covers the conversion
#      instructions of SPV_INTEL_bfloat16_conversion, which NEO 26.35 advertises everywhere.
function _driver_supports_bfloat16_spirv(dev=device())
    @static isdefined(Core, :BFloat16) || return false
    oneL0.LTS[] && return false
    ip = oneL0.ip_version(dev)
    ip === nothing && return false
    arch_release = (Int(ip >> 22), Int((ip >> 14) & 0xff))
    return arch_release >= (12, 55) && arch_release ∉ ((12, 70), (12, 71))
end

# The atomic operations that GPUCompiler may select SPIR-V instructions for; the others are
# implemented with compare-and-swap loops.
function _atomics_capabilities(dev, properties; supports_fp16, supports_fp64)
    int64 = properties.flags & oneL0.ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS ==
            oneL0.ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS

    fp_atomics = oneL0.float_atomic_properties(dev)
    fp_atomics === nothing && return SPIRVAtomics(; int64)
    has_add(flags, flag) = flags & flag == flag
    global_add = oneL0.ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD
    local_add = oneL0.ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD

    # IGC rejects (and exits the process on) SPIR-V modules with half-precision atomic
    # additions on devices that don't report them, e.g. Xe-LP. Xe-LP does report
    # double-precision atomics without supporting double-precision arithmetic, though.
    return SPIRVAtomics(;
        int64,
        fadd_f16_global = supports_fp16 && has_add(fp_atomics.fp16flags, global_add),
        fadd_f16_local  = supports_fp16 && has_add(fp_atomics.fp16flags, local_add),
        fadd_f32_global = has_add(fp_atomics.fp32flags, global_add),
        fadd_f32_local  = has_add(fp_atomics.fp32flags, local_add),
        fadd_f64_global = supports_fp64 && has_add(fp_atomics.fp64flags, global_add),
        fadd_f64_local  = supports_fp64 && has_add(fp_atomics.fp64flags, local_add),
    )
end

@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false,
                                    sub_group_size=nothing,
                                    debug_level=Base.JLOptions().debug_level, kwargs...)
    properties = oneL0.module_properties(dev)
    support = eltype_support(dev)
    supports_fp16 = support.float16
    supports_fp64 = support.float64

    if sub_group_size !== nothing
        sizes = oneL0.compute_properties(dev).subGroupSizes
        sub_group_size in sizes ||
            throw(ArgumentError("Sub-group size $sub_group_size is not supported by this device, which supports $(join(sizes, ", "))"))
    end

    # The extensions below have to be declared explicitly; the ones that atomics need are
    # added by GPUCompiler based on `atomics`.
    # TODO: emit printf format strings in constant memory
    extensions = String[
        "SPV_EXT_relaxed_printf_string_address_space",
    ]
    backend = :llvm
    if oneL0.LTS[]
        # The LTS SPIR-V stack cannot codegen native bfloat in generic kernels (clamp! ->
        # InvalidIRError, and SPV_KHR_bfloat16 segfaults NEO), so keep bf16 off entirely on
        # this path; the test suite gates bf16 eltypes on `!oneL0.LTS[]` to match.
        supports_bfloat16 = false
    else
        supports_bfloat16 = _driver_supports_bfloat16_spirv(dev)
        if supports_bfloat16
            push!(extensions, "SPV_KHR_bfloat16")
        end
    end
    extensions_str = join(map(ext -> "+$ext", extensions), ",")

    atomics = _atomics_capabilities(dev, properties; supports_fp16, supports_fp64)

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; backend, extensions = extensions_str, atomics,
                                   supports_fp16, supports_fp64, supports_bfloat16,
                                   driver = :intel, kwargs...)
    params = oneAPICompilerParams(; sub_group_size)
    CompilerConfig(target, params; kernel, name, always_inline, debug_level)
end

# run inference + LLVM codegen + SPIR-V emission. returns `(image, entry)`, both
# session-portable so they survive precompilation when stored on a cached `CodeInstance`.
function compile_to_obj(@nospecialize(job::CompilerJob))
    # TODO: on 1.9, this actually creates a context. cache those.
    return JuliaContext() do ctx
        asm, meta = GPUCompiler.compile(:obj, job)
        # the IR belongs to us: dispose of it, or it leaks along with the context
        @dispose ir = meta.ir begin
            (image = asm, entry = meta.entry.name)
        end
    end
end

# link the SPIR-V bytes into a session-local `ZeKernel` on the given context and device.
function link_kernel(image::Vector{UInt8}, entry::String, ctx::ZeContext, dev::ZeDevice)
    mod = ZeModule(ctx, dev, image)
    kernel = kernels(mod)[entry]

    # Kernels receive their arrays as `oneDeviceArray` structs, so to Level Zero every USM
    # pointer they dereference is an indirect access. Without declaring that, the driver
    # only maps the allocations we explicitly made resident, and a shared allocation that
    # reuses the address of a freed (but never evicted) one isn't visible to the kernel:
    # it reads zeros (JuliaGPU/oneAPI.jl#661). SYCL declares the same for all its kernels.
    oneL0.indirect_access!(
        kernel, oneL0.ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
            oneL0.ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
            oneL0.ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED
    )

    return kernel
end
