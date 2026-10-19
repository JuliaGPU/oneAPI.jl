export @oneapi, zefunction, kernel_convert


## high-level @oneapi interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :atomics, :sub_group_size]
const LAUNCH_KWARGS = [:groups, :items, :queue]

"""
    @oneapi [kwargs...] kernel(args...)

High-level interface for launching Julia kernels on Intel GPUs using oneAPI.

This macro compiles a Julia function to SPIR-V, prepares the arguments, and optionally
launches the kernel on the GPU.

# Keyword Arguments

## Macro Keywords (compile-time)
- `launch::Bool=true`: Whether to launch the kernel immediately. If `false`, returns the
  compiled kernel object without executing it.

## Compiler Keywords
- `kernel::Bool=false`: Whether to compile as a kernel (true) or device function (false)
- `name::Union{String,Nothing}=nothing`: Explicit name for the kernel
- `always_inline::Bool=false`: Whether to always inline device functions
- `atomics::SPIRVAtomics`: Override the atomic operations GPUCompiler may select SPIR-V
  instructions for, e.g. `atomics=oneAPI.SPIRVAtomics(fadd_f32_global=true)`. This replaces
  the whole descriptor that is otherwise derived from the device and toolchain. Enabling
  operations the device or toolchain doesn't support can make compilation fail, or terminate
  the process from within the driver's compiler (as IGC does for half-precision atomic
  additions on Xe-LP). Disabled floating-point operations are implemented with integer
  compare-and-swap loops instead, and disabling `int64` makes 64-bit atomic operations an
  error.
- `sub_group_size::Union{Int,Nothing}=nothing`: The sub-group size the kernel has to be
  compiled for, one of the device's `oneL0.compute_properties(dev).subGroupSizes`. By
  default, the compiler chooses one.

## Launch Keywords (runtime)
- `groups`: Number of workgroups (required). Can be an integer or tuple.
- `items`: Number of work-items per workgroup (required). Can be an integer or tuple.
- `queue=global_stream(...)`: Submission target — the task's `oneStream` by default. An
  explicit `ZeCommandQueue` is also accepted and submits each launch as a command list.

# Examples

```julia
# Simple vector addition kernel
function vadd(a, b, c)
    i = get_global_id()
    @inbounds c[i] = a[i] + b[i]
    return
end

a = oneArray(rand(Float32, 1024))
b = oneArray(rand(Float32, 1024))
c = similar(a)

# Launch with 4 workgroups of 256 items each
@oneapi groups=4 items=256 vadd(a, b, c)

# Compile without launching
kernel = @oneapi launch=false vadd(a, b, c)
kernel(a, b, c; groups=4, items=256)  # Launch later
```

See also: `zefunction`, `kernel_convert`
"""
macro oneapi(ex...)
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @oneapi should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    _, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key == :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @cuda should be a constant value"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@oneapi with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    push!(code.args,
        :($compile_and_launch($f, ($(var_exprs...),), $(Val(launch)),
                              (; $(call_kwargs...)), (; $(compiler_kwargs...)))))

    return esc(quote
        let
            $code
        end
    end)
end


# convert the arguments, call the compiler and launch the kernel while keeping the original
# arguments alive. this is a function rather than part of the macro expansion so that the
# arguments have concrete types when they are preserved: `GC.@preserve` of a value that is a
# union of singletons, e.g. `op` in `for op in (max, min)`, crashes codegen on Julia 1.12 and
# 1.13.0-1.13.1 (JuliaLang/julia#63482).
@inline function compile_and_launch(f::F, args::Tuple, ::Val{launch},
                                    call_kwargs::NamedTuple,
                                    compiler_kwargs::NamedTuple) where {F,launch}
    GC.@preserve f args begin
        kernel_f = kernel_convert(f)
        kernel_args = map(kernel_convert, args)
        kernel_tt = Tuple{map(Core.Typeof, kernel_args)...}
        kernel = zefunction(kernel_f, kernel_tt; compiler_kwargs...)
        if launch
            launch_managed(kernel, f, args; call_kwargs...)
            kernel
        else
            with_source(kernel, f)
        end
    end
end


## argument conversion

# conversion doesn't take ownership of the arrays' memory, since the launch target isn't
# known yet. instead, the allocations are collected, if requested, so that the launch can
# take ownership of them right before submitting the kernel.
struct KernelAdaptor
  refs::Union{Nothing, Vector{DataRef}}
end
KernelAdaptor() = KernelAdaptor(nothing)

# device pointers are passed as global pointers; a plain `Ptr` is a private pointer in SPIR-V
Adapt.adapt_storage(to::KernelAdaptor, p::ZePtr{T}) where {T} =
  reinterpret(LLVMPtr{T,AS.CrossWorkgroup}, p)

# convert oneAPI host arrays to device arrays
function Adapt.adapt_storage(to::KernelAdaptor, xs::oneArray{T,N}) where {T,N}
  to.refs === nothing || push!(to.refs, xs.data)
  ptr = reinterpret(LLVMPtr{T,AS.CrossWorkgroup}, device_pointer(xs))
  oneDeviceArray{T,N,AS.CrossWorkgroup}(size(xs), ptr, xs.maxsize - xs.offset)
end

# Base.RefValue isn't GPU compatible, so provide a compatible alternative.
# TODO: port improvements from CUDA.jl
struct ZeRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::ZeRefValue) = r.x
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue) = ZeRefValue(adapt(to, r[]))

# broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
# avoid that by using a special kind of ref that knows about the boxed type.
struct oneRefType{T} <: Ref{DataType} end
Base.getindex(r::oneRefType{T}) where T = T
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue{<:Union{DataType,Type}}) =
    oneRefType{r[]}()

# case where type is the function being broadcasted
# (on Julia 1.14, the function type parameter is `Core.TypeEgal{T} <: Type{T}`)
Adapt.adapt_structure(to::KernelAdaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, <:Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

# functions that capture a type, e.g., `Base.Fix1(convert, T)` as used by LinearAlgebra,
# which isn't a valid kernel argument either
Adapt.adapt_structure(to::KernelAdaptor, f::Base.Fix1{<:Any, <:Type{T}}) where {T} =
    let g = adapt(to, f.f); (x...) -> g(T, x...) end
Adapt.adapt_structure(to::KernelAdaptor, f::Base.Fix2{<:Any, <:Type{T}}) where {T} =
    let g = adapt(to, f.f); (x...) -> g(x..., T) end

"""
    kernel_convert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `oneAPI.KernelAdaptor` type.
"""
kernel_convert(arg, refs=nothing) = adapt(KernelAdaptor(refs), arg)


## abstract kernel functionality

abstract type AbstractKernel{F,TT} end

@inline @generated function call(kernel::AbstractKernel{F,TT}, args::Tuple; call_kwargs...) where {F,TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:( args[$i] ) for i in 1:fieldcount(args))...)

    # filter out ghost arguments that shouldn't be passed
    predicate = dt -> isghosttype(dt) || Core.Compiler.isconstType(dt)
    to_pass = map(!predicate, sig.parameters)
    call_t =                  Type[x[1] for x in zip(sig.parameters,  to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass)            if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    for (i,dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Any}
            call_args[i] = :C_NULL
        end
    end

    # Match GPUCompiler's hidden state argument. Keep onecall usable for foreign kernels.
    pushfirst!(call_t, KernelState)
    pushfirst!(call_args, :(KernelState()))

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        onecall(kernel.fun, $call_tt, ($(call_args...),); call_kwargs...)
    end
end


## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::ZeKernel

    # the callable the kernel was compiled from, whose memory launches take ownership of.
    # only set on the kernels `@oneapi launch=false` returns, as the cached ones are shared
    # by every callable that converts to `f`.
    source::Any
end
HostKernel{F,TT}(f, fun::ZeKernel) where {F,TT} = HostKernel{F,TT}(f, fun, nothing)

with_source(kernel::HostKernel{F,TT}, source) where {F,TT} =
    HostKernel{F,TT}(kernel.f, kernel.fun, source)

# Upper bound on the spill (scratch) memory a single work-group may require, in bytes.
#
# The driver allocates `spillMemSize * group_size` of scratch per work-group. `maxGroupSize`
# does not account for spill, so a heavily spilling kernel is reported as launchable at a
# group size whose scratch demand is enormous: on a Data Center GPU Max 1550, a kernel
# spilling 3648 B/thread is reported launchable at 1024 items/group, i.e. ~3.7 MB of scratch
# for one work-group.
#
# Level Zero exposes no scratch-space query, so the budget is a conservative constant rather
# than a derived one.
const MAX_GROUP_SCRATCH = 1024 * 1024

function launch_configuration(kernel::HostKernel{F,TT}) where {F,TT}
    # Level Zero's zeKernelSuggestGroupSize provides a launch configuration
    # that exactly cover the input size. This can result in very awkward
    # configurations, so roll our own version that behaves like CUDA's
    # occupancy API and assumes the kernel still does bounds checking.

    max_group_size = oneL0.max_group_size(kernel.fun)
    group_size = if max_group_size !== missing
        max_group_size
    else
        # without the MAX_GROUP_SIZE extension, we need to be conservative
        dev = kernel.fun.mod.device
        compute_props = oneL0.compute_properties(dev)
        max_size = compute_props.maxTotalGroupSize

        ## when the kernel uses many registers (which we can't query without
        ## extensions that landed _after_ MAX_GROUP_SIZE, so don't bother)
        ## the groupsize should be halved
        group_size = max_size ÷ 2
    end

    # keep a spilling kernel's per-group scratch within budget. CUDA.jl gets this for free
    # from an occupancy API that is register-pressure aware; Level Zero reports the spill
    # size but does not fold it into `maxGroupSize`, so account for it here. Rounded down to
    # a power of two, both because group sizes want to be anyway and to stay clear of the
    # limit rather than right at it.
    spill = oneL0.spill_mem_size(kernel.fun)
    if spill > 0 && group_size * spill > MAX_GROUP_SCRATCH
        group_size = max(1, prevpow(2, max(1, MAX_GROUP_SCRATCH ÷ spill)))
    end

    # TODO: align the group size based on preferredGroupSize

    return group_size
end


## host-side API

const zefunction_lock = ReentrantLock()

function zefunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    fun = zefunction_link(methodinstance(F, tt); kwargs...)
    return HostKernel{F,tt}(f, fun)
end

# the parts of `zefunction` that don't depend on the type of the function, kept separate so
# that they are compiled once rather than for every kernel
function zefunction_link(source::Core.MethodInstance; kwargs...)
    Base.@lock zefunction_lock begin
        ctx = context()
        dev = device()
        config = compiler_config(dev; kwargs...)::oneAPICompilerConfig
        job = CompilerJob(source, config)

        res = compile_or_lookup(job)::oneAPIResults

        # Resolve the ZeKernel for the active context and device. Linear scan over the
        # session-local cache; almost always n=1, so this is one `===`/`==` compare.
        fun = nothing
        @inbounds for (cached_ctx, cached_dev, cached_kernel) in res.kernels
            if cached_ctx == ctx && cached_dev == dev
                fun = cached_kernel
                break
            end
        end
        if fun === nothing
            fun = link_kernel(res.image::Vector{UInt8}, res.entry::String, ctx, dev)
            # Don't cache session-local kernel handles while precompiling: the results
            # struct is serialized into the package image along with its CodeInstance,
            # and the handles would come back dangling.
            if ccall(:jl_generating_output, Cint, ()) != 1
                push!(res.kernels, (ctx, dev, fun))
            end
        end

        return fun::ZeKernel
    end
end

# Look up cached compile artifacts for `job`, compiling on miss. Storage is managed
# by `GPUCompiler.cached_results` (Julia's integrated code cache on 1.11+, which also
# persists artifacts through precompilation; a session-local store on 1.10).
#
# `image === nothing` identifies a `oneAPIResults` that hasn't been compiled yet. Every
# lookup is reported to the `@device_code_*` hook, so reflection observes cached kernels
# without recompiling them.
# Specialize on the target/parameter types so callers can avoid boxing CompilerJob.
# Keep the body out of callers that specialize per kernel.
@noinline function compile_or_lookup(job::CompilerJob)::oneAPIResults
    GPUCompiler.run_compile_hook(job)
    res = GPUCompiler.cached_results(oneAPIResults, job)
    if res === nothing || res.image === nothing
        compiled = compile_to_obj(job)
        res = @something res GPUCompiler.cached_results(oneAPIResults, job)
        res.image = compiled.image
        res.entry = compiled.entry
    end
    return res
end

@inline function onecall(kernel::ZeKernel, tt, args::Tuple; groups::ZeDim=1, items::ZeDim=1,
                         queue::Union{oneStream, ZeCommandQueue}=global_stream(context(), device()),
                         roots=nothing)
    Base.@lock kernel begin
        # unrolled, as iterating a heterogeneous tuple would dispatch at run time
        ntuple(Val(length(args))) do i
            oneL0.arguments(kernel)[i] = args[i]
            return
        end
        launch_kernel(kernel, groups, items, queue, roots)
    end
end

# kept out of `onecall`, which is compiled for every kernel signature, as none of this
# depends on it
@noinline function launch_kernel(kernel::ZeKernel, groups::ZeDim, items::ZeDim,
                                 queue::Union{oneStream, ZeCommandQueue}, roots)
    groupsize!(kernel, items)
    launch!(queue, kernel, groups, roots)
end

# `roots` are the references to the memory the kernel uses, which a batch keeps alive
@inline function launch!(s::oneStream, kernel::ZeKernel, groups::ZeDim, roots=nothing)
    mkl_wait!(s)

    # NEO allocates a stream's scratch buffer at the first submission of a kernel whose
    # spill exceeds what is already allocated, and that allocation aborts the process on
    # failure (no null check). Cross each new spill high-water mark deliberately, at the
    # cleanest reachable moment, instead of at a GC-lottery-determined one, and submit
    # the kernel right away.
    spill = oneL0.spill_mem_size(kernel)
    direct = spill > s.scratch_hwm
    direct && scratch_hedge!(s, spill)

    @lock s.lock begin
        check_error(s)
        direct |= !s.batching || s.idle || oneL0.sync_each_submission()
        batch = direct ? nothing : open_batch!(s)
        if batch === nothing
            flush_batch!(s)
            append_launch!(s.immediate, kernel, groups)
            s.idle = false
        else
            append_launch!(batch.list, kernel, groups)
            s.recorder === current_task() || register_batching!(s)
            root!(batch, roots)
            batch.launches += 1
            batch.launches == BATCH_SIZE && flush_batch!(s)
        end
    end
    # wait cooperatively, as `synchronize` does, or the workaround blocks the thread
    oneL0.sync_each_submission() && oneL0.nonblocking_synchronize(s.immediate)
    return
end

# explicit-queue compatibility: `@oneapi queue=...` submits each launch as a command list
@inline function launch!(queue::ZeCommandQueue, kernel::ZeKernel, groups::ZeDim, roots=nothing)
    spill = oneL0.spill_mem_size(kernel)
    spill > queue.scratch_hwm && scratch_hedge!(queue, spill)
    execute!(queue) do list
        append_launch!(list, kernel, groups)
    end
end

# Slow path of the scratch hedge, firing once per (stream, spill tier): retire in-flight
# work, flush deferred releases, and run finalizers so dead driver objects and arrays are
# destroyed before NEO performs its null-check-free scratch allocation. Opt out with
# ONEAPI_SCRATCH_HEDGE=0; the high-water mark is maintained regardless, so the toggle
# only skips the drain.
@noinline function scratch_hedge!(target::Union{oneStream, ZeCommandQueue}, spill::Int)
    if oneL0.SCRATCH_HEDGE[]
        # (without releasing memory batches kept alive: the caller holds locks)
        if target isa oneStream
            err = synchronize_stream!(target)
            err === nothing || throw(err)
        else
            oneL0.synchronize(target)
        end
        oneL0._run_reclaim_callbacks()
        GC.gc(false)
        Threads.atomic_add!(oneL0.SCRATCH_HEDGE_COUNT, 1)
    end
    target.scratch_hwm = spill
    return
end

# `Vararg{Any,N}`, as Julia doesn't specialize a method on varargs it only passes on, and an
# explicit `Core.kwcall` method, as the one lowered from keyword arguments splats them
@inline (kernel::HostKernel)(args::Vararg{Any,N}) where {N} =
    launch_managed(kernel, kernel.source, args)
@inline Core.kwcall(kwargs::NamedTuple, kernel::HostKernel, args::Vararg{Any,N}) where {N} =
    launch_managed(kernel, kernel.source, args; kwargs...)

# launch a kernel, taking ownership of the memory its arguments use, and of the memory
# captured by `source`, the callable the kernel was compiled from (if known): its converted
# form, which the kernel holds, only contains pointers. ownership is taken with the
# allocations locked until the kernel has been submitted, so that other tasks never observe
# it without the work it describes.
@inline function launch_managed(kernel::HostKernel, source, args::Tuple;
                                queue::Union{oneStream, ZeCommandQueue}=global_stream(context(), device()),
                                kwargs...)
    refs = DataRef[]
    source === nothing || kernel_convert(source, refs)
    converted = map(arg -> kernel_convert(arg, refs), args)
    managed = Managed[ref[] for ref in refs]
    ret = GC.@preserve source args with_ownership(queue, managed) do
        call(kernel, converted; queue, roots=refs, kwargs...)
    end
    # memory that batches kept alive and turned out to be unused, also while waiting for
    # another stream above. this is the first point without locks held.
    release_pending!()
    return ret
end


## TODO: device-side kernels
