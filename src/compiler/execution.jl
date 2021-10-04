export @oneapi, zefunction, kernel_convert

macro oneapi(ex...)
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @oneapi should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, [:launch], [:name], [:groups, :items, :queue])
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
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                local $kernel_f = $kernel_convert($f_var)
                local $kernel_args = map($kernel_convert, ($(var_exprs...),))
                local $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                local $kernel = $zefunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)
    return esc(code)
end


## argument conversion

struct KernelAdaptor end

# convert oneL0 host pointers to device pointers
Adapt.adapt_storage(to::KernelAdaptor, p::ZePtr{T}) where {T} = reinterpret(Ptr{T}, p)

# Base.RefValue isn't GPU compatible, so provide a compatible alternative
struct ZeRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::ZeRefValue) = r.x
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue) = ZeRefValue(adapt(to, r[]))

"""
    kernel_convert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `oneAPI.KernelAdaptor` type.
"""
kernel_convert(arg) = adapt(KernelAdaptor(), arg)


## abstract kernel functionality

abstract type AbstractKernel{F,TT} end

@generated function call(kernel::AbstractKernel{F,TT}, args...; call_kwargs...) where {F,TT}
    sig = Base.signature_type(F, TT)
    args = (:F, (:( args[$i] ) for i in 1:length(args))...)

    # filter out ghost arguments that shouldn't be passed
    predicate = if VERSION >= v"1.5.0-DEV.581"
        dt -> isghosttype(dt) || Core.Compiler.isconstType(dt)
    else
        dt -> isghosttype(dt)
    end
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

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        Base.@_inline_meta

        onecall(kernel.fun, $call_tt, $(call_args...); call_kwargs...)
    end
end


## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    fun::ZeKernel
end


## host-side API

function zefunction(f::Core.Function, tt::Type=Tuple{}; name=nothing, kwargs...)
    dev = device()
    cache = get!(()->Dict{UInt,Any}(), zefunction_cache, dev)
    source = FunctionSpec(f, tt, true, name)
    target = SPIRVCompilerTarget(; kwargs...)
    params = oneAPICompilerParams()
    job = CompilerJob(target, source, params)
    GPUCompiler.cached_compilation(cache, job,
                                   zefunction_compile, zefunction_link)::HostKernel{f,tt}
end

const zefunction_cache = Dict{Any,Any}()

function zefunction_compile(@nospecialize(job::CompilerJob))
    mi, mi_meta = GPUCompiler.emit_julia(job)
    ir, ir_meta = GPUCompiler.emit_llvm(job, mi)
    asm, asm_meta = GPUCompiler.emit_asm(job, ir; format=LLVM.API.LLVMObjectFile)

    return (image=asm, entry=LLVM.name(ir_meta.entry))
end

# JIT into an executable kernel object
function zefunction_link(@nospecialize(job::CompilerJob), compiled)
    ctx = context()
    dev = device()
    mod = ZeModule(ctx, dev, compiled.image)
    kernel = kernels(mod)[compiled.entry]

    return HostKernel{job.source.f,job.source.tt}(kernel)
end

@inline function onecall(kernel::ZeKernel, tt, args...; groups::ZeDim=1, items::ZeDim=1,
                         queue::ZeCommandQueue=global_queue(context(), device()))
    for (i, arg) in enumerate(args)
        oneL0.arguments(kernel)[i] = arg
    end

    groupsize!(kernel, items)
    execute!(queue) do list
        append_launch!(list, kernel, groups)
    end
end

function (kernel::HostKernel)(args...; kwargs...)
    call(kernel, map(kernel_convert, args)...; kwargs...)
end


## TODO: device-side kernels
