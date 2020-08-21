export @oneapi, kernel_convert

macro oneapi(ex...)
    call = ex[end]
    kwargs = ex[1:end-1]

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @cuda should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, [:name], [:groups, :items, :config, :queue])
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            GC.@preserve $(vars...) begin
                local $kernel_args = map($kernel_convert, ($(var_exprs...),))
                local $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                local $kernel = $compile($f, $kernel_tt; $(compiler_kwargs...))
                $kernel($kernel_args...; $(call_kwargs...))
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
    to_pass = map(!isghosttype, sig.parameters)
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

        _call(kernel, $call_tt, $(call_args...); call_kwargs...)
    end
end

(kernel::AbstractKernel)(args...; kwargs...) = call(kernel, args...; kwargs...)


## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    fun::ZeKernel
end

function compile(f::Core.Function, tt::Type=Tuple{}; name=nothing, kwargs...)
    dev = device()
    env = hash(dev)

    spec = FunctionSpec(f, tt, true, name)
    GPUCompiler.cached_compilation(_compile, spec, env; kwargs...)::HostKernel{f,tt}
end

function _compile(source::FunctionSpec; kwargs...)
    ctx = context()
    dev = device()
    target = SPIRVCompilerTarget(; kwargs...)
    params = oneAPICompilerParams()
    job = CompilerJob(target, source, params)
    image, kernel_fn, undefined_fns = GPUCompiler.compile(:obj, job)

    # JIT into an executable kernel object
    mod = ZeModule(ctx, dev, image)
    kernel = kernels(mod)[kernel_fn]

    return HostKernel{source.f,source.tt}(kernel)
end

@inline function _call(kernel::HostKernel, tt, args...; config=nothing, kwargs...)
    if config !== nothing
        _call(kernel.fun, tt, args...; kwargs..., config(kernel)...)
    else
        _call(kernel.fun, tt, args...; kwargs...)
    end
end

@inline function _call(kernel::ZeKernel, tt, args...; groups::ZeDim=1, items::ZeDim=1,
                       queue::ZeCommandQueue=global_queue(context(), device()))
    for (i, arg) in enumerate(args)
        arguments(kernel)[i] = arg
    end

    groupsize!(kernel, items)
    execute!(queue) do list
        append_launch!(list, kernel, groups)
    end
end


## TODO: device-side kernels
