export @oneapi, zefunction, kernel_convert


## high-level @oneapi interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline]
const LAUNCH_KWARGS = [:groups, :items, :queue]

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
    vars, var_exprs = assign_args!(code, args)

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
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $kernel_convert($f_var)
                $kernel_args = map($kernel_convert, ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $zefunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)

    return esc(quote
        let
            $code
        end
    end)
end


## argument conversion

struct KernelAdaptor end

# convert oneAPI host pointers to device pointers
Adapt.adapt_storage(to::KernelAdaptor, p::ZePtr{T}) where {T} = reinterpret(Ptr{T}, p)

# convert oneAPI host arrays to device arrays
Adapt.adapt_storage(::KernelAdaptor, xs::oneArray{T,N}) where {T,N} =
  Base.unsafe_convert(oneDeviceArray{T,N,AS.Global}, xs)

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
Adapt.adapt_structure(to::KernelAdaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

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

@inline @generated function call(kernel::AbstractKernel{F,TT}, args...; call_kwargs...) where {F,TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:( args[$i] ) for i in 1:length(args))...)

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

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        onecall(kernel.fun, $call_tt, $(call_args...); call_kwargs...)
    end
end


## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::ZeKernel
end

function launch_configuration(kernel::HostKernel{F,TT}) where {F,TT}
    # Level Zero's zeKernelSuggestGroupSize provides a launch configuration
    # that exactly cover the input size. This can result in very awkward
    # configurations, so roll our own version that behaves like CUDA's
    # occupancy API and assumes the kernel still does bounds checking.

    kernel_props = oneL0.properties(kernel.fun)
    group_size = if kernel_props.maxGroupSize !== missing
        kernel_props.maxGroupSize
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

    # TODO: align the group size based on preferredGroupSize

    return group_size
end


## host-side API

const zefunction_lock = ReentrantLock()

function zefunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    dev = device()

    Base.@lock zefunction_lock begin
        # compile the function
        cache = compiler_cache(dev)
        source = methodinstance(F, tt)
        config = compiler_config(dev; kwargs...)::oneAPICompilerConfig
        fun = GPUCompiler.cached_compilation(cache, source, config, compile, link)

        # create a callable object that captures the function instance. we don't need to think
        # about world age here, as GPUCompiler already does and will return a different object
        h = hash(fun, hash(f, hash(tt)))
        kernel = get(_kernel_instances, h, nothing)
        if kernel === nothing
            # create the kernel state object
            kernel = HostKernel{F,tt}(f, fun)
            _kernel_instances[h] = kernel
        end
        return kernel::HostKernel{F,tt}
    end
end

# cache of kernel instances
const _kernel_instances = Dict{UInt, Any}()

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
