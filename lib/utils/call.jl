export @checked, @debug_ccall

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function body wrapped by an invocation of the
`@check` macro (to be implemented by the caller of this macro), and `unsafe_foo` where no
such invocation is present and the status code is returned to the caller.
"""
macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    # generate a "safe" version that performs a check
    safe_body = quote
        @check $body
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unsafe" version that returns the error code instead
    unsafe_sig = Expr(:call, Symbol("unsafe_", sig.args[1]), sig.args[2:end]...)
    unsafe_def = Expr(:function, unsafe_sig, body)

    return esc(:($safe_def, $unsafe_def))
end

macro debug_ccall(target, rettyp, argtyps, args...)
    @assert Meta.isexpr(target, :tuple)
    f, lib = target.args

    quote
        # get the call target, as e.g. libcuda() triggers initialization, even though we
        # can't use the result in the ccall expression below as it's supposed to be constant
        $(esc(target))

        print($f, '(')
        for (i, arg) in enumerate(($(map(esc, args)...),))
            i > 1 && print(", ")
            render_arg(stdout, arg)
        end
        print(')')
        rv = ccall($(esc(target)), $(esc(rettyp)), $(esc(argtyps)), $(map(esc, args)...))
        println(" = ", rv)
        for (i, arg) in enumerate(($(map(esc, args)...),))
            if arg isa Base.RefValue
                println(" $i: ", arg[])
            end
        end
        rv
    end
end

render_arg(io, arg) = print(io, arg)
render_arg(io, arg::Union{<:Base.RefValue, AbstractArray}) = summary(io, arg)
