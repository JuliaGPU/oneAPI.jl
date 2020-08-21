# printf

# Formatted Output (B.17)

@generated function promote_c_argument(arg)
    # > When a function with a variable-length argument list is called, the variable
    # > arguments are passed using C's old ``default argument promotions.'' These say that
    # > types char and short int are automatically promoted to int, and type float is
    # > automatically promoted to double. Therefore, varargs functions will never receive
    # > arguments of type char, short int, or float.

    if arg == Cchar || arg == Cshort
        return :(Cint(arg))
    elseif arg == Cfloat
        return :(Cdouble(arg))
    else
        return :(arg)
    end
end

macro printf(fmt::String, args...)
    fmt_val = Val(Symbol(fmt))

    return :(emit_printf($fmt_val, $(map(arg -> :(promote_c_argument($arg)), esc.(args))...)))
end

@generated function emit_printf(::Val{fmt}, argspec...) where {fmt}
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = [argspec...]

    JuliaContext() do ctx
        T_void = LLVM.VoidType(ctx)
        T_int32 = LLVM.Int32Type(ctx)
        T_pint8 = LLVM.PointerType(LLVM.Int8Type(ctx))

        # create functions
        param_types = LLVMType[convert(LLVMType, typ, ctx) for typ in arg_types]
        llvm_f, _ = create_function(T_int32, param_types)
        mod = LLVM.parent(llvm_f)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            str = globalstring_ptr!(builder, String(fmt))

            # invoke printf and return
            printf_typ = LLVM.FunctionType(T_int32, [T_pint8]; vararg=true)
            printf = LLVM.Function(mod, "printf", printf_typ)
            push!(function_attributes(printf), EnumAttribute("nobuiltin"))
            chars = call!(builder, printf, [str, parameters(llvm_f)...])

            ret!(builder, chars)
        end

        arg_tuple = Expr(:tuple, arg_exprs...)
        call_function(llvm_f, Int32, Tuple{arg_types...}, arg_tuple)
    end
end
