
const opencl_builtins = String["printf"]

# OpenCL functions need to be mangled according to the C++ Itanium spec. We implement a very
# limited version of that spec here, just enough to support OpenCL built-ins.
#
# This macro also keeps track of called builtins, generating `ccall("extern...", llvmcall)`
# expressions for them (so that we can exclude them during IR verification).
macro builtin_ccall(name, ret, argtypes, args...)
    @assert Meta.isexpr(argtypes, :tuple)
    argtypes = argtypes.args

    # C++-style mangling; very limited to just support these intrinsics
    # TODO: generalize for use with other intrinsics? do we need to mangle those?
    mangled = "_Z$(length(name))$name"
    for t in argtypes
        # with `@eval @builtin_ccall`, we get actual types in the ast, otherwise symbols
        t = isa(t, Symbol) ? eval(t) : t

        mangled *= if t == Cint
            'i'
        elseif t == Cuint
            'j'
        elseif t == Clong
            'l'
        elseif t == Culong
            'm'
        elseif t == Cshort
            's'
        elseif t == Cushort
            't'
        elseif t == Cchar
            'c'
        elseif t == Cuchar
            'h'
        elseif t == Cfloat
            'f'
        elseif t == Cdouble
            'd'
        else
            error("Unknown type $t")
        end
    end

    push!(opencl_builtins, mangled)
    esc(quote
        ccall($("extern $mangled"), llvmcall, $ret, ($(argtypes...),), $(args...))
    end)
end


## device overrides

# local method table for device functions
@static if isdefined(Base.Experimental, Symbol("@overlay"))
Base.Experimental.@MethodTable(method_table)
else
const method_table = nothing
end

# list of overrides (only for Julia 1.6)
const overrides = quote end

macro device_override(ex)
    code = quote
        $GPUCompiler.@override($method_table, $ex)
    end
    if isdefined(Base.Experimental, Symbol("@overlay"))
        return esc(code)
    else
        push!(overrides.args, code)
        return
    end
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        $(combinedef(def))
        @device_override $ex
    end)
end
