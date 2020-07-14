
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
        mangled *= if t == :Cint
            'i'
        elseif t == :Cuint
            'j'
        elseif t == :Culong
            'm'
        elseif t == :Cfloat
            'f'
        elseif t == :Cdouble
            'd'
        else
            error("Unknown type $t")
        end
    end

    push!(opencl_builtins, mangled)
    esc(quote
        ccall($"extern $mangled", llvmcall, $ret, ($(argtypes...),), $(args...))
    end)
end
