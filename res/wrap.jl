# script to parse oneAPI headers and generate Julia wrappers


#
# Parsing
#

using Clang.Generators
using JuliaFormatter

function wrap(name, headers...; library="lib$name", defines=[], include_dirs=[])
    options =  load_options(joinpath(@__DIR__, "wrap.toml"))
    options["general"]["library_name"] = library
    options["general"]["output_file_path"] = "lib$(name).jl"

    args = get_default_args()
    ctx = create_context([headers...], args, options)

    build!(ctx, BUILDSTAGE_NO_PRINTING)

    # tweak exprs
    for node in get_nodes(ctx.dag)
        exprs = get_exprs(node)
        for (i, expr) in enumerate(exprs)
            exprs[i] = add_check_pass(expr)
        end
    end

    build!(ctx, BUILDSTAGE_PRINTING_ONLY)

    return options["general"]["output_file_path"]
end



## rewrite passes

# insert `@checked` before each function with a `ccall` returning a checked type`
checked_types = [
    "ze_result_t",
]
function add_check_pass(x::Expr)
    Meta.isexpr(x, :function) || return x
    body = x.args[2].args[1]
    if Meta.isexpr(body, :macrocall)  # `@ccall`
        ret_type = string(body.args[3].args[2])
        if ret_type in checked_types
            return Expr(:macrocall, Symbol("@checked"), nothing, x)
        end
    else
        @assert Meta.isexpr(body, :call)  # `ccall`
        ret_type = string(body.args[3])
        if ret_type in checked_types
            return Expr(:macrocall, Symbol("@checked"), nothing, x)
        end
    end
    return x
end

#
# Main application
#

using oneAPI_Level_Zero_Headers_jll

function process(name, headers...; modname=name, kwargs...)
    output_file = wrap(name, headers...; kwargs...)

    let file = output_file
        text = read(file, String)

        ## header

        squeezed = replace(text, "\n\n\n"=>"\n\n")
        while length(text) != length(squeezed)
            text = squeezed
            squeezed = replace(text, "\n\n\n"=>"\n\n")
        end
        text = squeezed


        write(file, text)
    end


    ## manual patches

    patchdir = joinpath(@__DIR__, "patches", name)
    if isdir(patchdir)
        for entry in readdir(patchdir)
            if endswith(entry, ".patch")
                path = joinpath(patchdir, entry)
                run(`patch -p1 -i $path`)
            end
        end
    end


    ## move to destination and format

    let src = output_file
        dst = joinpath(dirname(@__DIR__), "lib", modname, src)
        cp(src, dst; force=true)
        format(dst, YASStyle(), always_use_return=false)
    end

    return
end

function main()
    process("ze", oneAPI_Level_Zero_Headers_jll.ze_api; library="libze_loader", modname="level-zero")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
