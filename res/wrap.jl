# script to parse oneAPI headers and generate Julia wrappers


#
# Parsing
#

using Clang.Generators

function wrap(name, headers...; library="lib$name", defines=[], include_dirs=[])
    options =  load_options(joinpath(@__DIR__, "wrap.toml"))
    options["general"]["library_name"] = library
    options["general"]["output_file_path"] = "lib$(name).jl"

    ctx = create_context([headers...], String[], options)

    build!(ctx)

    return options["general"]["output_file_path"]
end


#
# Fixing-up
#

using CSTParser, Tokenize

## pass infrastructure

struct Edit{T}
    loc::T
    text::String
end

function pass(x, state, f = (x, state)->nothing)
    f(x, state)
    if length(x) > 0
        for a in x
            pass(a, state, f)
        end
    else
        state.offset += x.fullspan
    end
    state
end

function apply(text, edit::Edit{Int})
    string(text[1:edit.loc], edit.text, text[nextind(text, edit.loc):end])
end
function apply(text, edit::Edit{UnitRange{Int}})
    # println("Rewriting '$(text[edit.loc])' to '$(edit.text)'")
    string(text[1:prevind(text, first(edit.loc))], edit.text, text[nextind(text, last(edit.loc)):end])
end


## rewrite passes

mutable struct State
    offset::Int
    edits::Vector{Edit}
end

# insert `@checked` before each function with a `ccall` returning a checked type`
checked_types = [
    "ze_result_t",
]
function insert_check_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :function
        _, def, body, _ = x
        @assert body isa CSTParser.EXPR && body.head == :block
        @assert length(body) == 1

        # Clang.jl-generated ccalls should be directly part of a function definition
        call = body.args[1]
        @assert call isa CSTParser.EXPR && call.head == :call && call[1].val == "ccall"

        # get the ccall return type
        rv = call[5]

        if rv.val in checked_types
            push!(state.edits, Edit(state.offset, "@checked "))
        end
    end
end


## indenting passes

mutable struct IndentState
    offset::Int
    lines
    edits::Vector{Edit}
end

function get_lines(text)
    lines = Tuple{Int,Int}[]
    pt = Tokens.EMPTY_TOKEN(Tokens.Token)
    for t in CSTParser.Tokenize.tokenize(text)
        if pt.endpos[1] != t.startpos[1]
            if t.kind == Tokens.WHITESPACE
                nl = findfirst("\n", t.val) != nothing
                if !nl
                    push!(lines, (length(t.val), 0))
                else
                end
            else
                push!(lines, (0, 0))
            end
        elseif t.startpos[1] != t.endpos[1] && t.kind == Tokens.TRIPLE_STRING
            nls = findall(x->x == '\n', t.val)
            for nl in nls
                push!(lines, (t.startpos[2] - 1, nl + t.startbyte))
            end
        elseif t.startpos[1] != t.endpos[1] && t.kind == Tokens.WHITESPACE
            push!(lines, (t.endpos[2], t.endbyte - t.endpos[2] + 1))
        end
        pt = t
    end
    lines
end

function wrap_at_comma(x, I, state, indent, offset, column, debug=false)
    comma = nothing
    for i in I
        y = x[i]
        # debug && @show y
        # debug && dump(y)
        if column + y.fullspan > 92 && comma !== nothing
            column = indent
            push!(state.edits, Edit(comma, ",\n" * " "^column))
            column += offset - comma[1] - 1 # other stuff might have snuck between the comma and the current expr
            comma = nothing
         elseif y.head == :COMMA
             comma = (offset+1):(offset+y.fullspan)
        end
        offset += y.fullspan
        column += y.fullspan
    end
end

function indent_ccall_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :call && x[1].val == "ccall"
        # figure out how much to indent by looking at where the expr starts
        line = findlast(y -> state.offset >= y[2], state.lines) # index, not the actual number
        line_indent, line_offset = state.lines[line]
        expr_indent = state.offset - line_offset
        indent = expr_indent + sum(x->x.fullspan, [x[i] for i in 1:2])

        if length(x[7]) > 2    # non-empty tuple type
            # break before the tuple type
            offset = state.offset + sum(x->x.fullspan, [x[i] for i in 1:6])
            push!(state.edits, Edit(offset:offset, "\n" * " "^indent))

            # wrap tuple type
            wrap_at_comma(x[7], 1:lastindex(x[7]), state, indent+1, offset, indent+1, )
        end

        if length(x) > 9
            # break before the arguments
            offset = state.offset + sum(x->x.fullspan, [x[i] for i in 1:8])
            push!(state.edits, Edit(offset:offset, "\n" * " "^indent))

            # wrap arguments
            wrap_at_comma(x, 9:lastindex(x), state, indent, offset, indent)
        end
    end
end

function indent_definition_pass(x, state)
    if x isa CSTParser.EXPR && x.head == :function
        # figure out how much to indent by looking at where the expr starts
        line = findlast(y -> state.offset >= y[2], state.lines) # index, not the actual number
        line_indent, line_offset = state.lines[line]
        expr_indent = state.offset - line_offset
        indent = expr_indent + x[1].fullspan + sum(x->x.fullspan, [x[2][i] for i in 1:2])

        if length(x[2]) > 2    # non-empty args
            offset = state.offset + x[1].fullspan + sum(x->x.fullspan, [x[2][i] for i in 1:2])
            wrap_at_comma(x[2], 3:(lastindex(x[2])-1), state, indent, offset, indent)
        end
    end
end



#
# Main application
#

using oneAPI_Level_Zero_Headers_jll

function process(name, headers...; modname=name, kwargs...)
    output_file = wrap(name, headers...; kwargs...)

    let file = output_file
        text = read(file, String)


        ## rewriting passes

        state = State(0, Edit[])
        ast = CSTParser.parse(text, true)

        state.offset = 0
        pass(ast, state, insert_check_pass)

        # apply
        state.offset = 0
        sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
        for i = 1:length(state.edits)
            text = apply(text, state.edits[i])
        end

        ## indentation passes

        lines = get_lines(text)
        state = IndentState(0, lines, Edit[])
        ast = CSTParser.parse(text, true)

        state.offset = 0
        pass(ast, state, indent_definition_pass)

        state.offset = 0
        pass(ast, state, indent_ccall_pass)

        # apply
        state.offset = 0
        sort!(state.edits, lt = (a,b) -> first(a.loc) < first(b.loc), rev = true)
        for i = 1:length(state.edits)
            text = apply(text, state.edits[i])
        end


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


    ## move to destination

    let src = output_file
        dst = joinpath(dirname(@__DIR__), "lib", modname, src)
        cp(src, dst; force=true)
    end


    return
end

function main()
    process("ze", oneAPI_Level_Zero_Headers_jll.ze_api; library="libze_loader", modname="level-zero")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
