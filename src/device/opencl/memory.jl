# Shared Memory (part of B.2)

export @LocalMemory

lmem_id = 0

macro LocalMemory(T, dims)
    # FIXME: generating a unique id in the macro is incorrect, as multiple parametrically typed
    #        functions will alias the id (and the size might be a parameter). but incrementing in
    #        the @generated function doesn't work, as it is supposed to be pure and identical
    #        invocations will erroneously share (and even cause multiple lmem globals).
    id = gensym("static_lmem")

    quote
        len = prod($(esc(dims)))
        ptr = emit_localmemory(Val($(QuoteNode(id))), $(esc(T)), Val(len))
        oneDeviceArray($(esc(dims)), ptr)
    end
end

# get a pointer to local memory, with known (static) or zero length (dynamic shared memory)
@generated function emit_localmemory(::Val{id}, ::Type{T}, ::Val{len}=Val(0)) where {id,T,len}
    eltyp = convert(LLVMType, T)

    T_ptr = convert(LLVMType, LLVMPtr{T,AS.Local})

    # create a function
    llvm_f, _ = create_function(T_ptr)

    # create the global variable
    mod = LLVM.parent(llvm_f)
    gv_typ = LLVM.ArrayType(eltyp, len)
    gv = GlobalVariable(mod, gv_typ, GPUCompiler.safe_name(string(id)), AS.Local)
    if len > 0
        linkage!(gv, LLVM.API.LLVMInternalLinkage)
        initializer!(gv, null(gv_typ))
    end
    # TODO: Make the alignment configurable
    alignment!(gv, Base.datatype_alignment(T))

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = gep!(builder, gv, [ConstantInt(0, JuliaContext()),
                                 ConstantInt(0, JuliaContext())])

        val = ptrtoint!(builder, ptr, T_ptr)
        ret!(builder, val)
    end

    call_function(llvm_f, LLVMPtr{T,AS.Local})
end
