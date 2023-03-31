# Shared Memory (part of B.2)

export @LocalMemory, oneLocalArray

@inline function oneLocalArray(::Type{T}, dims) where {T}
    len = prod(dims)
    # NOTE: this relies on const-prop to forward the literal length to the generator.
    #       maybe we should include the size in the type, like StaticArrays does?
    ptr = emit_localmemory(T, Val(len))
    oneDeviceArray(dims, ptr)
end

macro LocalMemory(T, dims)
    Base.depwarn("@LocalMemory is deprecated, please use the oneLocalArray function", :oneLocalArray)

    quote
        oneLocalArray($(esc(T)), $(esc(dims)))
    end
end

# get a pointer to local memory, with known (static) or zero length (dynamic)
@generated function emit_localmemory(::Type{T}, ::Val{len}=Val(0)) where {T,len}
    Context() do ctx
        eltyp = convert(LLVMType, T; ctx)
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.Local}; ctx)

        # create a function
        llvm_f, _ = create_function(T_ptr)

        # create the global variable
        mod = LLVM.parent(llvm_f)
        gv_typ = LLVM.ArrayType(eltyp, len)
        gv = GlobalVariable(mod, gv_typ, "local_memory", AS.Local)
        if len > 0
            linkage!(gv, LLVM.API.LLVMInternalLinkage)
            initializer!(gv, null(gv_typ))
        end
        # TODO: Make the alignment configurable
        alignment!(gv, Base.datatype_alignment(T))

        # generate IR
        IRBuilder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry"; ctx)
            position!(builder, entry)

            ptr = gep!(builder, gv_typ, gv, [ConstantInt(0; ctx), ConstantInt(0; ctx)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.Local})
    end
end
