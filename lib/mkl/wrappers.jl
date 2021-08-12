using CxxWrap

@readmodule(joinpath(@__DIR__, "../../deps/liboneapilib.so"), :define_module_mkl)
@wraptypes
CxxWrap.argument_overloads(t::Type{<:ZeCxxPtr{T}}) where {T} = [oneArray{T}]
@wrapfunctions

# XXX: CxxWrap makes use a C++-specific wrapper for ZePtr, ZeCxxPtr,
#      so forward all relevant calls to the underlying Level 0 pointer.
Base.cconvert(::Type{<:ZeCxxPtr}, x) = x
Base.unsafe_convert(::Type{<:ZeCxxPtr{T}}, x::oneArray{T}) where {T} =
    ZeCxxPtr{T}(reinterpret(Ptr{T}, Base.unsafe_convert(ZePtr{T}, x)))

function __init__()
  @initcxx
end


#
# Auxiliary
#

function Base.convert(::Type{onemklTranspose}, trans::Char)
    if trans == 'N'
        return ONEMKL_TRANSPOSE_NONTRANS
    elseif trans == 'T'
        return ONEMKL_TRANSPOSE_TRANS
    elseif trans == 'C'
        return ONEMLK_TRANSPOSE_CONJTRANS
    else
        throw(ArgumentError("Unknown transpose $trans"))
    end
end



#
# BLAS
#

# level 3

for (fname, elty) in
        ((:onemklDgemm,:Float64),
         (:onemklSgemm,:Float32),
         (:onemklHgemm, :Float16),
         (:onemklZgemm,:ComplexF64),
         (:onemklCgemm,:ComplexF32))
    @eval begin
        function gemm!(transA::Char,
                       transB::Char,
                       alpha::Number,
                       A::oneVecOrMat{$elty},
                       B::oneVecOrMat{$elty},
                       beta::Number,
                       C::oneVecOrMat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end

            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))

            device(A) == device(B) == device(C) || error("Multi-device GEMM not supported")
            context(A) == context(B) == context(C) || error("Multi-context GEMM not supported")
            queue = global_queue(context(A), device(A))

            # FIXME: CxxWrap-generated function signatures are too narrowly typed,
            #        breaking ccall-based conversions
            transA = convert(onemklTranspose, transA)
            transB = convert(onemklTranspose, transB)
            alpha = $elty(alpha)
            beta = $elty(beta)

            $fname(sycl_queue(queue), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end

        function gemm(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::oneVecOrMat{$elty},
                      B::oneVecOrMat{$elty})
            gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end

        function gemm(transA::Char,
                      transB::Char,
                      A::oneVecOrMat{$elty},
                      B::oneVecOrMat{$elty})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end
