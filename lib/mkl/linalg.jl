using LinearAlgebra

function gemm_dispatch!(C::oneVecOrMat, A, B, alpha::Number=true, beta::Number=false)
    if ndims(A) > 2
        throw(ArgumentError("A has more than 2 dimensions"))
    elseif ndims(B) > 2
        throw(ArgumentError("B has more than 2 dimensions"))
    end
    mA, nA = size(A,1), size(A,2)
    mB, nB = size(B,1), size(B,2)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 0 || nA == 0 || nB == 0
        if size(C) != (mA, nB)
            throw(DimensionMismatch("C has dimensions $(size(C)), should have ($mA,$nB)"))
        end
        return LinearAlgebra.rmul!(C, 0)
    end

    tA, dA = if A isa Transpose
        'T', parent(A)
    elseif A isa Adjoint
        'C', parent(A)
    else
        'N', A
    end

    tB, dB = if B isa Transpose
        'T', parent(B)
    elseif B isa Adjoint
        'C', parent(B)
    else
        'N', B
    end

    T = eltype(C)
    if T <: onemklFloat && dA isa oneDenseArray{T} && dB isa oneDenseArray{T}
        gemm!(tA, tB, alpha, dA, dB, beta, C)
    else
        GPUArrays.generic_matmatmul!(C, A, B, alpha, beta)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(C::oneMatrix, A::oneVecOrMat, B::oneVecOrMat, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneMatrix, A::Transpose{<:Any, <:oneVecOrMat}, B::oneMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneMatrix, A::oneMatrix, B::Transpose{<:Any, <:oneVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneMatrix, A::Transpose{<:Any, <:oneVecOrMat}, B::Transpose{<:Any, <:oneVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneMatrix, A::Adjoint{<:Any, <:oneVecOrMat}, B::oneMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneMatrix, A::oneMatrix, B::Adjoint{<:Any, <:oneVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneMatrix, A::Adjoint{<:Any, <:oneVecOrMat}, B::Adjoint{<:Any, <:oneVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneMatrix, A::Transpose{<:Any, <:oneVecOrMat}, B::Adjoint{<:Any, <:oneVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneMatrix, A::Adjoint{<:Any, <:oneVecOrMat}, B::Transpose{<:Any, <:oneVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
    end
end
