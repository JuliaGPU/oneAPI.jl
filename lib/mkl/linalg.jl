using LinearAlgebra

function gemm_dispatch!(C::oneStridedVecOrMat, A, B, alpha::Number=true, beta::Number=false)
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
    if T <: onemklFloat && dA isa oneStridedArray{T} && dB isa oneStridedArray{T} &&
       T != Float16 # onemklHgemm is currently not hooked-up
        gemm!(tA, tB, alpha, dA, dB, beta, C)
    else
        GPUArrays.generic_matmatmul!(C, A, B, alpha, beta)
    end
end

LinearAlgebra.BLAS.asum(x::oneStridedVecOrMat{<:onemklFloat}) = oneMKL.asum(length(x), x)

function LinearAlgebra.axpy!(alpha::Number, x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}) where T<:onemklFloat
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.axpy!(length(x), alpha, x, y)
end

LinearAlgebra.rmul!(x::oneStridedVecOrMat{<:onemklFloat}, k::Number) = 
	oneMKL.scal!(length(x), convert(eltype(x),k), x)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::oneStridedVecOrMat{<:onemklFloat}, k::Real) =
	invoke(rmul!, Tuple{typeof(x), Number}, x, k)
LinearAlgebra.norm(x::oneStridedVecOrMat{<:onemklFloat}) = oneMKL.nrm2(length(x), x)

function LinearAlgebra.dot(x::oneStridedArray{T}, y::oneStridedArray{T}) where T<:Union{Float32, Float64}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.dot(n, x, y)
end

function LinearAlgebra.dot(x::oneStridedArray{T}, y::oneStridedArray{T}) where T<:Union{ComplexF32, ComplexF64}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.dotc(n, x, y)
end

@inline function LinearAlgebra.mul!(y::oneStridedVecOrMat{T}, A::Hermitian{T,<:oneStridedVecOrMat}, x::oneStridedVecOrMat{T},
             α::Number, β::Number) where {T<:Union{ComplexF32,ComplexF64}}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return oneMKL.hemv!(A.uplo, alpha, A.data, x, beta, y)
    else
        error("only supports BLAS type, got $T")
    end
end

# symmetric mul!
# level 2
@inline function LinearAlgebra.mul!(y::oneStridedVecOrMat{T}, A::Hermitian{T,<:oneStridedVecOrMat}, x::oneStridedVecOrMat{T},
             α::Number, β::Number) where {T<:Union{Float32,Float64}}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return oneMKL.symv!(A.uplo, alpha, A.data, x, beta, y)
    else
        error("only supports BLAS type, got $T")
    end
end

## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:oneStridedVecOrMat},
                            b::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trmv!($uploc, 'N', $isunitc, parent(A), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:oneStridedVecOrMat},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            oneMKL.trsv!($uploc, 'N', $isunitc, parent(A), B)
    end
end

## adjoint/transpose multiplication ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{<:Any,<:Transpose{T,<:oneStridedMatrix}},
                            b::oneStridedVector{T}) where {T<:onemklFloat} =
            oneMKL.trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            b::oneStridedVector{T}) where {T<:Union{Float32,Float64}} =
            oneMKL.trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            b::oneStridedVector{T}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trmv!($uploc, 'C', $isunitc, parent(parent(A)), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:oneStridedMatrix}},
                            B::oneStridedVector{T}) where {T<:onemklFloat} =
            oneMKL.trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            B::oneStridedVector{T}) where {T<:Union{Float32,Float64}} =
            oneMKL.trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            B::oneStridedVector{T}) where {T<:Union{ComplexF32,ComplexF64}} =
            oneMKL.trsv!($uploc, 'C', $isunitc, parent(parent(A)), B)
    end
end

for NT in (Number, Real)
    # NOTE: alpha/beta also ::Real to avoid ambiguities with certain Base methods
    @eval begin
        LinearAlgebra.mul!(C::oneStridedMatrix, A::oneStridedVecOrMat, B::oneStridedVecOrMat, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneStridedMatrix, A::Transpose{<:Any, <:oneStridedVecOrMat}, B::oneStridedMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::oneStridedMatrix, B::Transpose{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::Transpose{<:Any, <:oneStridedVecOrMat}, B::Transpose{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneStridedMatrix, A::Adjoint{<:Any, <:oneStridedVecOrMat}, B::oneStridedMatrix, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::oneStridedMatrix, B::Adjoint{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::Adjoint{<:Any, <:oneStridedVecOrMat}, B::Adjoint{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)

        LinearAlgebra.mul!(C::oneStridedMatrix, A::Transpose{<:Any, <:oneStridedVecOrMat}, B::Adjoint{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
        LinearAlgebra.mul!(C::oneStridedMatrix, A::Adjoint{<:Any, <:oneStridedVecOrMat}, B::Transpose{<:Any, <:oneStridedVecOrMat}, a::$NT, b::$NT) =
            gemm_dispatch!(C, A, B, a, b)
    end
end
