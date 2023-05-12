# interfacing with LinearAlgebra standard library

import LinearAlgebra
using LinearAlgebra: Transpose, Adjoint,
                     Hermitian, Symmetric,
                     LowerTriangular, UnitLowerTriangular,
                     UpperTriangular, UnitUpperTriangular,
                     MulAddMul

if isdefined(LinearAlgebra, :wrap) # i.e., VERSION >= v"1.10.0-DEV.1365"
    using LinearAlgebra: wrap
else
    function wrap(A::AbstractVecOrMat, tA::AbstractChar)
        if tA == 'N'
            return A
        elseif tA == 'T'
            return transpose(A)
        elseif tA == 'C'
            return adjoint(A)
        elseif tA == 'H'
            return Hermitian(A, :U)
        elseif tA == 'h'
            return Hermitian(A, :L)
        elseif tA == 'S'
            return Symmetric(A, :U)
        else # tA == 's'
            return Symmetric(A, :L)
        end
    end
end
                    
#
# BLAS 1
#

LinearAlgebra.rmul!(x::oneStridedVecOrMat{T}, k::Number) where T<:Union{onemklHalf,onemklFloat} =
    scal!(length(x), T(k), x)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::oneStridedVecOrMat{<:onemklFloat}, k::Real) =
	invoke(LinearAlgebra.rmul!, Tuple{typeof(x), Number}, x, k)

LinearAlgebra.norm(x::oneStridedVecOrMat{<:Union{Float16,ComplexF16,onemklFloat}}) = nrm2(length(x), x)

function LinearAlgebra.dot(x::oneStridedVector{T}, y::oneStridedVector{T}) where T<:Union{Float16, Float32, Float64}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    dot(n, x, y)
end

function LinearAlgebra.dot(x::oneStridedVector{T}, y::oneStridedVector{T}) where T<:Union{ComplexF16,ComplexF32, ComplexF64}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    dotc(n, x, y)
end

function LinearAlgebra.:(*)(transx::Transpose{<:Any,<:oneStridedVector{T}}, y::oneStridedVector{T}) where T <:Union{ComplexF16, ComplexF32, ComplexF64}
    x = transx.parent
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    oneMKL.dotu(n, x, y)
end

LinearAlgebra.BLAS.asum(x::oneStridedVecOrMat{<:onemklFloat}) = asum(length(x), x)

function LinearAlgebra.axpy!(alpha::Number, x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}) where T<:Union{onemklHalf,onemklFloat}
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    axpy!(length(x), alpha, x, y)
end

function LinearAlgebra.axpby!(alpha::Number, x::oneStridedVecOrMat{T}, beta::Number, y::oneStridedVecOrMat{T}) where T<:onemklFloat
    length(x)==length(y) || throw(DimensionMismatch("axpby arguments have lengths $(length(x)) and $(length(y))"))
    axpby!(length(x), alpha, x, beta, y)
end

function LinearAlgebra.rotate!(x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}, c::Number, s::Number) where T<:onemklFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("rotate arguments have lengths $nx and $ny"))
    rot!(nx, x, y, c, s)
end

function LinearAlgebra.reflect!(x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}, c::Number, s::Number) where T<:onemklFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("reflect arguments have lengths $nx and $ny"))
    rot!(nx, x, y, c, s)
    scal!(ny, -one(real(T)), y)
    x, y
end

#
# BLAS 2
#

# TODO: Should there be a LinearAlgebra._generic_matvecmul! that dispatches to gemv!, symv! and hemv! ?
# hermitian
@inline function LinearAlgebra.mul!(y::oneStridedVector{T},
                                    A::Hermitian{T,<:oneStridedMatrix},
                                    x::oneStridedVector{T},
             α::Number, β::Number) where {T<:Union{ComplexF32,ComplexF64}}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return hemv!(A.uplo, alpha, A.data, x, beta, y)
    else
        error("only supports BLAS type, got $T")
    end
end

# symmetric
@inline function LinearAlgebra.mul!(y::oneStridedVector{T},
                                    A::Hermitian{T,<:oneStridedMatrix},
                                    x::oneStridedVector{T},
                                    α::Number, β::Number) where {T<:Union{Float32,Float64}}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return symv!(A.uplo, alpha, A.data, x, beta, y)
    else
        error("only supports BLAS type, got $T")
    end
end

# triangular
## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:oneStridedMatrix},
                            b::oneStridedVector{T}) where {T<:onemklFloat} =
            trmv!($uploc, 'N', $isunitc, parent(A), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:oneStridedMatrix},
                            B::oneStridedVector{T}) where {T<:onemklFloat} =
            trsv!($uploc, 'N', $isunitc, parent(A), B)
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
            trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            b::oneStridedVector{T}) where {T<:Union{Float32,Float64}} =
            trmv!($uploc, 'T', $isunitc, parent(parent(A)), b)
        LinearAlgebra.lmul!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            b::oneStridedVector{T}) where {T<:Union{ComplexF32,ComplexF64}} =
            trmv!($uploc, 'C', $isunitc, parent(parent(A)), b)

        # Left division
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:oneStridedMatrix}},
                            B::oneStridedVector{T}) where {T<:onemklFloat} =
            trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            B::oneStridedVector{T}) where {T<:Union{Float32,Float64}} =
            trsv!($uploc, 'T', $isunitc, parent(parent(A)), B)
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:oneStridedMatrix}},
                            B::oneStridedVector{T}) where {T<:Union{ComplexF32,ComplexF64}} =
            trsv!($uploc, 'C', $isunitc, parent(parent(A)), B)
    end
end


#
# BLAS 3
#

function LinearAlgebra.generic_matmatmul!(C::oneStridedMatrix, tA, tB, A::oneStridedVecOrMat, B::oneStridedVecOrMat, _add::MulAddMul=MulAddMul())
    T = eltype(C)
    alpha, beta = promote(_add.alpha, _add.beta, zero(T))
    mA, nA = size(A, tA == 'N' ? 1 : 2), size(A, tA == 'N' ? 2 : 1)
    mB, nB = size(B, tB == 'N' ? 1 : 2), size(B, tB == 'N' ? 2 : 1)

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

    if all(in(('N', 'T', 'C')), (tA, tB))
        if T <: onemklFloat && eltype(A) == eltype(B) == T && T != Float16 # onemklHgemm is currently not hooked-up
            gemm!(tA, tB, alpha, A, B, beta, C)
        end
    end
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        # TODO: should the gemm part above be included in this branch?
        if (tA == 'S' || tA == 's') && tB == 'N'
            return symm!('L', tA == 'S' ? 'U' : 'L', alpha, A, B, beta, C)
        elseif (tB == 'S' || tB == 's') && tA == 'N'
            return symm!('R', tB == 'S' ? 'U' : 'L', alpha, B, A, beta, C)
        elseif (tA == 'H' || tA == 'h') && tB == 'N'
            return hemm!('L', tA == 'H' ? 'U' : 'L', alpha, A, B, beta, C)
        elseif (tB == 'H' || tB == 'h') && tA == 'N'
            return hemm!('R', tB == 'H' ? 'U' : 'L', alpha, B, A, beta, C)
        end
    else
        error("only supports BLAS type, got $T")
    end
    GPUArrays.generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), alpha, beta)
end

if VERSION < v"1.10.0-DEV.1365"
# catch other functions that are called by LinearAlgebra's mul!
LinearAlgebra.gemm_wrapper!(C::oneStridedMatrix, tA::AbstractChar, tB::AbstractChar, A::oneStridedVecOrMat, B::oneStridedVecOrMat, _add::MulAddMul) =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add)
# disambiguation
LinearAlgebra.gemm_wrapper!(C::oneStridedMatrix{T}, tA::AbstractChar, tB::AbstractChar, A::oneStridedVecOrMat{T}, B::oneStridedVecOrMat{T}, _add::MulAddMul) where {T<:LinearAlgebra.BlasFloat} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add)
function LinearAlgebra.syrk_wrapper!(C::oneStridedMatrix, tA::AbstractChar, A::oneStridedVecOrMat, _add::MulAddMul = MulAddMul())
    if tA == 'T'
        LinearAlgebra.generic_matmatmul!(C, 'T', 'N', A, A, _add)
    else # tA == 'N'
        LinearAlgebra.generic_matmatmul!(C, 'N', 'T', A, A, _add)
    end
end
function LinearAlgebra.herk_wrapper!(C::oneStridedMatrix, tA::AbstractChar, A::oneStridedVecOrMat, _add::MulAddMul = MulAddMul())
    if tA == 'C'
        LinearAlgebra.generic_matmatmul!(C, 'C', 'N', A, A, _add)
    else # tA == 'N'
        LinearAlgebra.generic_matmatmul!(C, 'N', 'C', A, A, _add)
    end
end
end # VERSION

# triangular
## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:oneStridedVecOrMat},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B, B)
        LinearAlgebra.rmul!(A::oneStridedVecOrMat{T},
                            B::$t{T,<:oneStridedVecOrMat}) where {T<:onemklFloat} =
            trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A, A)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:oneStridedVecOrMat},
                            B::oneStridedVecOrMat{T}) where {T<:onemklFloat} =
            trsm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B)
    end
end
