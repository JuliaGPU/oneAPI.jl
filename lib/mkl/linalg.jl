# interfacing with LinearAlgebra standard library

import LinearAlgebra
using LinearAlgebra: Transpose, Adjoint,
                     Hermitian, Symmetric,
                     LowerTriangular, UnitLowerTriangular,
                     UpperTriangular, UnitUpperTriangular,
                     MulAddMul, wrap

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

LinearAlgebra.generic_matvecmul!(Y::oneVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, _add::MulAddMul) =
    LinearAlgebra.generic_matvecmul!(Y, tA, A, B, _add.alpha, _add.beta)
function LinearAlgebra.generic_matvecmul!(Y::oneVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, a::Number, b::Number)
    mA, nA = tA == 'N' ? size(A) : reverse(size(A))

    if nA != length(B)
        throw(DimensionMismatch("second dimension of A, $nA, does not match length of B, $(length(B))"))
    end

    if mA != length(Y)
        throw(DimensionMismatch("first dimension of A, $mA, does not match length of Y, $(length(Y))"))
    end

    if mA == 0
        return Y
    end

    if nA == 0
        return rmul!(Y, 0)
    end

    T = eltype(Y)
    alpha, beta = promote(a, b, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        if T <: onemklFloat && eltype(A) == eltype(B) == T
            if tA in ('N', 'T', 'C')
                return gemv!(tA, alpha, A, B, beta, Y)
            elseif tA in ('S', 's')
                return symv!(tA == 'S' ? 'U' : 'L', alpha, A, x, beta, y)
            elseif tA in ('H', 'h')
                return hemv!(tA == 'H' ? 'U' : 'L', alpha, A, x, beta, y)
            end
        end
    end
    return LinearAlgebra.generic_matmatmul!(Y, tA, 'N', A, B, alpha, beta)
end

# triangular
## multiplication
LinearAlgebra.generic_trimatmul!(c::oneStridedVector{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, b::oneStridedVector{T}) where {T<:onemklFloat} =
    trmv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, A, c === b ? c : copyto!(c, b))
## division
LinearAlgebra.generic_trimatdiv!(C::oneStridedVector{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedVector{T}) where {T<:onemklFloat} =
    trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, A, C === B ? C : copyto!(C, B))


#
# BLAS 3
#

LinearAlgebra.generic_matmatmul!(
    C::oneStridedVecOrMat, tA, tB, A::oneStridedVecOrMat,
    B::oneStridedVecOrMat, _add::MulAddMul,
) = LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
function LinearAlgebra.generic_matmatmul!(
        C::oneStridedVecOrMat, tA, tB, A::oneStridedVecOrMat,
        B::oneStridedVecOrMat, alpha::Number, beta::Number,
    )
    T = eltype(C)
    mA, nA = size(A, tA == 'N' ? 1 : 2), size(A, tA == 'N' ? 2 : 1)
    mB, nB = size(B, tB == 'N' ? 1 : 2), size(B, tB == 'N' ? 2 : 1)

    nA != mB && throw(
        DimensionMismatch(
            "A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"
        )
    )
    (C === A || B === C) && throw(
        ArgumentError(
            "output matrix must not be aliased with input matrix"
        )
    )

    if mA == 0 || nA == 0 || nB == 0
        size(C) != (mA, nB) && throw(
            DimensionMismatch(
                "C has dimensions $(size(C)), should have ($mA,$nB)"
            )
        )
        return LinearAlgebra.rmul!(C, 0)
    end

    T = eltype(C)

    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        # TODO: should the gemm part above be included in this branch?
        α, β = T(alpha), T(beta)
        if (
                all(in(('N', 'T', 'C')), (tA, tB)) && T <: Union{onemklFloat, onemklComplex, onemklHalf} &&
                    A isa oneStridedArray{T} && B isa oneStridedArray{T}
            )
            return gemm!(tA, tB, α, A, B, β, C)
        elseif (tA == 'S' || tA == 's') && tB == 'N'
            return symm!('L', tA == 'S' ? 'U' : 'L', α, A, B, β, C)
        elseif (tB == 'S' || tB == 's') && tA == 'N'
            return symm!('R', tB == 'S' ? 'U' : 'L', α, B, A, β, C)
        elseif (tA == 'H' || tA == 'h') && tB == 'N'
            return hemm!('L', tA == 'H' ? 'U' : 'L', α, A, B, β, C)
        elseif (tB == 'H' || tB == 'h') && tA == 'N'
            return hemm!('R', tB == 'H' ? 'U' : 'L', α, B, A, β, C)
        end
    end

    GPUArrays.generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), alpha, beta)
end

# triangular
LinearAlgebra.generic_trimatmul!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trmm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, C === B ? C : copyto!(C, B))
LinearAlgebra.generic_mattrimul!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trmm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, C === A ? C : copyto!(C, A))
LinearAlgebra.generic_trimatdiv!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trsm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, C === B ? C : copyto!(C, B))
LinearAlgebra.generic_mattridiv!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trsm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, C === A ? C : copyto!(C, A))

#
# BLAS extensions
#

# Extend LinearAlgebra.BLAS.herk! to dispatch to oneAPI implementation
for (elty) in ([Float32, ComplexF32], [Float64, ComplexF64])
    @eval begin
        LinearAlgebra.BLAS.herk!(uplo::Char, trans::Char, alpha::$elty[1], A::oneStridedVecOrMat{$elty[2]}, beta::$elty[1], C::oneStridedMatrix{$elty[2]}) =
            herk!(uplo, trans, alpha, A, beta, C)
    end
end

# Extend LinearAlgebra.BLAS.syrk! to dispatch to oneAPI implementation
for (elty) in (Float32, Float64, ComplexF32, ComplexF64)
    @eval begin
        LinearAlgebra.BLAS.syrk!(uplo::Char, trans::Char, alpha::$elty, A::oneStridedVecOrMat{$elty}, beta::$elty, C::oneStridedMatrix{$elty}) =
            syrk!(uplo, trans, alpha, A, beta, C)
    end
end
