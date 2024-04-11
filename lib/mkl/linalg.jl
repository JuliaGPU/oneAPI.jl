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

function LinearAlgebra.generic_matvecmul!(Y::oneVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, _add::MulAddMul)
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
    alpha, beta = promote(_add.alpha, _add.beta, zero(T))
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
    LinearAlgebra.generic_matmatmul!(Y, tA, 'N', A, B, MulAddMul(alpha, beta))
end

if VERSION < v"1.10.0-DEV.1365"
@inline LinearAlgebra.gemv!(Y::oneVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, a::Number, b::Number) =
    LinearAlgebra.generic_matvecmul!(Y, tA, A, B, MulAddMul(a, b))
# disambiguation with LinearAlgebra.jl
@inline LinearAlgebra.gemv!(Y::oneVector{T}, tA::AbstractChar, A::oneStridedMatrix{T}, B::oneStridedVector{T}, a::Number, b::Number) where {T<:onemklFloat} =
    LinearAlgebra.generic_matvecmul!(Y, tA, A, B, MulAddMul(a, b))
end

# triangular
if isdefined(LinearAlgebra, :generic_trimatmul!) # VERSION >= v"1.10-DEVXYZ"
# multiplication
LinearAlgebra.generic_trimatmul!(c::oneStridedVector{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, b::AbstractVector{T}) where {T<:onemklFloat} =
    trmv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, A, c === b ? c : copyto!(c, b))
# division
LinearAlgebra.generic_trimatdiv!(C::oneStridedVector{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::AbstractVector{T}) where {T<:onemklFloat} =
    trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, A, C === B ? C : copyto!(C, B))
else
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
end # VERSION


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
        if T <: Union{onemklFloat, onemklComplex, onemklHalf} && eltype(A) == eltype(B) == T
            return gemm!(tA, tB, alpha, A, B, beta, C)
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
if isdefined(LinearAlgebra, :generic_trimatmul!) # VERSION >= v"1.10-DEVXYZ"
LinearAlgebra.generic_trimatmul!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trmm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, C === B ? C : copyto!(C, B))
LinearAlgebra.generic_mattrimul!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trmm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, C === A ? C : copyto!(C, A))
LinearAlgebra.generic_trimatdiv!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trsm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, C === B ? C : copyto!(C, B))
LinearAlgebra.generic_mattridiv!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trsm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), B, C === A ? C : copyto!(C, A))
else
## direct multiplication/division
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Multiplication
        LinearAlgebra.lmul!(A::$t{T,<:oneStridedMatrix},
                            B::oneStridedMatrix{T}) where {T<:onemklFloat} =
            trmm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B)
        LinearAlgebra.rmul!(A::oneStridedMatrix{T},
                            B::$t{T,<:oneStridedMatrix}) where {T<:onemklFloat} =
            trmm!('R', $uploc, 'N', $isunitc, one(T), parent(B), A)

        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:oneStridedMatrix},
                            B::oneStridedMatrix{T}) where {T<:onemklFloat} =
            trsm!('L', $uploc, 'N', $isunitc, one(T), parent(A), B)
    end
end
end # VERSION
