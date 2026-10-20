# interfacing with LinearAlgebra standard library

import LinearAlgebra
using LinearAlgebra: Transpose, Adjoint,
                     Hermitian, Symmetric,
                     LowerTriangular, UnitLowerTriangular,
                     UpperTriangular, UnitUpperTriangular,
                     MulAddMul, wrap

# oneMKL addresses a matrix, or a vector passed as one, through its leading dimension
# `stride(X, 2)`, which must be at least the number of rows. Reversed views fail this.
column_major(X::oneStridedVecOrMat) = stride(X, 1) == 1 && stride(X, 2) >= max(1, size(X, 1))

# BLAS walks a vector with a negative increment backwards from its end, while a reversed
# view points at its first element, so vector operands need a positive stride.
blas_layout(x::oneStridedVector) = stride(x, 1) > 0
blas_layout(A::oneStridedMatrix) = column_major(A)

# run a oneMKL routine on contiguous copies of operands it can't address
blas_input(X) = blas_layout(X) ? X : copy(X)
blas_output(f, C) = blas_layout(C) ? f(C) : copyto!(C, f(similar(C)))

#
# BLAS 1
#

# oneMKL can only scale arrays that it can address as a vector
LinearAlgebra.rmul!(x::oneStridedVecOrMat{T}, k::Number) where T<:Union{onemklHalf,onemklFloat} =
    x isa oneStridedVecOrDenseMat ? scal!(length(x), T(k), x) : (x .*= k)

# Work around ambiguity with GPUArrays wrapper
LinearAlgebra.rmul!(x::oneStridedVecOrMat{<:onemklFloat}, k::Real) =
	invoke(LinearAlgebra.rmul!, Tuple{typeof(x), Number}, x, k)

LinearAlgebra.norm(x::oneStridedVecOrDenseMat{<:Union{Float16,ComplexF16,onemklFloat}}) = nrm2(length(x), x)

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

LinearAlgebra.BLAS.asum(x::oneStridedVecOrDenseMat{<:onemklFloat}) = asum(length(x), x)

function LinearAlgebra.axpy!(alpha::Number, x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}) where T<:Union{onemklHalf,onemklFloat}
    length(x)==length(y) || throw(DimensionMismatch("axpy arguments have lengths $(length(x)) and $(length(y))"))
    if x isa oneStridedVecOrDenseMat && y isa oneStridedVecOrDenseMat
        axpy!(length(x), alpha, x, y)
    else
        vec(y) .+= vec(x) .* alpha
    end
    y
end

function LinearAlgebra.axpby!(alpha::Number, x::oneStridedVecOrMat{T}, beta::Number, y::oneStridedVecOrMat{T}) where T<:onemklFloat
    length(x)==length(y) || throw(DimensionMismatch("axpby arguments have lengths $(length(x)) and $(length(y))"))
    if x isa oneStridedVecOrDenseMat && y isa oneStridedVecOrDenseMat
        axpby!(length(x), alpha, x, beta, y)
    else
        vec(y) .= vec(x) .* alpha .+ vec(y) .* beta
    end
    y
end

function LinearAlgebra.rotate!(x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}, c::Number, s::Number) where T<:onemklFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("rotate arguments have lengths $nx and $ny"))
    if x isa oneStridedVecOrDenseMat && y isa oneStridedVecOrDenseMat
        rot!(nx, x, y, c, s)
    else
        xv, yv = vec(x), vec(y)
        t = c .* xv .+ s .* yv
        yv .= c .* yv .- conj(s) .* xv
        xv .= t
    end
    x, y
end

function LinearAlgebra.reflect!(x::oneStridedVecOrMat{T}, y::oneStridedVecOrMat{T}, c::Number, s::Number) where T<:onemklFloat
    nx = length(x)
    ny = length(y)
    nx==ny || throw(DimensionMismatch("reflect arguments have lengths $nx and $ny"))
    if x isa oneStridedVecOrDenseMat && y isa oneStridedVecOrDenseMat
        rot!(nx, x, y, c, s)
        scal!(ny, -one(real(T)), y)
    else
        xv, yv = vec(x), vec(y)
        t = c .* xv .+ s .* yv
        yv .= conj(s) .* xv .- c .* yv
        xv .= t
    end
    x, y
end

#
# BLAS 2

function LinearAlgebra.mul!(Y::oneStridedVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, a::Number, b::Number)
    mA, nA = tA == 'N' ? size(A) : reverse(size(A))

    if nA != length(B)
        throw(DimensionMismatch("second dimension of A, $nA, does not match length of B, $(length(B))"))
    end

    if mA != length(Y)
        throw(DimensionMismatch("first dimension of A, $mA, does not match length of Y, $(length(Y))"))
    end

    T = eltype(Y)

    # an empty inner dimension still needs to apply beta: Y := b*Y
    if mA == 0 || nA == 0
        return iszero(b) ? fill!(Y, zero(T)) : rmul!(Y, b)
    end

    alpha, beta = promote(a, b, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        if T <: onemklFloat && eltype(A) == eltype(B) == T &&
           blas_layout(A) && blas_layout(B) && blas_layout(Y)
            if tA in ('N', 'T', 'C')
                return gemv!(tA, alpha, A, B, beta, Y)
            elseif tA in ('S', 's') && T <: Real
                # complex symv! is not wrapped; fall through to the matrix-matrix
                # `mul!`, which can use symm! instead
                return symv!(tA == 'S' ? 'U' : 'L', alpha, A, B, beta, Y)
            elseif tA in ('H', 'h')
                # hemv! only supports complex eltypes, but a real Hermitian matrix
                # is symmetric
                fun = T <: Real ? symv! : hemv!
                return fun(tA == 'H' ? 'U' : 'L', alpha, A, B, beta, Y)
            end
        end
    end
    return LinearAlgebra.mul!(Y, tA, 'N', A, B, alpha, beta)
end

# triangular
## multiplication
LinearAlgebra.generic_trimatmul!(c::oneStridedVector{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, b::oneStridedVector{T}) where {T<:onemklFloat} =
    blas_output(c) do c
        trmv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, blas_input(A), c === b ? c : copyto!(c, b))
    end
## division
LinearAlgebra.generic_trimatdiv!(C::oneStridedVector{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedVector{T}) where {T<:onemklFloat} =
    blas_output(C) do C
        trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, blas_input(A), C === B ? C : copyto!(C, B))
    end


#
# BLAS 3
#

function LinearAlgebra.mul!(
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

    # an empty inner dimension still needs to apply beta: C := beta*C
    if mA == 0 || nA == 0 || nB == 0
        size(C) != (mA, nB) && throw(
            DimensionMismatch(
                "C has dimensions $(size(C)), should have ($mA,$nB)"
            )
        )
        return iszero(beta) ? fill!(C, zero(T)) : rmul!(C, beta)
    end

    mkl_layout = column_major(A) && column_major(B) && column_major(C)
    if mkl_layout && T <: Union{onemklFloat, onemklComplex, onemklHalf} &&
            alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        # TODO: should the gemm part above be included in this branch?
        α, β = T(alpha), T(beta)
        if (
                all(in(('N', 'T', 'C')), (tA, tB)) && T <: Union{onemklFloat, onemklComplex, onemklHalf} &&
                    A isa oneStridedArray{T} && B isa oneStridedArray{T}
            )
            return gemm!(tA, tB, α, A, B, β, C)
        elseif T <: onemklFloat && A isa oneStridedArray{T} && B isa oneStridedArray{T}
            # hemm! only supports complex eltypes, but a real Hermitian matrix
            # is symmetric
            if (tA == 'S' || tA == 's') && tB == 'N'
                return symm!('L', tA == 'S' ? 'U' : 'L', α, A, B, β, C)
            elseif (tB == 'S' || tB == 's') && tA == 'N'
                return symm!('R', tB == 'S' ? 'U' : 'L', α, B, A, β, C)
            elseif (tA == 'H' || tA == 'h') && tB == 'N'
                fun = T <: Real ? symm! : hemm!
                return fun('L', tA == 'H' ? 'U' : 'L', α, A, B, β, C)
            elseif (tB == 'H' || tB == 'h') && tA == 'N'
                fun = T <: Real ? symm! : hemm!
                return fun('R', tB == 'H' ? 'U' : 'L', α, B, A, β, C)
            end
        end
    end

    GPUArrays.generic_matmatmul!(C, wrap(A, tA), wrap(B, tB), alpha, beta)
end

# Julia < 1.13 dispatches on the non-public `generic_matvecmul!` and `generic_matmatmul!`,
# which JuliaLang/LinearAlgebra.jl#1671 superseded by the `mul!` methods above. Forward from
# the old names, both the alpha/beta variants (1.12) and the ones taking a final MulAddMul
# (1.10 and 1.11).
@static if VERSION < v"1.13.0-rc4"
    LinearAlgebra.generic_matvecmul!(Y::oneStridedVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, alpha::Number, beta::Number) =
        LinearAlgebra.mul!(Y, tA, A, B, alpha, beta)
    LinearAlgebra.generic_matvecmul!(Y::oneStridedVector, tA::AbstractChar, A::oneStridedMatrix, B::oneStridedVector, _add::MulAddMul) =
        LinearAlgebra.mul!(Y, tA, A, B, _add.alpha, _add.beta)
    LinearAlgebra.generic_matmatmul!(C::oneStridedVecOrMat, tA, tB, A::oneStridedVecOrMat, B::oneStridedVecOrMat, alpha::Number, beta::Number) =
        LinearAlgebra.mul!(C, tA, tB, A, B, alpha, beta)
    LinearAlgebra.generic_matmatmul!(C::oneStridedVecOrMat, tA, tB, A::oneStridedVecOrMat, B::oneStridedVecOrMat, _add::MulAddMul) =
        LinearAlgebra.mul!(C, tA, tB, A, B, _add.alpha, _add.beta)
end

# triangular
#
# `lmul!`/`rmul!`/`ldiv!`/`rdiv!` reach these with the output aliasing the non-triangular
# operand; that case stays on the in-place oneMKL routines. Otherwise the out-of-place
# "variants" write straight into C without first copying the operand over.
LinearAlgebra.generic_trimatmul!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    blas_output(C) do C
        C === B ? trmm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(A), C) :
                  trmm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(A), blas_input(B), C)
    end
LinearAlgebra.generic_mattrimul!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    blas_output(C) do C
        C === A ? trmm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(B), C) :
                  trmm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(B), blas_input(A), C)
    end
LinearAlgebra.generic_trimatdiv!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    blas_output(C) do C
        C === B ? trsm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(A), C) :
                  trsm!('L', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(A), blas_input(B), C)
    end
LinearAlgebra.generic_mattridiv!(C::oneStridedMatrix{T}, uploc, isunitc, tfun::Function, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    blas_output(C) do C
        C === A ? trsm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(B), C) :
                  trsm!('R', uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), blas_input(B), blas_input(A), C)
    end

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
