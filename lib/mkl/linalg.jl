# interfacing with LinearAlgebra standard library

import LinearAlgebra

# GPUArrays implements LinearAlgebra's products, triangular operations and BLAS 1 functions,
# and calls these hooks for the operations that oneMKL supports.

#
# BLAS 1
#

GPUArrays.blas_scal!(α::Union{Bool,T}, x::oneStridedVecOrDenseMat{T}) where {T<:Union{onemklHalf,onemklFloat}} =
    scal!(length(x), T(α), x)
GPUArrays.blas_scal!(α::Real, x::oneStridedVecOrDenseMat{T}) where {T<:onemklComplex} =
    scal!(length(x), real(T)(α), x)
GPUArrays.blas_dot(x::oneStridedVecOrDenseMat{T}, y::oneStridedVecOrDenseMat{T}) where {T<:Union{Float16,Float32,Float64}} =
    dot(length(x), x, y)
GPUArrays.blas_dot(x::oneStridedVecOrDenseMat{T}, y::oneStridedVecOrDenseMat{T}) where {T<:Union{ComplexF16,onemklComplex}} =
    dotc(length(x), x, y)
GPUArrays.blas_dotu(x::oneStridedVecOrDenseMat{T}, y::oneStridedVecOrDenseMat{T}) where {T<:Union{ComplexF16,onemklComplex}} =
    dotu(length(x), x, y)
GPUArrays.blas_nrm2(x::oneStridedVecOrDenseMat{<:Union{Float16,ComplexF16,onemklFloat}}) = nrm2(length(x), x)
GPUArrays.blas_asum(x::oneStridedVecOrDenseMat{<:onemklFloat}) = asum(length(x), x)
GPUArrays.blas_axpy!(α::Number, x::oneStridedVecOrDenseMat{T}, y::oneStridedVecOrDenseMat{T}) where {T<:Union{onemklHalf,onemklFloat}} =
    axpy!(length(x), α, x, y)
GPUArrays.blas_axpby!(α::Number, x::oneStridedVecOrDenseMat{T}, β::Number, y::oneStridedVecOrDenseMat{T}) where {T<:onemklFloat} =
    axpby!(length(x), α, x, β, y)
GPUArrays.blas_rot!(x::oneStridedVecOrDenseMat{T}, y::oneStridedVecOrDenseMat{T}, c::Real, s::Number) where {T<:onemklFloat} =
    rot!(length(x), x, y, c, s)

#
# BLAS 2
#

GPUArrays.blas_gemv!(tA::AbstractChar, α::Union{Bool,T}, A::oneStridedMatrix{T}, x::oneStridedVector{T},
                     β::Union{Bool,T}, y::oneStridedVector{T}) where {T<:onemklFloat} =
    gemv!(Char(tA), α, A, x, β, y)
# complex symv is not part of BLAS
GPUArrays.blas_symv!(uplo::AbstractChar, α::Union{Bool,T}, A::oneStridedMatrix{T}, x::oneStridedVector{T},
                     β::Union{Bool,T}, y::oneStridedVector{T}) where {T<:Union{Float32,Float64}} =
    symv!(Char(uplo), α, A, x, β, y)
GPUArrays.blas_hemv!(uplo::AbstractChar, α::Union{Bool,T}, A::oneStridedMatrix{T}, x::oneStridedVector{T},
                     β::Union{Bool,T}, y::oneStridedVector{T}) where {T<:onemklComplex} =
    hemv!(Char(uplo), α, A, x, β, y)
GPUArrays.blas_trmv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     A::oneStridedMatrix{T}, x::oneStridedVector{T}) where {T<:onemklFloat} =
    trmv!(Char(uplo), Char(trans), Char(diag), A, x)
GPUArrays.blas_trsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     A::oneStridedMatrix{T}, x::oneStridedVector{T}) where {T<:onemklFloat} =
    trsv!(Char(uplo), Char(trans), Char(diag), A, x)

#
# BLAS 3
#

function GPUArrays.blas_gemm!(tA::AbstractChar, tB::AbstractChar, α::Number, A::oneStridedVecOrMat,
                              B::oneStridedVecOrMat, β::Number, C::oneStridedVecOrMat)
    T = eltype(C)
    if T <: Union{onemklHalf,onemklFloat} && eltype(A) == eltype(B) == T &&
       α isa Union{Bool,T} && β isa Union{Bool,T}
        return gemm!(Char(tA), Char(tB), α, A, B, β, C)
    end
    return GPUArrays.generic_matmatmul!(C, tA, tB, A, B, α, β)
end
GPUArrays.blas_symm!(side::AbstractChar, uplo::AbstractChar, α::Union{Bool,T}, A::oneStridedMatrix{T},
                     B::oneStridedMatrix{T}, β::Union{Bool,T}, C::oneStridedMatrix{T}) where {T<:onemklFloat} =
    symm!(Char(side), Char(uplo), α, A, B, β, C)
GPUArrays.blas_hemm!(side::AbstractChar, uplo::AbstractChar, α::Union{Bool,T}, A::oneStridedMatrix{T},
                     B::oneStridedMatrix{T}, β::Union{Bool,T}, C::oneStridedMatrix{T}) where {T<:onemklComplex} =
    hemm!(Char(side), Char(uplo), α, A, B, β, C)
# C === B uses the in-place routine, otherwise the out-of-place variant writes straight into C
function GPUArrays.blas_trmm!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                              α::Union{Bool,T}, A::oneStridedMatrix{T}, B::oneStridedMatrix{T},
                              C::oneStridedMatrix{T}) where {T<:onemklFloat}
    side, uplo, trans, diag = Char(side), Char(uplo), Char(trans), Char(diag)
    C === B ? trmm!(side, uplo, trans, diag, α, A, C) : trmm!(side, uplo, trans, diag, α, A, B, C)
end
GPUArrays.blas_trsm!(side::AbstractChar, uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                     α::Union{Bool,T}, A::oneStridedMatrix{T}, B::oneStridedMatrix{T}) where {T<:onemklFloat} =
    trsm!(Char(side), Char(uplo), Char(trans), Char(diag), α, A, B)
GPUArrays.blas_syrk!(uplo::AbstractChar, trans::AbstractChar, α::Union{Bool,T}, A::oneStridedMatrix{T},
                     β::Union{Bool,T}, C::oneStridedMatrix{T}) where {T<:onemklFloat} =
    syrk!(Char(uplo), Char(trans), α, A, β, C)
GPUArrays.blas_herk!(uplo::AbstractChar, trans::AbstractChar, α::Real, A::oneStridedMatrix{T},
                     β::Real, C::oneStridedMatrix{T}) where {T<:onemklComplex} =
    herk!(Char(uplo), Char(trans), real(T)(α), A, real(T)(β), C)
GPUArrays.blas_dgmm!(side::AbstractChar, A::oneStridedMatrix{T}, x::oneStridedVector{T},
                     C::oneStridedMatrix{T}) where {T<:onemklFloat} =
    dgmm!(Char(side), A, x, C)

# GPUArrays' dot doesn't cover these wrappers, and LinearAlgebra's indexes them elementwise
LinearAlgebra.dot(A::Symmetric{<:Union{Real,Complex},<:oneMatrix}, B::Symmetric{<:Union{Real,Complex},<:oneMatrix}) =
    GPUArrays.gpu_dot(A, B)
LinearAlgebra.dot(A::Hermitian{<:Union{Real,Complex},<:oneMatrix}, B::Hermitian{<:Union{Real,Complex},<:oneMatrix}) =
    GPUArrays.gpu_dot(A, B)

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
