# interfacing with other packages

using LinearAlgebra: BlasComplex, BlasFloat, BlasReal

# GPUArrays implements the sparse products for any storage, through LinearAlgebra's
# storage-level `mul!` (which it also forwards to from the `generic_matvecmul!` and
# `generic_matmatmul!` that Julia < 1.13 calls). The methods below use oneMKL where it
# applies, and the generic implementations otherwise.

# oneMKL's sparse symv only reads the referenced triangle, so it also handles real Hermitian
# matrices, but not complex ones (it doesn't conjugate the mirrored triangle)
_use_symv(tA::AbstractChar, ::Type{T}) where {T} = tA in ('S', 's') || (T <: Real && tA in ('H', 'h'))

# whether oneMKL can operate on `A` (CSC matrices are handled as the CSR representation of
# their transpose, which requires a recent oneMKL)
_onemkl_format(A::oneSparseMatrixCSR) = SparseArrays.indtype(A) <: onemklSparseInt
_onemkl_format(A::oneSparseMatrixCSC) = SparseArrays.indtype(A) <: onemklSparseInt && csc_supported()

function LinearAlgebra.mul!(C::oneVector{T}, tA::AbstractChar,
                            A::Union{oneSparseMatrixCSR{T}, oneSparseMatrixCSC{T}}, B::oneVector{T},
                            alpha::Number, beta::Number) where {T <: BlasFloat}
    if _onemkl_format(A)
        tA in ('N', 'T', 'C') && return sparse_gemv!(tA, alpha, A, B, beta, C)
        _use_symv(tA, T) && return sparse_symv!(isuppercase(tA) ? 'U' : 'L', alpha, A, B, beta, C)
    end
    return GPUArrays.generic_spmv!(C, tA, A, B, alpha, beta)
end

function LinearAlgebra.mul!(C::oneMatrix{T}, tA::AbstractChar, tB::AbstractChar,
                            A::Union{oneSparseMatrixCSR{T}, oneSparseMatrixCSC{T}}, B::oneMatrix{T},
                            alpha::Number, beta::Number) where {T <: BlasFloat}
    # oneMKL's sparse gemm does not support transposed dense operands
    if _onemkl_format(A) && tA in ('N', 'T', 'C') && tB == 'N'
        return sparse_gemm!(tA, tB, alpha, A, B, beta, C)
    end
    return GPUArrays.generic_spmm!(C, tA, tB, A, B, alpha, beta)
end

# triangular solves with a non-transposed CSR matrix
function LinearAlgebra.generic_trimatdiv!(C::oneVector{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneVector{T}) where {T <: BlasFloat}
    return sparse_trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
end

function LinearAlgebra.generic_trimatdiv!(C::oneMatrix{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}) where {T <: BlasFloat}
    return sparse_trsm!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', isunitc, one(T), A, B, C)
end
