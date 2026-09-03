# interfacing with other packages

using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul

function LinearAlgebra.mul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSR{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    return sparse_gemv!(tA, alpha, A, B, beta, C)
end

function LinearAlgebra.mul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSC{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    # sparse_gemv! already maps op(A) onto the transposed CSR handle, so tA is passed through
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    return sparse_gemv!(tA, alpha, A, B, beta, C)
end

function LinearAlgebra.mul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    return sparse_gemm!(tA, tB, alpha, A, B, beta, C)
end

function LinearAlgebra.mul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSC{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    # sparse_gemm! already maps op(A) onto the transposed CSR handle, so tA is passed through
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    return sparse_gemm!(tA, tB, alpha, A, B, beta, C)
end

# Julia < 1.13 dispatches on the non-public `generic_matvecmul!` and `generic_matmatmul!`,
# which JuliaLang/LinearAlgebra.jl#1671 superseded by the `mul!` methods above. Forward from
# the old names, both the alpha/beta variants (1.12) and the ones taking a final MulAddMul
# (1.10 and 1.11).
@static if VERSION < v"1.13.0-rc4"
    for SparseMatrixType in (:oneSparseMatrixCSR, :oneSparseMatrixCSC)
        @eval begin
            LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::$SparseMatrixType{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat} =
                LinearAlgebra.mul!(C, tA, A, B, alpha, beta)
            LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::$SparseMatrixType{T}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat} =
                LinearAlgebra.mul!(C, tA, A, B, _add.alpha, _add.beta)
            LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::$SparseMatrixType{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat} =
                LinearAlgebra.mul!(C, tA, tB, A, B, alpha, beta)
            LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::$SparseMatrixType{T}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat} =
                LinearAlgebra.mul!(C, tA, tB, A, B, _add.alpha, _add.beta)
        end
    end
end

function LinearAlgebra.generic_trimatdiv!(C::oneVector{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneVector{T}) where {T <: BlasFloat}
    return sparse_trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
end

function LinearAlgebra.generic_trimatdiv!(C::oneMatrix{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}) where {T <: BlasFloat}
    return sparse_trsm!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', isunitc, one(T), A, B, C)
end
