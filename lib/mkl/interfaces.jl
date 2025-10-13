# interfacing with other packages

using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul

# legacy methods with final MulAddMul argument
LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSR{T}, B::oneVector{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}} =
    LinearAlgebra.generic_matvecmul!(C, tA, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSC{T}, B::oneVector{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}} =
    LinearAlgebra.generic_matvecmul!(C, tA, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)
LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSC{T}, B::oneMatrix{T}, _add::MulAddMul) where {T <: Union{Float16, ComplexF16, BlasFloat}} =
    LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, _add.alpha, _add.beta)

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSR{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    return sparse_gemv!(tA, alpha, A, B, beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSC{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasReal}
    tA = tA in ('S', 's', 'H', 'h') ? 'T' : flip_trans(tA)
    return sparse_gemv!(tA, alpha, A, B, beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    return sparse_gemm!(tA, tB, alpha, A, B, beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSC{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasReal}
    tA = tA in ('S', 's', 'H', 'h') ? 'T' : flip_trans(tA)
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    return sparse_gemm!(tA, tB, alpha, A, B, beta, C)
end

function LinearAlgebra.generic_trimatdiv!(C::oneVector{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneVector{T}) where {T <: BlasFloat}
    return sparse_trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
end

function LinearAlgebra.generic_trimatdiv!(C::oneMatrix{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}) where {T <: BlasFloat}
    return sparse_trsm!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', isunitc, one(T), A, B, C)
end
