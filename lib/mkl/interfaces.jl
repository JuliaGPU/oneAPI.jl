# interfacing with other packages

using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSR{T}, B::oneVector{T}, _add::MulAddMul) where T <: BlasFloat
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    sparse_gemv!(tA, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSC{T}, B::oneVector{T}, _add::MulAddMul) where T <: BlasReal
    tA = tA in ('S', 's', 'H', 'h') ? 'T' : (tA == 'N' ? 'T' : 'N')
    sparse_gemv!(tA, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}, _add::MulAddMul) where T <: BlasFloat
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    sparse_gemm!(tA, tB, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSC{T}, B::oneMatrix{T}, _add::MulAddMul) where T <: BlasReal
    tA = tA in ('S', 's', 'H', 'h') ? 'T' : (tA == 'N' ? 'T' : 'N')
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    sparse_gemm!(tA, tB, _add.alpha, A, B, _add.beta, C)
end

for SparseMatrixType in (:oneSparseMatrixCSR,)
    @eval begin
        function LinearAlgebra.generic_trimatdiv!(C::oneVector{T}, uploc, isunitc, tfun::Function, A::$SparseMatrixType{T}, B::oneVector{T}) where T <: BlasFloat
            sparse_trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
        end
    end
end
