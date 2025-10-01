# interfacing with other packages

using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSR{T}, B::oneVector{T}, _add::MulAddMul) where T <: BlasFloat
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    sparse_gemv!(tA, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSC{T}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    return sparse_gemv!(tA, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCOO{T}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    sparse_gemv!(tA, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}, _add::MulAddMul) where T <: BlasFloat
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    sparse_gemm!(tA, tB, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSC{T}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    return sparse_gemm!(tA, tB, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCOO{T}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    sparse_gemm!(tA, tB, _add.alpha, A, B, _add.beta, C)
end

function LinearAlgebra.generic_trimatdiv!(C::oneVector{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneVector{T}) where T <: BlasFloat
    sparse_trsv!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', isunitc, one(T), A, B, C)
end

function LinearAlgebra.generic_trimatdiv!(C::oneMatrix{T}, uploc, isunitc, tfun::Function, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}) where T <: BlasFloat
    sparse_trsm!(uploc, tfun === identity ? 'N' : tfun === transpose ? 'T' : 'C', 'N', isunitc, one(T), A, B, C)
end

# Handle Transpose and Adjoint wrappers for sparse matrices
# Let the low-level wrappers handle the CSC->CSR conversion and flip_trans logic

# Matrix-vector multiplication with transpose/adjoint
function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Transpose{T, <:oneSparseMatrixCSR{T}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tA_final = tA == 'N' ? 'T' : (tA == 'T' ? 'N' : 'C')
    return sparse_gemv!(tA_final, _add.alpha, A.parent, B, _add.beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Adjoint{T, <:oneSparseMatrixCSR{T}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    if tA == 'T'
        alpha = _add.alpha
        beta = _add.beta
        B .= conj.(B)
        C .= conj.(C)
        sparse_gemv!('N', conj(alpha), A.parent, B, conj(beta), C)
        C .= conj.(C)
        B .= conj.(B)
    else
        tA_final = tA == 'N' ? 'C' : 'N'
        sparse_gemv!(tA_final, _add.alpha, A.parent, B, _add.beta, C)
    end
    return C
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Transpose{T, <:oneSparseMatrixCSC{T}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tA_final = tA == 'N' ? 'T' : (tA == 'T' ? 'N' : 'C')
    return sparse_gemv!(tA_final, _add.alpha, A.parent, B, _add.beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Adjoint{T, <:oneSparseMatrixCSC{T}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    if tA == 'T'
        alpha = _add.alpha
        beta = _add.beta
        B .= conj.(B)
        C .= conj.(C)
        sparse_gemv!('N', conj(alpha), A.parent, B, conj(beta), C)
        C .= conj.(C)
        B .= conj.(B)
    else
        tA_final = tA == 'N' ? 'C' : 'N'
        sparse_gemv!(tA_final, _add.alpha, A.parent, B, _add.beta, C)
    end
    return C
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Transpose{T, <:oneSparseMatrixCOO{T}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tA_final = tA == 'N' ? 'T' : (tA == 'T' ? 'N' : 'C')
    return sparse_gemv!(tA_final, _add.alpha, A.parent, B, _add.beta, C)
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Adjoint{T, <:oneSparseMatrixCOO{T}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    if tA == 'T'
        alpha = _add.alpha
        beta = _add.beta
        B .= conj.(B)
        C .= conj.(C)
        sparse_gemv!('N', conj(alpha), A.parent, B, conj(beta), C)
        C .= conj.(C)
        B .= conj.(B)
    else
        tA_final = tA == 'N' ? 'C' : 'N'
        sparse_gemv!(tA_final, _add.alpha, A.parent, B, _add.beta, C)
    end
    return C
end

# Handle Transpose{T, Adjoint{T, ...}} for complex matrices
# transpose(adjoint(A)) for complex matrices needs special handling
function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Transpose{T, <:Adjoint{T, <:oneSparseMatrixCSR{T}}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasComplex}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    # transpose(adjoint(A)) = conj(A), so we need to conjugate
    alpha = _add.alpha
    beta = _add.beta
    B .= conj.(B)
    C .= conj.(C)
    if tA == 'N'
        sparse_gemv!('N', conj(alpha), A.parent.parent, B, conj(beta), C)
    elseif tA == 'T'
        sparse_gemv!('T', conj(alpha), A.parent.parent, B, conj(beta), C)
    else # tA == 'C'
        sparse_gemv!('C', conj(alpha), A.parent.parent, B, conj(beta), C)
    end
    C .= conj.(C)
    B .= conj.(B)
    return C
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Transpose{T, <:Adjoint{T, <:oneSparseMatrixCSC{T}}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasComplex}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    # transpose(adjoint(A)) = conj(A), so we need to conjugate
    alpha = _add.alpha
    beta = _add.beta
    B .= conj.(B)
    C .= conj.(C)
    if tA == 'N'
        sparse_gemv!('N', conj(alpha), A.parent.parent, B, conj(beta), C)
    elseif tA == 'T'
        sparse_gemv!('T', conj(alpha), A.parent.parent, B, conj(beta), C)
    else # tA == 'C'
        sparse_gemv!('C', conj(alpha), A.parent.parent, B, conj(beta), C)
    end
    C .= conj.(C)
    B .= conj.(B)
    return C
end

function LinearAlgebra.generic_matvecmul!(C::oneVector{T}, tA::AbstractChar, A::Transpose{T, <:Adjoint{T, <:oneSparseMatrixCOO{T}}}, B::oneVector{T}, _add::MulAddMul) where {T <: BlasComplex}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    # transpose(adjoint(A)) = conj(A), so we need to conjugate
    alpha = _add.alpha
    beta = _add.beta
    B .= conj.(B)
    C .= conj.(C)
    if tA == 'N'
        sparse_gemv!('N', conj(alpha), A.parent.parent, B, conj(beta), C)
    elseif tA == 'T'
        sparse_gemv!('T', conj(alpha), A.parent.parent, B, conj(beta), C)
    else # tA == 'C'
        sparse_gemv!('C', conj(alpha), A.parent.parent, B, conj(beta), C)
    end
    C .= conj.(C)
    B .= conj.(B)
    return C
end

# Custom * operators for Transpose{T, Adjoint{T, ...}} to ensure correct output size allocation
function Base.:*(A::Transpose{T, <:Adjoint{T, <:oneSparseMatrixCSR{T}}}, x::oneVector{T}) where {T <: BlasComplex}
    m, n = size(A)
    y = similar(x, T, m)
    LinearAlgebra.generic_matvecmul!(y, 'N', A, x, LinearAlgebra.MulAddMul(one(T), zero(T)))
    return y
end

function Base.:*(A::Transpose{T, <:Adjoint{T, <:oneSparseMatrixCSC{T}}}, x::oneVector{T}) where {T <: BlasComplex}
    m, n = size(A)
    y = similar(x, T, m)
    LinearAlgebra.generic_matvecmul!(y, 'N', A, x, LinearAlgebra.MulAddMul(one(T), zero(T)))
    return y
end

function Base.:*(A::Transpose{T, <:Adjoint{T, <:oneSparseMatrixCOO{T}}}, x::oneVector{T}) where {T <: BlasComplex}
    m, n = size(A)
    y = similar(x, T, m)
    LinearAlgebra.generic_matvecmul!(y, 'N', A, x, LinearAlgebra.MulAddMul(one(T), zero(T)))
    return y
end

# Matrix-matrix multiplication with transpose/adjoint
function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::Transpose{T, <:oneSparseMatrixCSR{T}}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    tA_final = tA == 'N' ? 'T' : (tA == 'T' ? 'N' : 'C')
    return sparse_gemm!(tA_final, tB, _add.alpha, A.parent, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::Adjoint{T, <:oneSparseMatrixCSR{T}}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    if tA == 'T'
        alpha = _add.alpha
        beta = _add.beta
        B .= conj.(B)
        C .= conj.(C)
        sparse_gemm!('N', tB, conj(alpha), A.parent, B, conj(beta), C)
        C .= conj.(C)
        B .= conj.(B)
    else
        tA_final = tA == 'N' ? 'C' : 'N'
        sparse_gemm!(tA_final, tB, _add.alpha, A.parent, B, _add.beta, C)
    end
    return C
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::Transpose{T, <:oneSparseMatrixCSC{T}}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    tA_final = tA == 'N' ? 'T' : (tA == 'T' ? 'N' : 'C')
    return sparse_gemm!(tA_final, tB, _add.alpha, A.parent, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::Adjoint{T, <:oneSparseMatrixCSC{T}}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    if tA == 'T'
        alpha = _add.alpha
        beta = _add.beta
        B .= conj.(B)
        C .= conj.(C)
        sparse_gemm!('N', tB, conj(alpha), A.parent, B, conj(beta), C)
        C .= conj.(C)
        B .= conj.(B)
    else
        tA_final = tA == 'N' ? 'C' : 'N'
        sparse_gemm!(tA_final, tB, _add.alpha, A.parent, B, _add.beta, C)
    end
    return C
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::Transpose{T, <:oneSparseMatrixCOO{T}}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    tA_final = tA == 'N' ? 'T' : (tA == 'T' ? 'N' : 'C')
    return sparse_gemm!(tA_final, tB, _add.alpha, A.parent, B, _add.beta, C)
end

function LinearAlgebra.generic_matmatmul!(C::oneMatrix{T}, tA, tB, A::Adjoint{T, <:oneSparseMatrixCOO{T}}, B::oneMatrix{T}, _add::MulAddMul) where {T <: BlasFloat}
    tA = tA in ('S', 's', 'H', 'h') ? 'N' : tA
    tB = tB in ('S', 's', 'H', 'h') ? 'N' : tB
    if tA == 'T'
        alpha = _add.alpha
        beta = _add.beta
        B .= conj.(B)
        C .= conj.(C)
        sparse_gemm!('N', tB, conj(alpha), A.parent, B, conj(beta), C)
        C .= conj.(C)
        B .= conj.(B)
    else
        tA_final = tA == 'N' ? 'C' : 'N'
        sparse_gemm!(tA_final, tB, _add.alpha, A.parent, B, _add.beta, C)
    end
    return C
end
