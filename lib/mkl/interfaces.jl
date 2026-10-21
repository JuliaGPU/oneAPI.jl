# interfacing with other packages

using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, MulAddMul

# LinearAlgebra's storage-level `mul!` passes a `Symmetric` or `Hermitian` operand as its
# parent plus 'S'/'s'/'H'/'h'. Apart from the symmetric mat-vec below, oneMKL has no
# symmetric or Hermitian sparse products, so materialize the full matrix the wrapper
# represents. (issue #718)
function _unwrap_symmherm(t::AbstractChar, A)
    uppercase(t) in ('S', 'H') || return t, A
    return 'N', _materialize_symmherm(LinearAlgebra.wrap(A, t))
end
_materialize_symmherm(A::HermOrSym{<:Any, <:oneAbstractSparseMatrix}) = sparse(A)
function _materialize_symmherm(A::HermOrSym{<:Any, <:oneMatrix})
    # `oneMatrix(A)` would go through the CPU
    B = LinearAlgebra.copytri!(copy(parent(A)), A.uplo, A isa Hermitian)
    if A isa Hermitian
        d = view(B, diagind(B))
        d .= real.(d)
    end
    return B
end

# oneMKL's sparse symv only reads the referenced triangle, so it also handles real Hermitian
# matrices, but not complex ones (it doesn't conjugate the mirrored triangle)
_use_symv(tA::AbstractChar, ::Type{T}) where {T} = tA in ('S', 's') || (T <: Real && tA in ('H', 'h'))

function LinearAlgebra.mul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSR{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    _use_symv(tA, T) && return sparse_symv!(isuppercase(tA) ? 'U' : 'L', alpha, A, B, beta, C)
    tA, A = _unwrap_symmherm(tA, A)
    return sparse_gemv!(tA, alpha, A, B, beta, C)
end

function LinearAlgebra.mul!(C::oneVector{T}, tA::AbstractChar, A::oneSparseMatrixCSC{T}, B::oneVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    # sparse_gemv! already maps op(A) onto the transposed CSR handle, so tA is passed through
    _use_symv(tA, T) && return sparse_symv!(isuppercase(tA) ? 'U' : 'L', alpha, A, B, beta, C)
    tA, A = _unwrap_symmherm(tA, A)
    return sparse_gemv!(tA, alpha, A, B, beta, C)
end

function LinearAlgebra.mul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSR{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    tA, A = _unwrap_symmherm(tA, A)
    tB, B = _unwrap_symmherm(tB, B)
    return sparse_gemm!(tA, tB, alpha, A, B, beta, C)
end

function LinearAlgebra.mul!(C::oneMatrix{T}, tA, tB, A::oneSparseMatrixCSC{T}, B::oneMatrix{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    # sparse_gemm! already maps op(A) onto the transposed CSR handle, so tA is passed through
    tA, A = _unwrap_symmherm(tA, A)
    tB, B = _unwrap_symmherm(tB, B)
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
