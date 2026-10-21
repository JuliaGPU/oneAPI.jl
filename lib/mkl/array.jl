# Ordinary formats are canonical GPUArrays storage, without native handles.
export oneSparseMatrixCSR, oneSparseMatrixCSC, oneSparseMatrixCOO, PreparedSparse
using GPUArrays: GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO
using SparseArrays: nnz, nonzeros
import Adapt
import Adapt: adapt
const oneSparseMatrixCSR{T,Ti} = GPUSparseMatrixCSR{T,Ti,<:oneVector{<:Integer},<:oneVector{Ti},<:oneVector{T}}
const oneSparseMatrixCSC{T,Ti} = GPUSparseMatrixCSC{T,Ti,<:oneVector{<:Integer},<:oneVector{Ti},<:oneVector{T}}
const oneSparseMatrixCOO{T,Ti} = GPUSparseMatrixCOO{T,Ti,<:oneVector{Ti},<:oneVector{T}}
const oneAbstractSparseMatrix{T,Ti} = Union{oneSparseMatrixCSR{T,Ti},oneSparseMatrixCSC{T,Ti},oneSparseMatrixCOO{T,Ti}}
const oneSparseMatrixAdjOrTrans = Union{Transpose{<:Any,<:oneAbstractSparseMatrix},Adjoint{<:Any,<:oneAbstractSparseMatrix}}
_storage(A::oneAbstractSparseMatrix) = GPUArrays.sparse_buffers(A)

for (alias,generic) in ((:oneSparseMatrixCSR,:GPUSparseMatrixCSR),
                         (:oneSparseMatrixCSC,:GPUSparseMatrixCSC),
                         (:oneSparseMatrixCOO,:GPUSparseMatrixCOO))
    @eval begin
        $alias(p::oneVector,i::oneVector,v::oneVector,dims::NTuple{2,Integer}) = $generic(p,i,v,dims)
        $alias{T}(p::oneVector,i::oneVector,v::oneVector,dims::NTuple{2,Integer}) where {T} =
            GPUArrays.with_values($generic(p,i,v,dims),T)
        $alias{T,Ti}(p::oneVector,i::oneVector,v::oneVector,dims::NTuple{2,Integer}) where {T,Ti} =
            GPUArrays.with_eltypes($generic(p,i,v,dims),T,Ti)
        $alias(A::AbstractArray) = $generic(adapt(oneArray,A))
        $alias{T}(A::AbstractArray) where {T} = $generic{T}(adapt(oneArray,A))
        $alias{T,Ti}(A::AbstractArray) where {T,Ti} = $generic{T,Ti}(adapt(oneArray,A))
    end
end

# Device densification never collects or iterates sparse host scalars.
oneAPI.oneArray(A::oneAbstractSparseMatrix) = copyto!(oneArray{eltype(A)}(undef,size(A)),A)
oneAPI.oneArray{T}(A::oneAbstractSparseMatrix) where {T} = copyto!(oneArray{T}(undef,size(A)),A)

"""
    unsafe_free!(A::oneAbstractSparseMatrix)

Release component buffers only when exclusively owned and not borrowed by a prepared
operation, view or device representation. Raw sparse construction borrows buffers.
"""
function oneAPI.unsafe_free!(A::oneAbstractSparseMatrix)
    foreach(unsafe_free!,_storage(A))
    nothing
end
