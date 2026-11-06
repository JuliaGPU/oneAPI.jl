# sparse arrays
#
# GPUArrays implements the sparse formats generically, parametrized on their storage; oneAPI
# names them for `oneVector` storage. The arrays hold no oneMKL state: the operations in
# `wrappers_sparse.jl` describe a matrix to oneMKL for the duration of a call, or for the
# lifetime of an explicitly prepared `oneSparseMatrixHandle`.

export oneSparseVector, oneSparseMatrixCSR, oneSparseMatrixCSC, oneSparseMatrixCOO

using GPUArrays: GPUSparseVector, GPUSparseMatrixCSR, GPUSparseMatrixCSC, GPUSparseMatrixCOO
using SparseArrays: nnz, nonzeros
import Adapt
import Adapt: adapt

"""
    oneSparseVector{Tv,Ti}

A sparse vector with values of type `Tv` and indices of type `Ti` stored in `oneVector`s:
GPUArrays' `GPUSparseVector` with oneAPI storage.
"""
const oneSparseVector{Tv, Ti} = GPUSparseVector{Tv, Ti, <:oneVector{Ti}, <:oneVector{Tv}}

"""
    oneSparseMatrixCSR{Tv,Ti}

A sparse matrix in compressed sparse row format stored in `oneVector`s: GPUArrays'
`GPUSparseMatrixCSR` with oneAPI storage, and oneMKL's native format. Construct one from a
host sparse matrix, a dense matrix or a sparse matrix in another format with
`oneSparseMatrixCSR(A)` or `oneSparseMatrixCSR{Tv,Ti}(A)`, or from its buffers with
`oneSparseMatrixCSR(rowPtr, colVal, nzVal, dims)`, which borrows them.
"""
const oneSparseMatrixCSR{Tv, Ti} = GPUSparseMatrixCSR{Tv, Ti, <:oneVector{Ti}, <:oneVector{Tv}}

"""
    oneSparseMatrixCSC{Tv,Ti}

A sparse matrix in compressed sparse column format stored in `oneVector`s: GPUArrays'
`GPUSparseMatrixCSC` with oneAPI storage, the layout of `SparseMatrixCSC`. oneMKL handles it
as the CSR representation of its transpose, which requires oneMKL 2025.3 or later. See
[`oneSparseMatrixCSR`](@ref) for the constructors.
"""
const oneSparseMatrixCSC{Tv, Ti} = GPUSparseMatrixCSC{Tv, Ti, <:oneVector{Ti}, <:oneVector{Tv}}

"""
    oneSparseMatrixCOO{Tv,Ti}

A sparse matrix in coordinate format, sorted by row, stored in `oneVector`s: GPUArrays'
`GPUSparseMatrixCOO` with oneAPI storage. See [`oneSparseMatrixCSR`](@ref) for the
constructors.
"""
const oneSparseMatrixCOO{Tv, Ti} = GPUSparseMatrixCOO{Tv, Ti, <:oneVector{Ti}, <:oneVector{Tv}}

const oneAbstractSparseMatrix{Tv, Ti} =
    Union{oneSparseMatrixCSR{Tv, Ti}, oneSparseMatrixCSC{Tv, Ti}, oneSparseMatrixCOO{Tv, Ti}}

const oneSparseMatrixAdjOrTrans =
    Union{Transpose{<:Any, <:oneAbstractSparseMatrix}, Adjoint{<:Any, <:oneAbstractSparseMatrix}}

# from buffers, which the array borrows
for (alias, generic) in ((:oneSparseMatrixCSR, :GPUSparseMatrixCSR),
                         (:oneSparseMatrixCSC, :GPUSparseMatrixCSC),
                         (:oneSparseMatrixCOO, :GPUSparseMatrixCOO))
    @eval begin
        $alias(i::oneVector, j::oneVector, v::oneVector, dims::NTuple{2, Integer}) =
            $generic(compat_ptr($generic, i, dims), j, v, dims)
        $alias{Tv, Ti}(i::oneVector{Ti}, j::oneVector{Ti}, v::oneVector{Tv},
                       dims::NTuple{2, Integer}) where {Tv, Ti} =
            $generic(compat_ptr($generic, i, dims), j, v, dims)
    end
end
oneSparseVector(i::oneVector, v::oneVector, n::Integer) = GPUSparseVector(i, v, n)

# oneAPI 2.10 accepted an empty pointer for a compressed matrix without entries; GPUArrays
# requires one entry per row (or column), plus one
function compat_ptr(::Type{S}, ptr::oneVector, dims) where {S <: Union{GPUSparseMatrixCSR, GPUSparseMatrixCSC}}
    isempty(ptr) || return ptr
    n = S <: GPUSparseMatrixCSR ? dims[1] : dims[2]
    Base.depwarn("Constructing a sparse matrix with an empty pointer vector is deprecated; " *
                 "pass a vector of $(n + 1) ones instead.", :compat_ptr)
    return fill!(similar(ptr, n + 1), one(eltype(ptr)))
end
compat_ptr(::Type, ind::oneVector, dims) = ind

# Fields of the oneAPI 2.10 structs that GPUArrays' types do not have. (`A.nnz` was of the
# index type.)
@inline function Base.getproperty(A::oneAbstractSparseMatrix, name::Symbol)
    if name === :nnz
        Base.depwarn("`A.nnz` is deprecated, use `nnz(A)` instead.", :getproperty)
        return SparseArrays.indtype(A)(nnz(A))
    elseif name === :handle
        Base.depwarn("Sparse arrays no longer cache a oneMKL handle; use `oneMKL.oneSparseMatrixHandle(A)`.", :getproperty)
        return nothing
    end
    return getfield(A, name)
end
Base.propertynames(A::oneAbstractSparseMatrix) = (fieldnames(typeof(A))..., :nnz)

# Constructors on the aliases move the input to oneAPI storage, keeping its format and
# types, and then convert on the device.
for (alias, generic) in ((:oneSparseVector, :GPUSparseVector),
                         (:oneSparseMatrixCSR, :GPUSparseMatrixCSR),
                         (:oneSparseMatrixCSC, :GPUSparseMatrixCSC),
                         (:oneSparseMatrixCOO, :GPUSparseMatrixCOO))
    @eval begin
        $alias(A::AbstractArray) = $generic(adapt(oneArray, A))
        $alias{Tv}(A::AbstractArray) where {Tv} = $generic{Tv}(adapt(oneArray, A))
        $alias{Tv, Ti}(A::AbstractArray) where {Tv, Ti} = $generic{Tv, Ti}(adapt(oneArray, A))
    end
end

# densify on the device (the generic constructor converts through the host)
for (N, Sparse) in ((1, :oneSparseVector), (2, :oneAbstractSparseMatrix))
    @eval oneAPI.oneArray{T, $N, B}(A::$Sparse) where {T, B} =
        copyto!(oneArray{T, $N, B}(undef, size(A)), A)
end

"""
    unsafe_free!(A::oneAbstractSparseMatrix)

Free the buffers of `A` without waiting for the garbage collector. A sparse array
constructed from buffers borrows them, so this also frees the storage of every array that
shares a buffer with `A`: only call it on arrays that own their buffers, like the result of
an operation or a conversion.
"""
function oneAPI.unsafe_free!(A::oneAbstractSparseMatrix)
    foreach(unsafe_free!, GPUArrays.sparse_buffers(A))
    return
end
