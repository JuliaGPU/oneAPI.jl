export oneSparseMatrixCSR, oneSparseMatrixCSC, oneSparseMatrixCOO

using ..oneAPI: KernelAdaptor, oneDeviceVector, AS
using LinearAlgebra: Transpose, Adjoint
using SparseArrays: SparseVector, SparseMatrixCSC, nnz, nonzeros
import Adapt
import Adapt: adapt

# The oneMKL sparse types are children of the GPUArrays sparse hierarchy, so all the generic
# functionality from GPUArrays (broadcast, mapreduce, norms, findnz, ...) applies to them.
#
# The `handle` field caches the oneMKL matrix handle. It is created lazily, on the first oneMKL
# operation (see `matrix_handle` in wrappers_sparse.jl), so that generic code can construct and
# convert sparse matrices freely without paying for oneMKL handle set-up, and so that the struct
# can hold element types oneMKL does not support.

# The storage vectors are often shared between matrices (e.g. the output of a single-input
# broadcast reuses the pointer array of its input, and type conversions reuse the vectors whose
# type does not change). Every matrix therefore holds its own reference to the underlying
# memory, so that `unsafe_free!` on one matrix does not free the storage of another.
_own_ref(x::oneVector) = GPUArrays.derive(eltype(x), x, size(x), 0)

mutable struct oneSparseMatrixCSR{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCSR{Tv, Ti}
    handle::Union{Nothing, matrix_handle_t}
    rowPtr::oneVector{Ti}
    colVal::oneVector{Ti}
    nzVal::oneVector{Tv}
    dims::NTuple{2, Int}
    nnz::Ti

    function oneSparseMatrixCSR{Tv, Ti}(
            rowPtr::oneVector{Ti}, colVal::oneVector{Ti}, nzVal::oneVector{Tv},
            dims::NTuple{2, <:Integer}
        ) where {Tv, Ti <: Integer}
        A = new{Tv, Ti}(nothing, _own_ref(rowPtr), _own_ref(colVal), _own_ref(nzVal), Int.(dims), Ti(length(nzVal)))
        return finalizer(sparse_release_matrix_handle, A)
    end
end

mutable struct oneSparseMatrixCSC{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCSC{Tv, Ti}
    handle::Union{Nothing, matrix_handle_t}
    colPtr::oneVector{Ti}
    rowVal::oneVector{Ti}
    nzVal::oneVector{Tv}
    dims::NTuple{2, Int}
    nnz::Ti

    function oneSparseMatrixCSC{Tv, Ti}(
            colPtr::oneVector{Ti}, rowVal::oneVector{Ti}, nzVal::oneVector{Tv},
            dims::NTuple{2, <:Integer}
        ) where {Tv, Ti <: Integer}
        A = new{Tv, Ti}(nothing, _own_ref(colPtr), _own_ref(rowVal), _own_ref(nzVal), Int.(dims), Ti(length(nzVal)))
        return finalizer(sparse_release_matrix_handle, A)
    end
end

mutable struct oneSparseMatrixCOO{Tv, Ti} <: GPUArrays.AbstractGPUSparseMatrixCOO{Tv, Ti}
    handle::Union{Nothing, matrix_handle_t}
    rowInd::oneVector{Ti}
    colInd::oneVector{Ti}
    nzVal::oneVector{Tv}
    dims::NTuple{2, Int}
    nnz::Ti

    function oneSparseMatrixCOO{Tv, Ti}(
            rowInd::oneVector{Ti}, colInd::oneVector{Ti}, nzVal::oneVector{Tv},
            dims::NTuple{2, <:Integer}
        ) where {Tv, Ti <: Integer}
        A = new{Tv, Ti}(nothing, _own_ref(rowInd), _own_ref(colInd), _own_ref(nzVal), Int.(dims), Ti(length(nzVal)))
        return finalizer(sparse_release_matrix_handle, A)
    end
end

const oneAbstractSparseMatrix{Tv, Ti} = Union{
    oneSparseMatrixCSR{Tv, Ti}, oneSparseMatrixCSC{Tv, Ti}, oneSparseMatrixCOO{Tv, Ti},
}
const oneSparseMatrixAdjOrTrans = Union{
    Transpose{<:Any, <:oneAbstractSparseMatrix}, Adjoint{<:Any, <:oneAbstractSparseMatrix},
}

# untyped constructors from device vectors (GPUArrays' generic code relies on these)
oneSparseMatrixCSR(
    rowPtr::oneVector{Ti}, colVal::oneVector{Ti}, nzVal::oneVector{Tv}, dims::NTuple{2, <:Integer}
) where {Tv, Ti <: Integer} = oneSparseMatrixCSR{Tv, Ti}(rowPtr, colVal, nzVal, dims)
oneSparseMatrixCSC(
    colPtr::oneVector{Ti}, rowVal::oneVector{Ti}, nzVal::oneVector{Tv}, dims::NTuple{2, <:Integer}
) where {Tv, Ti <: Integer} = oneSparseMatrixCSC{Tv, Ti}(colPtr, rowVal, nzVal, dims)
oneSparseMatrixCOO(
    rowInd::oneVector{Ti}, colInd::oneVector{Ti}, nzVal::oneVector{Tv}, dims::NTuple{2, <:Integer}
) where {Tv, Ti <: Integer} = oneSparseMatrixCOO{Tv, Ti}(rowInd, colInd, nzVal, dims)

# the storage vectors of a matrix, in the order (pointer/first index, second index, values)
_storage(A::oneSparseMatrixCSR) = (A.rowPtr, A.colVal, A.nzVal)
_storage(A::oneSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal)
_storage(A::oneSparseMatrixCOO) = (A.rowInd, A.colInd, A.nzVal)


## GPUArrays interface

GPUArrays.sparse_array_type(::Type{<:oneSparseMatrixCSR}) = oneSparseMatrixCSR
GPUArrays.sparse_array_type(::Type{<:oneSparseMatrixCSC}) = oneSparseMatrixCSC
GPUArrays.sparse_array_type(::Type{<:oneSparseMatrixCOO}) = oneSparseMatrixCOO

GPUArrays.dense_array_type(::Type{<:oneAbstractSparseMatrix}) = oneArray

GPUArrays.csr_type(::Type{<:Union{oneAbstractSparseMatrix, oneSparseMatrixAdjOrTrans}}) = oneSparseMatrixCSR
GPUArrays.csc_type(::Type{<:Union{oneAbstractSparseMatrix, oneSparseMatrixAdjOrTrans}}) = oneSparseMatrixCSC
GPUArrays.coo_type(::Type{<:Union{oneAbstractSparseMatrix, oneSparseMatrixAdjOrTrans}}) = oneSparseMatrixCOO


## array interface

Base.length(A::oneAbstractSparseMatrix) = prod(A.dims)
Base.size(A::oneAbstractSparseMatrix) = A.dims

# `similar` preserving the sparsity structure
Base.similar(A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCSR{Tv, Ti}(copy(A.rowPtr), copy(A.colVal), similar(A.nzVal), size(A))
Base.similar(A::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCSC{Tv, Ti}(copy(A.colPtr), copy(A.rowVal), similar(A.nzVal), size(A))
Base.similar(A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCOO{Tv, Ti}(copy(A.rowInd), copy(A.colInd), similar(A.nzVal), size(A))

Base.similar(A::oneSparseMatrixCSR{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    oneSparseMatrixCSR{T, Ti}(copy(A.rowPtr), copy(A.colVal), similar(A.nzVal, T), size(A))
Base.similar(A::oneSparseMatrixCSC{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    oneSparseMatrixCSC{T, Ti}(copy(A.colPtr), copy(A.rowVal), similar(A.nzVal, T), size(A))
Base.similar(A::oneSparseMatrixCOO{<:Any, Ti}, ::Type{T}) where {T, Ti} =
    oneSparseMatrixCOO{T, Ti}(copy(A.rowInd), copy(A.colInd), similar(A.nzVal, T), size(A))

# `similar` with a different shape: an empty (all-zero) matrix of the same format
_empty_ptr(::Type{Ti}, len::Integer) where {Ti} = fill!(oneVector{Ti}(undef, len), one(Ti))
Base.similar(::oneSparseMatrixCSR{<:Any, Ti}, ::Type{T}, m::Integer, n::Integer) where {T, Ti} =
    oneSparseMatrixCSR{T, Ti}(_empty_ptr(Ti, m + 1), oneVector{Ti}(undef, 0), oneVector{T}(undef, 0), (m, n))
Base.similar(::oneSparseMatrixCSC{<:Any, Ti}, ::Type{T}, m::Integer, n::Integer) where {T, Ti} =
    oneSparseMatrixCSC{T, Ti}(_empty_ptr(Ti, n + 1), oneVector{Ti}(undef, 0), oneVector{T}(undef, 0), (m, n))
Base.similar(::oneSparseMatrixCOO{<:Any, Ti}, ::Type{T}, m::Integer, n::Integer) where {T, Ti} =
    oneSparseMatrixCOO{T, Ti}(oneVector{Ti}(undef, 0), oneVector{Ti}(undef, 0), oneVector{T}(undef, 0), (m, n))

Base.similar(A::oneAbstractSparseMatrix, ::Type{T}, dims::Dims{2}) where {T} = similar(A, T, dims...)
Base.similar(A::oneAbstractSparseMatrix{Tv}, m::Integer, n::Integer) where {Tv} = similar(A, Tv, m, n)
Base.similar(A::oneAbstractSparseMatrix{Tv}, dims::Dims{2}) where {Tv} = similar(A, Tv, dims...)

# other dimensionalities (e.g. a column slice) are dense
Base.similar(::oneAbstractSparseMatrix, ::Type{T}, dims::Dims) where {T} = oneArray{T}(undef, dims)
Base.similar(A::oneAbstractSparseMatrix{Tv}, dims::Dims) where {Tv} = similar(A, Tv, dims)

# scalar indexing (GPUArrays provides the CSC method)
function Base.getindex(A::oneSparseMatrixCSR{Tv}, i0::Integer, i1::Integer) where {Tv}
    @boundscheck checkbounds(A, i0, i1)
    c1 = Int(A.rowPtr[i0])
    c2 = Int(A.rowPtr[i0 + 1]) - 1
    c1 > c2 && return zero(Tv)
    c1 = searchsortedfirst(A.colVal, i1, c1, c2, Base.Order.Forward)
    (c1 > c2 || A.colVal[c1] != i1) && return zero(Tv)
    return A.nzVal[c1]
end

function Base.getindex(A::oneSparseMatrixCOO{Tv}, i0::Integer, i1::Integer) where {Tv}
    @boundscheck checkbounds(A, i0, i1)
    # COO entries are not guaranteed to be sorted, so search for the entry
    k = findfirst((A.rowInd .== i0) .& (A.colInd .== i1))
    k === nothing && return zero(Tv)
    return A.nzVal[k]
end

# non-scalar indexing goes through a dense copy: the generic fallback would launch a kernel that
# indexes the device-side sparse matrix, which GPUArrays does not implement (yet)
Base.getindex(A::oneAbstractSparseMatrix, ::Colon, ::Colon) = copy(A)
for I in (:Colon, :Integer, :AbstractVector), J in (:Colon, :Integer, :AbstractVector)
    (I === :Integer && J === :Integer) && continue
    (I === :Colon && J === :Colon) && continue
    @eval Base.getindex(A::oneAbstractSparseMatrix, I::$I, J::$J) = oneArray(A)[I, J]
end

# copying
Base.copy(A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCSR{Tv, Ti}(copy(A.rowPtr), copy(A.colVal), copy(A.nzVal), size(A))
Base.copy(A::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCSC{Tv, Ti}(copy(A.colPtr), copy(A.rowVal), copy(A.nzVal), size(A))
Base.copy(A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCOO{Tv, Ti}(copy(A.rowInd), copy(A.colInd), copy(A.nzVal), size(A))

_copy_as(::Type{T}, x::oneVector) where {T} = eltype(x) === T ? copy(x) : T.(x)

# `copyto!` may change the sparsity structure, so it replaces the storage vectors (they may be
# shared with other matrices, e.g. the output of a single-input broadcast reuses the pointer
# array of its input) and drops the oneMKL handle, which refers to the old storage.
function Base.copyto!(dst::oneSparseMatrixCSR{Tv, Ti}, src::oneSparseMatrixCSR) where {Tv, Ti}
    size(dst) == size(src) || throw(ArgumentError("Inconsistent Sparse Matrix size"))
    _invalidate_handle!(dst)
    dst.rowPtr = _copy_as(Ti, src.rowPtr)
    dst.colVal = _copy_as(Ti, src.colVal)
    dst.nzVal = _copy_as(Tv, src.nzVal)
    dst.nnz = src.nnz
    return dst
end
function Base.copyto!(dst::oneSparseMatrixCSC{Tv, Ti}, src::oneSparseMatrixCSC) where {Tv, Ti}
    size(dst) == size(src) || throw(ArgumentError("Inconsistent Sparse Matrix size"))
    _invalidate_handle!(dst)
    dst.colPtr = _copy_as(Ti, src.colPtr)
    dst.rowVal = _copy_as(Ti, src.rowVal)
    dst.nzVal = _copy_as(Tv, src.nzVal)
    dst.nnz = src.nnz
    return dst
end
function Base.copyto!(dst::oneSparseMatrixCOO{Tv, Ti}, src::oneSparseMatrixCOO) where {Tv, Ti}
    size(dst) == size(src) || throw(ArgumentError("Inconsistent Sparse Matrix size"))
    _invalidate_handle!(dst)
    dst.rowInd = _copy_as(Ti, src.rowInd)
    dst.colInd = _copy_as(Ti, src.colInd)
    dst.nzVal = _copy_as(Tv, src.nzVal)
    dst.nnz = src.nnz
    return dst
end

# dense conversion, on the device (sparse .+ dense broadcasts to a dense array)
oneAPI.oneArray(A::Union{oneSparseMatrixCSR{Tv}, oneSparseMatrixCSC{Tv}}) where {Tv} =
    A .+ fill!(similar(A.nzVal, Tv, size(A)), zero(Tv))
oneAPI.oneArray(A::oneSparseMatrixCOO) = oneArray(collect(A))


## interop with SparseArrays

# CPU to GPU
function oneSparseMatrixCSR(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    At = SparseMatrixCSC(transpose(A))
    return oneSparseMatrixCSR{Tv, Ti}(
        oneVector{Ti}(At.colptr), oneVector{Ti}(At.rowval), oneVector{Tv}(At.nzval), size(A)
    )
end
oneSparseMatrixCSC(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    oneSparseMatrixCSC{Tv, Ti}(
    oneVector{Ti}(A.colptr), oneVector{Ti}(A.rowval), oneVector{Tv}(A.nzval), size(A)
)
function oneSparseMatrixCOO(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    row, col, val = findnz(A)
    return oneSparseMatrixCOO{Tv, Ti}(oneVector{Ti}(row), oneVector{Ti}(col), oneVector{Tv}(val), size(A))
end

# transposes of CPU matrices: CSR(Aᵀ) is CSC(A) with the roles of the index arrays swapped
function oneSparseMatrixCSR(t::Transpose{Tv, <:SparseMatrixCSC{Tv, Ti}}) where {Tv, Ti}
    A = parent(t)
    return oneSparseMatrixCSR{Tv, Ti}(
        oneVector{Ti}(A.colptr), oneVector{Ti}(A.rowval), oneVector{Tv}(A.nzval), size(t)
    )
end
function oneSparseMatrixCSR(t::Adjoint{Tv, <:SparseMatrixCSC{Tv, Ti}}) where {Tv, Ti}
    A = parent(t)
    return oneSparseMatrixCSR{Tv, Ti}(
        oneVector{Ti}(A.colptr), oneVector{Ti}(A.rowval), oneVector{Tv}(conj.(A.nzval)), size(t)
    )
end
const SparseMatrixCSCAdjOrTrans = Union{Transpose{<:Any, <:SparseMatrixCSC}, Adjoint{<:Any, <:SparseMatrixCSC}}
oneSparseMatrixCSC(t::SparseMatrixCSCAdjOrTrans) = oneSparseMatrixCSC(SparseMatrixCSC(t))
oneSparseMatrixCOO(t::SparseMatrixCSCAdjOrTrans) = oneSparseMatrixCOO(SparseMatrixCSC(t))

for X in (:oneSparseMatrixCSR, :oneSparseMatrixCSC, :oneSparseMatrixCOO)
    @eval begin
        # sparse vectors are single-column matrices
        $X(v::SparseVector) = $X(SparseMatrixCSC(v))
        # element type conversion
        $X{Tv}(A::SparseMatrixCSC{<:Any, Ti}) where {Tv, Ti} = $X(SparseMatrixCSC{Tv, Ti}(A))
        $X{Tv, Ti}(A::SparseMatrixCSC) where {Tv, Ti} = $X(SparseMatrixCSC{Tv, Ti}(A))
        $X{Tv}(A::Union{SparseVector, SparseMatrixCSCAdjOrTrans}) where {Tv} = $X{Tv}(SparseMatrixCSC(A))
    end
end

# GPU to CPU (GPUArrays provides the CSC method)
function SparseArrays.SparseMatrixCSC(A::oneSparseMatrixCSR)
    m, n = size(A)
    At = SparseMatrixCSC(n, m, Array(A.rowPtr), Array(A.colVal), Array(A.nzVal))
    return SparseMatrixCSC(transpose(At))
end
SparseArrays.SparseMatrixCSC(A::oneSparseMatrixCOO) =
    sparse(Array(A.rowInd), Array(A.colInd), Array(A.nzVal), size(A)...)


## adapt

Adapt.adapt_storage(::Type{oneArray}, xs::SparseMatrixCSC) = oneSparseMatrixCSC(xs)
Adapt.adapt_storage(::Type{<:oneArray{T}}, xs::SparseMatrixCSC) where {T} = oneSparseMatrixCSC{T}(xs)
Adapt.adapt_storage(::Type{oneArray}, xs::oneAbstractSparseMatrix) = xs
Adapt.adapt_storage(::Type{Array}, xs::oneAbstractSparseMatrix) = SparseMatrixCSC(xs)

# device-side counterparts for use in kernels
Adapt.adapt_structure(to::KernelAdaptor, A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti} =
    GPUArrays.GPUSparseDeviceMatrixCSR{
    Tv, Ti, oneDeviceVector{Ti, AS.CrossWorkgroup}, oneDeviceVector{Tv, AS.CrossWorkgroup}, AS.CrossWorkgroup,
}(adapt(to, A.rowPtr), adapt(to, A.colVal), adapt(to, A.nzVal), A.dims, A.nnz)
Adapt.adapt_structure(to::KernelAdaptor, A::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} =
    GPUArrays.GPUSparseDeviceMatrixCSC{
    Tv, Ti, oneDeviceVector{Ti, AS.CrossWorkgroup}, oneDeviceVector{Tv, AS.CrossWorkgroup}, AS.CrossWorkgroup,
}(adapt(to, A.colPtr), adapt(to, A.rowVal), adapt(to, A.nzVal), A.dims, A.nnz)
Adapt.adapt_structure(to::KernelAdaptor, A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti} =
    GPUArrays.GPUSparseDeviceMatrixCOO{
    Tv, Ti, oneDeviceVector{Ti, AS.CrossWorkgroup}, oneDeviceVector{Tv, AS.CrossWorkgroup}, AS.CrossWorkgroup,
}(adapt(to, A.rowInd), adapt(to, A.colInd), adapt(to, A.nzVal), A.dims, A.nnz)


## input/output

for (gpu, cpu) in [:oneSparseMatrixCSR => :SparseMatrixCSC,
                   :oneSparseMatrixCSC => :SparseMatrixCSC,
                   :oneSparseMatrixCOO => :SparseMatrixCSC]
    @eval Base.show(io::IOContext, x::$gpu) =
        show(io, $cpu(x))

    @eval function Base.show(io::IO, mime::MIME"text/plain", S::$gpu)
        xnnz = nnz(S)
        m, n = size(S)
        print(io, m, "×", n, " ", typeof(S), " with ", xnnz, " stored ",
                  xnnz == 1 ? "entry" : "entries")
        if !(m == 0 || n == 0)
            println(io, ":")
            io = IOContext(io, :typeinfo => eltype(S))
            if ndims(S) == 1
                show(io, $cpu(S))
            else
                # so that we get the nice Braille pattern
                Base.print_array(io, $cpu(S))
            end
        end
    end
end
