# conversions between sparse formats, transposition, and sparse addition
#
# Everything here is built from generic GPUArrays operations (broadcast, sortperm, gather), so
# it runs on the device for any element type and does not involve oneMKL. Sorting uses plain
# integer keys, which is the configuration AcceleratedKernels' sortperm is reliable for.

# the first index (row for CSR, column for CSC) of every stored entry, from a pointer array
function _expand_ptr(ptr::oneVector{Ti}, nnzA::Integer) where {Ti}
    nnzA == 0 && return similar(ptr, 0)
    return Ti.(searchsortedlast.(Ref(ptr), oneVector{Ti}(1:nnzA)))
end

# compress (rows, cols, vals) triplets of an nrows×ncols matrix into a pointer-array layout:
# returns (ptr, idx, vals) with the entries sorted by row, then by column
function _compress(
        rows::oneVector{Ti}, cols::oneVector{Ti}, vals::oneVector, nrows::Integer, ncols::Integer
    ) where {Ti}
    nnzA = length(vals)
    if nnzA == 0
        return _empty_ptr(Ti, nrows + 1), similar(cols, 0), similar(vals, 0)
    end
    key = (Int.(rows) .- 1) .* Int(ncols) .+ Int.(cols)
    perm = sortperm(key)
    sorted_rows = rows[perm]
    ptr = Ti.(searchsortedfirst.(Ref(sorted_rows), oneVector{Ti}(1:(nrows + 1))))
    return ptr, cols[perm], vals[perm]
end

# transpose an m×n matrix in pointer-array layout (CSR: ptr=rowPtr, idx=colVal), yielding the
# n×m transpose in the same layout. Since CSC(A) has the same layout as CSR(Aᵀ), this also
# converts between the CSR and CSC formats.
function _transpose_csr(ptr::oneVector{Ti}, idx::oneVector{Ti}, vals::oneVector, m::Integer, n::Integer) where {Ti}
    return _compress(idx, _expand_ptr(ptr, length(vals)), vals, n, m)
end


## conversions

oneSparseMatrixCSR(A::oneSparseMatrixCSR) = A
oneSparseMatrixCSC(A::oneSparseMatrixCSC) = A
oneSparseMatrixCOO(A::oneSparseMatrixCOO) = A

# conversion of the element and index types, on the device
_convert_vector(::Type{T}, x::oneVector) where {T} = eltype(x) === T ? x : T.(x)
function _with_types(A::oneAbstractSparseMatrix, ::Type{Tv}, ::Type{Ti}) where {Tv, Ti}
    p, i, v = _storage(A)
    (eltype(v) === Tv && eltype(p) === Ti) && return A
    return GPUArrays.sparse_array_type(A){Tv, Ti}(
        _convert_vector(Ti, p), _convert_vector(Ti, i), _convert_vector(Tv, v), size(A)
    )
end

# typed constructors convert the format and the types (GPUArrays uses these, e.g. in `triu`)
for X in (:oneSparseMatrixCSR, :oneSparseMatrixCSC, :oneSparseMatrixCOO)
    @eval begin
        $X{Tv, Ti}(A::oneAbstractSparseMatrix) where {Tv, Ti} = _with_types($X(A), Tv, Ti)
        $X{Tv}(A::oneAbstractSparseMatrix{<:Any, Ti}) where {Tv, Ti} = _with_types($X(A), Tv, Ti)
    end
end

function oneSparseMatrixCSC(A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSC{Tv, Ti}(_transpose_csr(A.rowPtr, A.colVal, A.nzVal, m, n)..., (m, n))
end
function oneSparseMatrixCSR(A::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSR{Tv, Ti}(_transpose_csr(A.colPtr, A.rowVal, A.nzVal, n, m)..., (m, n))
end

function oneSparseMatrixCOO(A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rowInd = _expand_ptr(A.rowPtr, nnz(A))
    return oneSparseMatrixCOO{Tv, Ti}(rowInd, copy(A.colVal), copy(A.nzVal), size(A))
end
# via CSR so that the entries end up in row-major order
oneSparseMatrixCOO(A::oneSparseMatrixCSC) = oneSparseMatrixCOO(oneSparseMatrixCSR(A))

function oneSparseMatrixCSR(A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSR{Tv, Ti}(_compress(A.rowInd, A.colInd, A.nzVal, m, n)..., (m, n))
end
function oneSparseMatrixCSC(A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSC{Tv, Ti}(_compress(A.colInd, A.rowInd, A.nzVal, n, m)..., (m, n))
end


## transposition

function GPUArrays._sptranspose(A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSR{Tv, Ti}(_transpose_csr(A.rowPtr, A.colVal, A.nzVal, m, n)..., (n, m))
end
function GPUArrays._spadjoint(A::oneSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSR{Tv, Ti}(_transpose_csr(A.rowPtr, A.colVal, conj.(A.nzVal), m, n)..., (n, m))
end

function GPUArrays._sptranspose(A::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSC{Tv, Ti}(_transpose_csr(A.colPtr, A.rowVal, A.nzVal, n, m)..., (n, m))
end
function GPUArrays._spadjoint(A::oneSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCSC{Tv, Ti}(_transpose_csr(A.colPtr, A.rowVal, conj.(A.nzVal), n, m)..., (n, m))
end

function GPUArrays._sptranspose(A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    # re-sort so that the result is in row-major order
    return oneSparseMatrixCOO(oneSparseMatrixCSR{Tv, Ti}(_compress(A.colInd, A.rowInd, A.nzVal, n, m)..., (n, m)))
end
function GPUArrays._spadjoint(A::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    return oneSparseMatrixCOO(oneSparseMatrixCSR{Tv, Ti}(_compress(A.colInd, A.rowInd, conj.(A.nzVal), n, m)..., (n, m)))
end

# materializing lazy wrappers of device matrices
for X in (:oneSparseMatrixCSR, :oneSparseMatrixCSC, :oneSparseMatrixCOO)
    @eval begin
        $X(t::Transpose{<:Any, <:oneAbstractSparseMatrix}) = $X(GPUArrays._sptranspose(parent(t)))
        $X(t::Adjoint{<:Any, <:oneAbstractSparseMatrix}) = $X(GPUArrays._spadjoint(parent(t)))
    end
end


## addition and subtraction

# GPUArrays implements these through broadcasting over matrices of the same format (e.g. for
# `issymmetric`, which computes `A - transpose(A)`), so wrapped operands are materialized first.
_materialize(A::oneAbstractSparseMatrix) = A
_materialize(t::Transpose{<:Any, <:oneAbstractSparseMatrix}) = GPUArrays._sptranspose(parent(t))
_materialize(t::Adjoint{<:Any, <:oneAbstractSparseMatrix}) = GPUArrays._spadjoint(parent(t))

for op in (:+, :-), X in (:oneSparseMatrixCSR, :oneSparseMatrixCSC)
    @eval begin
        Base.$op(A::$X, B::$X) = broadcast($op, A, B)
        Base.$op(A::$X, B::Union{Transpose{<:Any, <:$X}, Adjoint{<:Any, <:$X}}) =
            broadcast($op, A, _materialize(B))
        Base.$op(A::Union{Transpose{<:Any, <:$X}, Adjoint{<:Any, <:$X}}, B::$X) =
            broadcast($op, _materialize(A), B)
    end
end


## structural operations on COO matrices
#
# GPUArrays implements triu/tril/kron/reshape/droptol! for CSR and CSC matrices by converting
# to the COO format and back, so the actual work happens here.

# keep the entries of `A` selected by `mask`
function _select(A::oneSparseMatrixCOO{Tv, Ti}, mask::AbstractVector{Bool}) where {Tv, Ti}
    return oneSparseMatrixCOO{Tv, Ti}(A.rowInd[mask], A.colInd[mask], A.nzVal[mask], size(A))
end

LinearAlgebra.triu(A::oneSparseMatrixCOO, k::Integer = 0) = _select(A, A.rowInd .+ k .<= A.colInd)
LinearAlgebra.tril(A::oneSparseMatrixCOO, k::Integer = 0) = _select(A, A.rowInd .+ k .>= A.colInd)

function SparseArrays.droptol!(A::oneSparseMatrixCOO, tol::Real)
    mask = abs.(A.nzVal) .> tol
    _invalidate_handle!(A)
    A.rowInd = A.rowInd[mask]
    A.colInd = A.colInd[mask]
    A.nzVal = A.nzVal[mask]
    A.nnz = length(A.nzVal)
    return A
end

function Base.reshape(A::oneSparseMatrixCOO{Tv, Ti}, dims::Dims{2}) where {Tv, Ti}
    prod(dims) == length(A) || throw(DimensionMismatch("new dimensions $dims must be consistent with array size $(length(A))"))
    m = size(A, 1)
    m2 = dims[1]
    # column-major linear index of every entry, re-interpreted in the new shape
    linear = (Int.(A.colInd) .- 1) .* m .+ Int.(A.rowInd)
    rowInd = Ti.(mod1.(linear, m2))
    colInd = Ti.(fld1.(linear, m2))
    return oneSparseMatrixCOO{Tv, Ti}(rowInd, colInd, copy(A.nzVal), dims)
end

# Diagonal matrices as COO, for kron
function oneSparseMatrixCOO(D::Diagonal{Tv}) where {Tv}
    n = size(D, 1)
    ind = oneVector{Int}(1:n)
    return oneSparseMatrixCOO{Tv, Int}(ind, copy(ind), oneVector{Tv}(D.diag), (n, n))
end

function LinearAlgebra.kron(A::oneSparseMatrixCOO{Tv, Ti}, B::oneSparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    mA, nA = size(A)
    mB, nB = size(B)
    nnzA, nnzB = Int(nnz(A)), Int(nnz(B))
    # every entry of A is paired with every entry of B; entries of A vary slowest
    Ar = repeat(A.rowInd; inner = nnzB)
    Ac = repeat(A.colInd; inner = nnzB)
    Av = repeat(A.nzVal; inner = nnzB)
    Br = repeat(B.rowInd; outer = nnzA)
    Bc = repeat(B.colInd; outer = nnzA)
    Bv = repeat(B.nzVal; outer = nnzA)
    rowInd = (Ar .- one(Ti)) .* Ti(mB) .+ Br
    colInd = (Ac .- one(Ti)) .* Ti(nB) .+ Bc
    return oneSparseMatrixCOO{Tv, Ti}(rowInd, colInd, Av .* Bv, (mA * mB, nA * nB))
end
function LinearAlgebra.kron(A::oneSparseMatrixCOO{Tv, Ti}, B::oneSparseMatrixCOO) where {Tv, Ti}
    T = promote_type(Tv, eltype(B))
    return kron(_with_types(A, T, Ti), _with_types(B, T, Ti))
end
LinearAlgebra.kron(A::oneSparseMatrixCOO, D::Diagonal) = kron(A, oneSparseMatrixCOO(D))
LinearAlgebra.kron(D::Diagonal, A::oneSparseMatrixCOO) = kron(oneSparseMatrixCOO(D), A)
