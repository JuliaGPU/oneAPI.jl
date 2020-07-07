# Contiguous on-device arrays

export oneDeviceArray, oneDeviceVector, oneDeviceMatrix


## construction

# NOTE: we can't support the typical `tuple or series of integer` style construction,
#       because we're currently requiring a trailing pointer argument.

struct oneDeviceArray{T,N,A} <: AbstractArray{T,N}
    shape::Dims{N}
    ptr::LLVMPtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    oneDeviceArray{T,N,A}(shape::Dims{N}, ptr::LLVMPtr{T,A}) where {T,A,N} = new(shape,ptr)
end

const oneDeviceVector = oneDeviceArray{T,1,A} where {T,A}
const oneDeviceMatrix = oneDeviceArray{T,2,A} where {T,A}

# outer constructors, non-parameterized
oneDeviceArray(dims::NTuple{N,<:Integer}, p::LLVMPtr{T,A})                where {T,A,N} = oneDeviceArray{T,N,A}(dims, p)
oneDeviceArray(len::Integer,              p::LLVMPtr{T,A})                where {T,A}   = oneDeviceVector{T,A}((len,), p)

# outer constructors, partially parameterized
oneDeviceArray{T}(dims::NTuple{N,<:Integer},   p::LLVMPtr{T,A}) where {T,A,N} = oneDeviceArray{T,N,A}(dims, p)
oneDeviceArray{T}(len::Integer,                p::LLVMPtr{T,A}) where {T,A}   = oneDeviceVector{T,A}((len,), p)
oneDeviceArray{T,N}(dims::NTuple{N,<:Integer}, p::LLVMPtr{T,A}) where {T,A,N} = oneDeviceArray{T,N,A}(dims, p)
oneDeviceVector{T}(len::Integer,               p::LLVMPtr{T,A}) where {T,A}   = oneDeviceVector{T,A}((len,), p)

# outer constructors, fully parameterized
oneDeviceArray{T,N,A}(dims::NTuple{N,<:Integer}, p::LLVMPtr{T,A}) where {T,A,N} = oneDeviceArray{T,N,A}(Int.(dims), p)
oneDeviceVector{T,A}(len::Integer,               p::LLVMPtr{T,A}) where {T,A}   = oneDeviceVector{T,A}((Int(len),), p)


## getters

Base.pointer(a::oneDeviceArray) = a.ptr
Base.pointer(a::oneDeviceArray, i::Integer) =
    pointer(a) + (i - 1) * Base.elsize(a)

Base.elsize(::Type{<:oneDeviceArray{T}}) where {T} = sizeof(T)
Base.size(g::oneDeviceArray) = g.shape
Base.length(g::oneDeviceArray) = prod(g.shape)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T,A}}, a::oneDeviceArray{T,N,A}) where {T,A,N} = pointer(a)


## indexing intrinsics

# NOTE: these intrinsics are now implemented using plain and simple pointer operations;
#       when adding support for isbits union arrays we will need to implement that here.

# TODO: arrays as allocated by the CUDA APIs are 256-byte aligned. we should keep track of
#       this information, because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@inline function arrayref(A::oneDeviceArray{T}, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    unsafe_load(pointer(A), index, Val(align))
end

@inline function arrayset(A::oneDeviceArray{T}, x::T, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    unsafe_store!(pointer(A), x, index, Val(align))
    return A
end

@inline function const_arrayref(A::oneDeviceArray{T}, index::Int) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    unsafe_cached_load(pointer(A), index, Val(align))
end


## indexing

Base.@propagate_inbounds Base.getindex(A::oneDeviceArray{T}, i1::Int) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::oneDeviceArray{T}, x, i1::Int) where {T} =
    arrayset(A, convert(T,x)::T, i1)

Base.IndexStyle(::Type{<:oneDeviceArray}) = Base.IndexLinear()


## const indexing

"""
    Const(A::oneDeviceArray)

Mark a oneDeviceArray as constant/read-only. The invariant guaranteed is that you will not
modify an oneDeviceArray for the duration of the current kernel.

This API can only be used on devices with compute capability 3.5 or higher.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
struct Const{T,N,AS} <: DenseArray{T,N}
    a::oneDeviceArray{T,N,AS}
end
Base.Experimental.Const(A::oneDeviceArray) = Const(A)

Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.size(C::Const) = size(C.a)
Base.axes(C::Const) = axes(C.a)
Base.@propagate_inbounds Base.getindex(A::Const, i1::Int) = const_arrayref(A.a, i1)

# deprecated
Base.@propagate_inbounds ldg(A::oneDeviceArray, i1::Integer) = const_arrayref(A, Int(i1))


## other

Base.show(io::IO, a::oneDeviceVector) =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show(io::IO, a::oneDeviceArray) =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")

Base.show(io::IO, mime::MIME"text/plain", a::oneDeviceArray) = show(io, a)

@inline function Base.unsafe_view(A::oneDeviceVector{T}, I::Vararg{Base.ViewIndex,1}) where {T}
    ptr = pointer(A, I[1].start)
    len = I[1].stop - I[1].start + 1
    return oneDeviceArray(len, ptr)
end

@inline function Base.iterate(A::oneDeviceArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end
