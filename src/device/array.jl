# Contiguous on-device arrays

export oneDeviceArray, oneDeviceVector, oneDeviceMatrix


## construction

# NOTE: we can't support the typical `tuple or series of integer` style construction,
#       because we're currently requiring a trailing pointer argument.

struct oneDeviceArray{T,N,A} <: DenseArray{T,N}
    ptr::LLVMPtr{T,A}
    maxsize::Int

    dims::Dims{N}
    len::Int

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    # TODO: deprecate; put `ptr` first like oneArray
    oneDeviceArray{T,N,A}(dims::Dims{N}, ptr::LLVMPtr{T,A},
                          maxsize::Int=prod(dims)*sizeof(T)) where {T,A,N} =
        new(ptr, maxsize, dims, prod(dims))
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


## array interface

Base.elsize(::Type{<:oneDeviceArray{T}}) where {T} = sizeof(T)

Base.size(g::oneDeviceArray) = g.dims
Base.sizeof(x::oneDeviceArray) = Base.elsize(x) * length(x)

# we store the array length too; computing prod(size) is expensive
Base.length(g::oneDeviceArray) = g.len

Base.pointer(x::oneDeviceArray{T,<:Any,A}) where {T,A} = Base.unsafe_convert(LLVMPtr{T,A}, x)
@inline function Base.pointer(x::oneDeviceArray{T,<:Any,A}, i::Integer) where {T,A}
    Base.unsafe_convert(LLVMPtr{T,A}, x) + Base._memory_offset(x, i)
end

typetagdata(a::oneDeviceArray{<:Any,<:Any,A}, i=1) where {A} =
  reinterpret(LLVMPtr{UInt8,A}, a.ptr + a.maxsize) + i - one(i)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T,A}}, x::oneDeviceArray{T,<:Any,A}) where {T,A} =
  x.ptr


## indexing intrinsics

# TODO: how are allocations aligned by the level zero API? keep track of this
#       because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@generated function alignment(::oneDeviceArray{T}) where {T}
    if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
end

@device_function @inline function arrayref(A::oneDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if isbitstype(T)
        arrayref_bits(A, index)
    else #if isbitsunion(T)
        arrayref_union(A, index)
    end
end

@inline function arrayref_bits(A::oneDeviceArray{T}, index::Integer) where {T}
    align = alignment(A)
    unsafe_load(pointer(A), index, Val(align))
end

@inline @generated function arrayref_union(A::oneDeviceArray{T,<:Any,AS}, index::Integer) where {T,AS}
    typs = Base.uniontypes(T)

    # generate code that conditionally loads a value based on the selector value.
    # lacking noreturn, we return T to avoid inference thinking this can return Nothing.
    ex = :(Base.llvmcall("unreachable", $T, Tuple{}))
    for (sel, typ) in Iterators.reverse(enumerate(typs))
        ex = quote
            if selector == $(sel-1)
                ptr = reinterpret(LLVMPtr{$typ,AS}, data_ptr)
                unsafe_load(ptr, 1, Val(align))
            else
                $ex
            end
        end
    end

    quote
        selector_ptr = typetagdata(A, index)
        selector = unsafe_load(selector_ptr)

        align = alignment(A)
        data_ptr = pointer(A, index)

        return $ex
    end
end

@device_function @inline function arrayset(A::oneDeviceArray{T}, x::T, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if isbitstype(T)
        arrayset_bits(A, x, index)
    else #if isbitsunion(T)
        arrayset_union(A, x, index)
    end
    return A
end

@inline function arrayset_bits(A::oneDeviceArray{T}, x::T, index::Integer) where {T}
    align = alignment(A)
    unsafe_store!(pointer(A), x, index, Val(align))
end

@inline @generated function arrayset_union(A::oneDeviceArray{T,<:Any,AS}, x::T, index::Integer) where {T,AS}
    typs = Base.uniontypes(T)
    sel = findfirst(isequal(x), typs)

    quote
        selector_ptr = typetagdata(A, index)
        unsafe_store!(selector_ptr, $(UInt8(sel-1)))

        align = alignment(A)
        data_ptr = pointer(A, index)

        unsafe_store!(reinterpret(LLVMPtr{$x,AS}, data_ptr), x, 1, Val(align))
        return
    end
end

@device_function @inline function const_arrayref(A::oneDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = alignment(A)
    unsafe_cached_load(pointer(A), index, Val(align))
end


## indexing

Base.IndexStyle(::Type{<:oneDeviceArray}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(A::oneDeviceArray{T}, i1::Integer) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::oneDeviceArray{T}, x, i1::Integer) where {T} =
    arrayset(A, convert(T,x)::T, i1)

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::oneDeviceArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
Base.@propagate_inbounds Base.getindex(A::oneDeviceArray,
                                       I::Union{Integer, CartesianIndex}...) =
    A[Base._to_linear_index(A, to_indices(A, I)...)]
Base.@propagate_inbounds Base.setindex!(A::oneDeviceArray, x,
                                        I::Union{Integer, CartesianIndex}...) =
    A[Base._to_linear_index(A, to_indices(A, I)...)] = x


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
Base.@propagate_inbounds Base.getindex(A::Const, i1::Integer) = const_arrayref(A.a, i1)

# deprecated
Base.@propagate_inbounds ldg(A::oneDeviceArray, i1::Integer) = const_arrayref(A, i1)


## other

Base.show(io::IO, a::oneDeviceVector) =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show(io::IO, a::oneDeviceArray) =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")

Base.show(io::IO, mime::MIME"text/plain", a::oneDeviceArray) = show(io, a)

@inline function Base.iterate(A::oneDeviceArray, i=1)
    if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end

function Base.reinterpret(::Type{T}, a::oneDeviceArray{S,N,A}) where {T,S,N,A}
  err = _reinterpret_exception(T, a)
  err === nothing || throw(err)

  if sizeof(T) == sizeof(S) # fast case
    return oneDeviceArray{T,N,A}(size(a), reinterpret(LLVMPtr{T,A}, a.ptr), a.maxsize)
  end

  isize = size(a)
  size1 = div(isize[1]*sizeof(S), sizeof(T))
  osize = tuple(size1, Base.tail(isize)...)
  return oneDeviceArray{T,N,A}(osize, reinterpret(LLVMPtr{T,A}, a.ptr), a.maxsize)
end
