# pointer types

export ZePtr, ZE_NULL, PtrOrZePtr, ZeRef, RefOrZeRef


#
# Device pointer
#

"""
    ZePtr{T}

A memory address that refers to data of type `T` that is accessible from q device. A `ZePtr`
is ABI compatible with regular `Ptr` objects, e.g. it can be used to `ccall` a function that
expects a `Ptr` to device memory, but it prevents erroneous conversions between the two.
"""
ZePtr

if sizeof(Ptr{Cvoid}) == 8
    primitive type ZePtr{T} 64 end
else
    primitive type ZePtr{T} 32 end
end

# constructor
ZePtr{T}(x::Union{Int,UInt,ZePtr}) where {T} = Base.bitcast(ZePtr{T}, x)

const ZE_NULL = ZePtr{Cvoid}(0)


## getters

Base.eltype(::Type{<:ZePtr{T}}) where {T} = T


## conversions

# to and from integers
## pointer to integer
Base.convert(::Type{T}, x::ZePtr) where {T<:Integer} = T(UInt(x))
## integer to pointer
Base.convert(::Type{ZePtr{T}}, x::Union{Int,UInt}) where {T} = ZePtr{T}(x)
Int(x::ZePtr)  = Base.bitcast(Int, x)
UInt(x::ZePtr) = Base.bitcast(UInt, x)

# between regular and oneAPI pointers
Base.convert(::Type{<:Ptr}, p::ZePtr) =
    throw(ArgumentError("cannot convert a device pointer to a host pointer"))

# between oneAPI pointers
Base.convert(::Type{ZePtr{T}}, p::ZePtr) where {T} = Base.bitcast(ZePtr{T}, p)

# defer conversions to unsafe_convert
Base.cconvert(::Type{<:ZePtr}, x) = x

# fallback for unsafe_convert
Base.unsafe_convert(::Type{P}, x::ZePtr) where {P<:ZePtr} = convert(P, x)

# from arrays
Base.unsafe_convert(::Type{ZePtr{S}}, a::AbstractArray{T}) where {S,T} =
    convert(ZePtr{S}, Base.unsafe_convert(ZePtr{T}, a))
Base.unsafe_convert(::Type{ZePtr{T}}, a::AbstractArray{T}) where {T} =
    error("conversion to pointer not defined for $(typeof(a))")

## limited pointer arithmetic & comparison

Base.isequal(x::ZePtr, y::ZePtr) = (x === y)
Base.isless(x::ZePtr{T}, y::ZePtr{T}) where {T} = x < y

Base.:(==)(x::ZePtr, y::ZePtr) = UInt(x) == UInt(y)
Base.:(<)(x::ZePtr,  y::ZePtr) = UInt(x) < UInt(y)
Base.:(-)(x::ZePtr,  y::ZePtr) = UInt(x) - UInt(y)

Base.:(+)(x::ZePtr, y::Integer) = oftype(x, Base.add_ptr(UInt(x), (y % UInt) % UInt))
Base.:(-)(x::ZePtr, y::Integer) = oftype(x, Base.sub_ptr(UInt(x), (y % UInt) % UInt))
Base.:(+)(x::Integer, y::ZePtr) = y + x



#
# Host or device pointer
#

"""
    PtrOrZePtr{T}

A special pointer type, ABI-compatible with both `Ptr` and `ZePtr`, for use in `ccall`
expressions to convert values to either a device or a host type (in that order). This is
required for APIs which accept pointers that either point to host or device memory.
"""
PtrOrZePtr


if sizeof(Ptr{Cvoid}) == 8
    primitive type PtrOrZePtr{T} 64 end
else
    primitive type PtrOrZePtr{T} 32 end
end

function Base.cconvert(::Type{PtrOrZePtr{T}}, val) where {T}
    # `cconvert` is always implemented for both `Ptr` and `ZePtr`, so pick the first result
    # that has done an actual conversion

    dev_val = Base.cconvert(ZePtr{T}, val)
    if dev_val !== val
        return dev_val
    end

    host_val = Base.cconvert(Ptr{T}, val)
    if host_val !== val
        return host_val
    end

    return val
end

function Base.unsafe_convert(::Type{PtrOrZePtr{T}}, val) where {T}
    ptr = if Core.Compiler.return_type(Base.unsafe_convert,
                                       Tuple{Type{Ptr{T}}, typeof(val)}) !== Union{}
        Base.unsafe_convert(Ptr{T}, val)
    elseif Core.Compiler.return_type(Base.unsafe_convert,
                                     Tuple{Type{ZePtr{T}}, typeof(val)}) !== Union{}
        Base.unsafe_convert(ZePtr{T}, val)
    else
        throw(ArgumentError("cannot convert to either a host or device pointer"))
    end

    return Base.bitcast(PtrOrZePtr{T}, ptr)
end


#
# Device reference objects
#

if sizeof(Ptr{Cvoid}) == 8
    primitive type ZeRef{T} 64 end
else
    primitive type ZeRef{T} 32 end
end

# general methods for ZeRef{T} type
Base.eltype(x::Type{<:ZeRef{T}}) where {T} = @isdefined(T) ? T : Any

Base.convert(::Type{ZeRef{T}}, x::ZeRef{T}) where {T} = x

# conversion or the actual ccall
Base.unsafe_convert(::Type{ZeRef{T}}, x::ZeRef{T}) where {T} = Base.bitcast(ZeRef{T}, Base.unsafe_convert(ZePtr{T}, x))
Base.unsafe_convert(::Type{ZeRef{T}}, x) where {T} = Base.bitcast(ZeRef{T}, Base.unsafe_convert(ZePtr{T}, x))

# ZeRef from literal pointer
Base.convert(::Type{ZeRef{T}}, x::ZePtr{T}) where {T} = x

# indirect constructors using ZeRef
Base.convert(::Type{ZeRef{T}}, x) where {T} = ZeRef{T}(x)


## ZeRef object backed by an array at index i

struct ZeRefArray{T,A<:AbstractArray{T}} <: Ref{T}
    x::A
    i::Int
    ZeRefArray{T,A}(x,i) where {T,A<:AbstractArray{T}} = new(x,i)
end
ZeRefArray{T}(x::AbstractArray{T}, i::Int=1) where {T} = ZeRefArray{T,typeof(x)}(x, i)
ZeRefArray(x::AbstractArray{T}, i::Int=1) where {T} = ZeRefArray{T}(x, i)
Base.convert(::Type{ZeRef{T}}, x::AbstractArray{T}) where {T} = ZeRefArray(x, 1)

function Base.unsafe_convert(P::Type{ZePtr{T}}, b::ZeRefArray{T}) where T
    return pointer(b.x, b.i)
end
function Base.unsafe_convert(P::Type{ZePtr{Any}}, b::ZeRefArray{Any})
    return convert(P, pointer(b.x, b.i))
end
Base.unsafe_convert(::Type{ZePtr{Cvoid}}, b::ZeRefArray{T}) where {T} =
    convert(ZePtr{Cvoid}, Base.unsafe_convert(ZePtr{T}, b))


## Union with all ZeRef 'subtypes'

const ZeRefs{T} = Union{ZePtr{T}, ZeRefArray{T}}


## RefOrZeRef

if sizeof(Ptr{Cvoid}) == 8
    primitive type RefOrZeRef{T} 64 end
else
    primitive type RefOrZeRef{T} 32 end
end

Base.convert(::Type{RefOrZeRef{T}}, x::Union{RefOrZeRef{T}, Ref{T}, ZeRef{T}, ZeRefs{T}}) where {T} = x

# prefer conversion to CPU ref: this is generally cheaper
Base.convert(::Type{RefOrZeRef{T}}, x) where {T} = Ref{T}(x)
Base.unsafe_convert(::Type{RefOrZeRef{T}}, x::Ref{T}) where {T} =
    Base.bitcast(RefOrZeRef{T}, Base.unsafe_convert(Ptr{T}, x))
Base.unsafe_convert(::Type{RefOrZeRef{T}}, x) where {T} =
    Base.bitcast(RefOrZeRef{T}, Base.unsafe_convert(Ptr{T}, x))

# support conversion from GPU ref
Base.unsafe_convert(::Type{RefOrZeRef{T}}, x::ZeRefs{T}) where {T} =
    Base.bitcast(RefOrZeRef{T}, Base.unsafe_convert(ZePtr{T}, x))

# support conversion from arrays
Base.convert(::Type{RefOrZeRef{T}}, x::Array{T}) where {T} = convert(Ref{T}, x)
Base.convert(::Type{RefOrZeRef{T}}, x::AbstractArray{T}) where {T} = convert(ZeRef{T}, x)
Base.unsafe_convert(P::Type{RefOrZeRef{T}}, b::ZeRefArray{T}) where T =
    Base.bitcast(RefOrZeRef{T}, Base.unsafe_convert(ZeRef{T}, b))
