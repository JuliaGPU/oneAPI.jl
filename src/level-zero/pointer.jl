# pointer types

export ZePtr, ZE_NULL, PtrOrZePtr


#
# device pointer
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
# device or host pointer
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
    # FIXME: this is expensive; optimize using isapplicable?
    ptr = try
        Base.unsafe_convert(Ptr{T}, val)
    catch
        try
            Base.unsafe_convert(ZePtr{T}, val)
        catch
            throw(ArgumentError("cannot convert to either a host or device pointer"))
        end
    end
    return Base.bitcast(PtrOrZePtr{T}, ptr)
end
