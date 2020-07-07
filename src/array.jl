# host array

export oneArray

mutable struct oneArray{T,N} <: AbstractGPUArray{T,N}
  buf::oneL0.DeviceBuffer
  dims::Dims{N}

  dev::ZeDevice
end


## constructors

# type and dimensionality specified, accepting dims as tuples of Ints
function oneArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    dev = device()
    buf = device_alloc(dev, prod(dims) * sizeof(T), Base.datatype_alignment(T))

    obj = oneArray{T,N}(buf, dims, dev)
    finalizer(obj) do obj
        free(buf)
    end
    return obj
end

# type and dimensionality specified, accepting dims as series of Ints
oneArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = oneArray{T,N}(undef, dims)

# type but not dimensionality specified
oneArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = oneArray{T,N}(undef, dims)
oneArray{T}(::UndefInitializer, dims::Integer...) where {T} =
    oneArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
oneArray{T,1}() where {T} = oneArray{T,1}(undef, 0)

Base.similar(a::oneArray{T,N}) where {T,N} = oneArray{T,N}(undef, size(a))
Base.similar(a::oneArray{T}, dims::Base.Dims{N}) where {T,N} = oneArray{T,N}(undef, dims)
Base.similar(a::oneArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = oneArray{T,N}(undef, dims)


## array interface

Base.elsize(::Type{<:oneArray{T}}) where {T} = sizeof(T)

Base.size(x::oneArray) = x.dims
Base.sizeof(x::oneArray) = Base.elsize(x) * length(x)

Base.pointer(x::oneArray{T}) where {T} = convert(ZePtr{T}, x.buf)
Base.pointer(x::oneArray, i::Integer) = pointer(x) + (i-1) * Base.elsize(x)


## interop with other arrays

@inline function oneArray{T,N}(xs::AbstractArray{T,N}) where {T,N}
  A = oneArray{T,N}(undef, size(xs))
  copyto!(A, xs)
  return A
end

oneArray{T,N}(xs::AbstractArray{S,N}) where {T,N,S} = oneArray{T,N}(map(T, xs))

# underspecified constructors
oneArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = oneArray{T,N}(xs)
(::Type{oneArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = oneArray{S,N}(x)
oneArray(A::AbstractArray{T,N}) where {T,N} = oneArray{T,N}(A)

# idempotency
oneArray{T,N}(xs::oneArray{T,N}) where {T,N} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: oneArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::oneArray{T}) where {T} = throw(ArgumentError("cannot take the host address of a $(typeof(x))"))
Base.unsafe_convert(::Type{Ptr{S}}, x::oneArray{T}) where {S,T} = throw(ArgumentError("cannot take the host address of a $(typeof(x))"))

Base.unsafe_convert(::Type{ZePtr{T}}, x::oneArray{T}) where {T} = pointer(x)
Base.unsafe_convert(::Type{ZePtr{S}}, x::oneArray{T}) where {S,T} = convert(ZePtr{S}, Base.unsafe_convert(ZePtr{T}, x))


## interop with GPU arrays

function Base.convert(::Type{oneDeviceArray{T,N,AS.Global}}, a::oneArray{T,N}) where {T,N}
  oneDeviceArray{T,N,AS.Global}(a.dims, reinterpret(LLVMPtr{T,AS.Global}, pointer(a)))
end

function Adapt.adapt_storage(::KernelAdaptor, xs::oneArray{T,N}) where {T,N}
  make_resident(xs.dev, xs.buf)
  convert(oneDeviceArray{T,N,AS.Global}, xs)
end


## interop with CPU arrays

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{oneArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(oneArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:oneArray{T}}, xs::AbstractArray) where {T} =
  isbits(xs) ? xs : convert(oneArray{T}, xs)

Adapt.adapt_storage(::Type{Array}, xs::oneArray) = convert(Array, xs)

Base.collect(x::oneArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)

function Base.copyto!(dest::oneArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
                      n::Integer) where T
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest.dev, dest, doffs, src, soffs, n)
  return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer, src::oneArray{T}, soffs::Integer,
                      n::Integer) where T
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(src.dev, dest, doffs, src, soffs, n)
  return dest
end

function Base.copyto!(dest::oneArray{T}, doffs::Integer, src::oneArray{T}, soffs::Integer,
                      n::Integer) where T
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  # TODO: which device to use here?
  unsafe_copyto!(dest.dev, dest, doffs, src, soffs, n)
  return dest
end

function Base.unsafe_copyto!(dev::ZeDevice, dest::oneArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dev::ZeDevice, dest::Array{T}, doffs, src::oneArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dev::ZeDevice, dest::oneArray{T}, doffs, src::oneArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end


## utilities

zeros(T::Type, dims...) = fill!(oneArray{T}(undef, dims...), 0)
ones(T::Type, dims...) = fill!(oneArray{T}(undef, dims...), 1)
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(oneArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(oneArray{typeof(v)}(undef, dims...), v)

function Base.fill!(A::oneArray{T}, val) where T
  B = [convert(T, val)]
  unsafe_fill!(A.dev, pointer(A), pointer(B), length(A))
  A
end


## GPUArrays interfaces

GPUArrays.device(x::oneArray) = x.dev
