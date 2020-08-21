# host array

export oneArray

mutable struct oneArray{T,N} <: AbstractGPUArray{T,N}
  buf::oneL0.DeviceBuffer
  dims::Dims{N}

  ctx::ZeContext
  dev::ZeDevice

  # ownership
  parent::Union{Nothing,oneArray}
  refcount::Int
  freed::Bool

  # primary array
  function oneArray{T,N}(buf, dims, ctx, dev) where {T,N}
    obj = new(buf, dims, ctx, dev, nothing, 0, false)
    retain(obj)
    finalizer(unsafe_free!, obj)
    return obj
  end

  # derived array
  function oneArray{T,N}(buf, dims::Dims{N}, parent::oneArray) where {T,N}
    self = new(buf, dims, parent.ctx, parent.dev, parent, 0, false)
    retain(self)
    retain(parent)
    finalizer(unsafe_free!, self)
    return self
  end
end

function unsafe_free!(xs::oneArray)
  # this call should only have an effect once, becuase both the user and the GC can call it
  xs.freed && return
  _unsafe_free!(xs)
  xs.freed = true
  return
end

function _unsafe_free!(xs::oneArray)
  @assert xs.refcount >= 0
  if release(xs)
    if xs.parent === nothing
      # primary array with all references gone
      free(xs.buf)
    else
      # derived object
      _unsafe_free!(xs.parent)
    end
  end

  return
end

@inline function retain(a::oneArray)
  a.refcount += 1
  return
end

@inline function release(a::oneArray)
  a.refcount -= 1
  return a.refcount == 0
end

Base.parent(A::oneArray) = something(A.parent, A)

function Base.dataids(A::oneArray)
  if A.parent === nothing
    (UInt(pointer(A)),)
  else
    (Base.dataids(parent(A))..., UInt(pointer(A)),)
  end
end

function Base.unaliascopy(A::oneArray)
  if A.parent === nothing
    copy(A)
  else
    offset = pointer(A) - pointer(A.parent)
    new_parent = Base.unaliascopy(A.parent)
    typeof(A)(pointer(new_parent) + offset, A.dims, new_parent, A.pooled, A.ctx)
  end
end

# optimized alias detection for views
function Base.mightalias(A::oneArray, B::oneArray)
    if parent(A) !== parent(B)
        # We cannot do any better than the usual dataids check
        return invoke(Base.mightalias, Tuple{AbstractArray, AbstractArray}, A, B)
    end

    rA = pointer(A):pointer(A)+sizeof(A)
    rB = pointer(B):pointer(B)+sizeof(B)
    return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

oneVector{T} = oneArray{T,1}
oneMatrix{T} = oneArray{T,2}
oneVecOrMat{T} = Union{oneVector{T},oneMatrix{T}}

# type and dimensionality specified, accepting dims as tuples of Ints
function oneArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    ctx = context()
    dev = device()
    buf = device_alloc(ctx, dev, prod(dims) * sizeof(T), Base.datatype_alignment(T))
    oneArray{T,N}(buf, dims, ctx, dev)
end

# type and dimensionality specified, accepting dims as series of Ints
oneArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = oneArray{T,N}(undef, dims)

# type but not dimensionality specified
oneArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = oneArray{T,N}(undef, dims)
oneArray{T}(::UndefInitializer, dims::Integer...) where {T} =
    oneArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
oneArray{T,1}() where {T} = oneArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:oneArray => (), :(oneArray{T}) => (:T,), :(oneArray{T,N}) => (:T, :N))
  @eval begin
    function $ctor(f::Function, args...) where {$(tvars...)}
      xs = $ctor(args...)
      try
        f(xs)
      finally
        unsafe_free!(xs)
      end
    end
  end
end

Base.similar(a::oneArray{T,N}) where {T,N} = oneArray{T,N}(undef, size(a))
Base.similar(a::oneArray{T}, dims::Base.Dims{N}) where {T,N} = oneArray{T,N}(undef, dims)
Base.similar(a::oneArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = oneArray{T,N}(undef, dims)

function Base.copy(a::oneArray{T,N}) where {T,N}
  buf = device_alloc(a.dev, sizeof(a), Base.datatype_alignment(T))
  b = oneArray{T,N}(buf, a.dims, a.dev)
  copyto!(b, a)
end


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

function Base._reshape(parent::oneArray, dims::Dims)
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return oneArray{eltype(parent),length(dims)}(parent.buf, dims, parent)
end
function Base._reshape(parent::oneArray{T,1}, dims::Tuple{Int}) where T
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return parent
end


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
  make_resident(xs.ctx, xs.dev, xs.buf)
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
  unsafe_copyto!(dest.ctx, dest.dev, dest, doffs, src, soffs, n)
  return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer, src::oneArray{T}, soffs::Integer,
                      n::Integer) where T
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(src.ctx, src.dev, dest, doffs, src, soffs, n)
  return dest
end

function Base.copyto!(dest::oneArray{T}, doffs::Integer, src::oneArray{T}, soffs::Integer,
                      n::Integer) where T
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  # TODO: which device to use here?
  unsafe_copyto!(dest.ctx, dest.dev, dest, doffs, src, soffs, n)
  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dest::oneArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dest::Array{T}, doffs, src::oneArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dest::oneArray{T}, doffs, src::oneArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end


## utilities

zeros(T::Type, dims...) = fill!(oneArray{T}(undef, dims...), 0)
ones(T::Type, dims...) = fill!(oneArray{T}(undef, dims...), 1)
zeros(dims...) = zeros(Float64, dims...)
ones(dims...) = ones(Float64, dims...)
fill(v, dims...) = fill!(oneArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(oneArray{typeof(v)}(undef, dims...), v)

function Base.fill!(A::oneArray{T}, val) where T
  B = [convert(T, val)]
  unsafe_fill!(A.ctx, A.dev, pointer(A), pointer(B), length(A))
  A
end
