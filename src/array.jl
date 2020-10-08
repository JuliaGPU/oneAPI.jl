# host array

export oneArray

@enum ArrayState begin
  ARRAY_UNMANAGED
  ARRAY_MANAGED
  ARRAY_FREED
end

mutable struct oneArray{T,N} <: AbstractGPUArray{T,N}
  buf::oneL0.DeviceBuffer
  dims::Dims{N}

  state::ArrayState

  ctx::ZeContext
  dev::ZeDevice

  function oneArray{T,N}(::UndefInitializer, dims::Dims{N})  where {T,N}
    Base.isbitsunion(T) && error("oneArray does not yet support union bits types")
    Base.isbitstype(T)  || error("oneArray only supports bits types") # allocatedinline on 1.3+
    ctx = context()
    dev = device()
    buf = device_alloc(ctx, dev, prod(dims) * sizeof(T), Base.datatype_alignment(T))
    obj = new{T,N}(buf, dims, ARRAY_MANAGED, ctx, dev)
    finalizer(unsafe_free!, obj)
    return obj
  end
end

function unsafe_free!(xs::oneArray)
  # this call should only have an effect once, becuase both the user and the GC can call it
  if xs.state == ARRAY_FREED
    return
  elseif xs.state == ARRAY_UNMANAGED
    throw(ArgumentError("Cannot free an unmanaged buffer."))
  end

  free(xs.buf)
  xs.state = ARRAY_FREED

  return
end

Base.dataids(A::oneArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::oneArray) = copy(A)


## convenience constructors

oneVector{T} = oneArray{T,1}
oneMatrix{T} = oneArray{T,2}
oneVecOrMat{T} = Union{oneVector{T},oneMatrix{T}}

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
  b = similar(a)
  @inbounds copyto!(b, a)
end


## array interface

Base.elsize(::Type{<:oneArray{T}}) where {T} = sizeof(T)

Base.size(x::oneArray) = x.dims
Base.sizeof(x::oneArray) = Base.elsize(x) * length(x)


## derived types

export oneDenseArray, oneDenseVector, oneDenseMatrix, oneDenseVecOrMat,
       oneDenseArray, oneDenseVector, oneDenseMatrix, oneDenseVecOrMat,
       oneWrappedArray, oneWrappedVector, oneWrappedMatrix, oneWrappedVecOrMat

oneContiguousSubArray{T,N,A<:oneArray} = Base.FastContiguousSubArray{T,N,A}

# dense arrays: stored contiguously in memory
oneDenseReinterpretArray{T,N,A<:Union{oneArray,oneContiguousSubArray}} = Base.ReinterpretArray{T,N,S,A} where S
oneDenseReshapedArray{T,N,A<:Union{oneArray,oneContiguousSubArray,oneDenseReinterpretArray}} = Base.ReshapedArray{T,N,A}
DenseSuboneArray{T,N,A<:Union{oneArray,oneDenseReshapedArray,oneDenseReinterpretArray}} = Base.FastContiguousSubArray{T,N,A}
oneDenseArray{T,N} = Union{oneArray{T,N}, DenseSuboneArray{T,N}, oneDenseReshapedArray{T,N}, oneDenseReinterpretArray{T,N}}
oneDenseVector{T} = oneDenseArray{T,1}
oneDenseMatrix{T} = oneDenseArray{T,2}
oneDenseVecOrMat{T} = Union{oneDenseVector{T}, oneDenseMatrix{T}}

# strided arrays
oneStridedSubArray{T,N,A<:Union{oneArray,oneDenseReshapedArray,oneDenseReinterpretArray},
                  I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                        Base.AbstractCartesianIndex}}}} = SubArray{T,N,A,I}
oneStridedArray{T,N} = Union{oneArray{T,N}, oneStridedSubArray{T,N}, oneDenseReshapedArray{T,N}, oneDenseReinterpretArray{T,N}}
oneStridedVector{T} = oneStridedArray{T,1}
oneStridedMatrix{T} = oneStridedArray{T,2}
oneStridedVecOrMat{T} = Union{oneStridedVector{T}, oneStridedMatrix{T}}

Base.pointer(x::oneStridedArray{T}) where {T} = Base.unsafe_convert(ZePtr{T}, x)
@inline function Base.pointer(x::oneStridedArray{T}, i::Integer) where T
    Base.unsafe_convert(ZePtr{T}, x) + Base._memory_offset(x, i)
end

# wrapped arrays: can be used in kernels
oneWrappedArray{T,N} = Union{oneArray{T,N}, WrappedArray{T,N,oneArray,oneArray{T,N}}}
oneWrappedVector{T} = oneWrappedArray{T,1}
oneWrappedMatrix{T} = oneWrappedArray{T,2}
oneWrappedVecOrMat{T} = Union{oneWrappedVector{T}, oneWrappedMatrix{T}}


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
Base.unsafe_convert(::Type{ZePtr{T}}, x::oneArray{T}) where {T} = convert(ZePtr{T}, x.buf)


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


## views

@inline function Base.view(A::oneArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    @boundscheck begin
        # Base's boundscheck accesses the indices, so make sure they reside on the CPU.
        # this is expensive, but it's a bounds check after all.
        J_cpu = map(j->adapt(Array, j), J)
        checkbounds(A, J_cpu...)
    end
    J_gpu = map(j->adapt(oneArray, j), J)
    Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(J_gpu...)), J_gpu...)
end

function Base.unsafe_convert(::Type{ZePtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P<:oneArray}
    return Base.unsafe_convert(ZePtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end


## reshape

Base.unsafe_convert(::Type{ZePtr{T}}, a::Base.ReshapedArray{T}) where {T} =
  Base.unsafe_convert(ZePtr{T}, parent(a))


## reinterpret

Base.unsafe_convert(::Type{ZePtr{T}}, a::Base.ReinterpretArray{T,N,S} where N) where {T,S} =
  ZePtr{T}(Base.unsafe_convert(ZePtr{S}, parent(a)))
