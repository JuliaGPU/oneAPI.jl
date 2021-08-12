# host array

export oneArray, oneVector, oneMatrix, oneVecOrMat

@enum ArrayState begin
  ARRAY_UNMANAGED
  ARRAY_MANAGED
  ARRAY_FREED
end

mutable struct oneArray{T,N} <: AbstractGPUArray{T,N}
  ptr::ZePtr{Nothing}
  dims::Dims{N}

  state::ArrayState

  ctx::ZeContext
  dev::ZeDevice

  function oneArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    Base.isbitsunion(T) && error("oneArray does not yet support union bits types")
    Base.isbitstype(T)  || error("oneArray only supports bits types") # allocatedinline on 1.3+
    ctx = context()
    dev = device()
    ptr = allocate(ctx, dev, prod(dims) * sizeof(T), Base.datatype_alignment(T))
    obj = new{T,N}(ptr, dims, ARRAY_MANAGED, ctx, dev)
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

  release(context(xs), device(xs), xs.ptr)
  xs.state = ARRAY_FREED

  # the object is dead, so we can also wipe the pointer
  xs.ptr = ZE_NULL

  return
end

device(a::oneArray) = a.dev
context(a::oneArray) = a.ctx


## alias detection

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

@inline function oneArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
  A = oneArray{T,N}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

# underspecified constructors
oneArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = oneArray{T,N}(xs)
(::Type{oneArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = oneArray{S,N}(x)
oneArray(A::AbstractArray{T,N}) where {T,N} = oneArray{T,N}(A)

# idempotency
oneArray{T,N}(xs::oneArray{T,N}) where {T,N} = xs

# Level Zero references
oneL0.ZeRef(x::Any) = oneL0.ZeRefArray(oneArray([x]))
oneL0.ZeRef{T}(x) where {T} = oneL0.ZeRefArray{T}(oneArray(T[x]))
oneL0.ZeRef{T}() where {T} = oneL0.ZeRefArray(oneArray{T}(undef, 1))


## conversions

Base.convert(::Type{T}, x::T) where T <: oneArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::oneArray{T}) where {T} =
  throw(ArgumentError("cannot take the host address of a $(typeof(x))"))
Base.unsafe_convert(::Type{ZePtr{T}}, x::oneArray{T}) where {T} = convert(ZePtr{T}, x.ptr)


## interop with GPU arrays

function Base.unsafe_convert(::Type{oneDeviceArray{T,N,AS.Global}}, a::oneDenseArray{T,N}) where {T,N}
  oneDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)))
end

Adapt.adapt_storage(::KernelAdaptor, xs::oneArray{T,N}) where {T,N} =
  Base.unsafe_convert(oneDeviceArray{T,N,AS.Global}, xs)

# we materialize ReshapedArray/ReinterpretArray/SubArray/... directly as a device array
Adapt.adapt_structure(::KernelAdaptor, xs::oneDenseArray{T,N}) where {T,N} =
  Base.unsafe_convert(oneDeviceArray{T,N,AS.Global}, xs)


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
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(context(dest), device(dest), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::oneDenseArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Integer, src::oneDenseArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(context(src), device(src), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::Array{T}, src::oneDenseArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::oneDenseArray{T}, doffs::Integer, src::oneDenseArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  @assert device(dest) == device(src) && context(dest) == context(src)
  unsafe_copyto!(context(dest), device(dest), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::oneDenseArray{T}, src::oneDenseArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice,
                             dest::oneDenseArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice,
                             dest::Array{T}, doffs, src::oneDenseArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end

  # copies to the host are synchronizing
  synchronize(global_queue(context(src), device(src)))

  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice,
                             dest::oneDenseArray{T}, doffs, src::oneDenseArray{T}, soffs, n) where T
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

function Base.fill!(A::oneDenseArray{T}, val) where T
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

device(a::SubArray) = device(parent(a))
context(a::SubArray) = context(parent(a))

# contiguous subarrays
function Base.unsafe_convert(::Type{ZePtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(ZePtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end

# reshaped subarrays
function Base.unsafe_convert(::Type{ZePtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(ZePtr{T}, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## reshape

device(a::Base.ReshapedArray) = device(parent(a))
context(a::Base.ReshapedArray) = context(parent(a))

Base.unsafe_convert(::Type{ZePtr{T}}, a::Base.ReshapedArray{T}) where {T} =
  Base.unsafe_convert(ZePtr{T}, parent(a))


## reinterpret

device(a::Base.ReinterpretArray) = device(parent(a))
context(a::Base.ReinterpretArray) = context(parent(a))

Base.unsafe_convert(::Type{ZePtr{T}}, a::Base.ReinterpretArray{T,N,S} where N) where {T,S} =
  ZePtr{T}(Base.unsafe_convert(ZePtr{S}, parent(a)))
