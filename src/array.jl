export oneArray, oneVector, oneMatrix, oneVecOrMat


## array type

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

function contains_eltype(T, X)
    if T === X
      return true
    elseif T isa Union
        for U in Base.uniontypes(T)
            contains_eltype(U, X) && return true
        end
    elseif hasfieldcount(T)
        for U in fieldtypes(T)
            contains_eltype(U, X) && return true
        end
    end
    return false
end

function check_eltype(T)
  Base.allocatedinline(T) || error("oneArray only supports element types that are stored inline")
  Base.isbitsunion(T) && error("oneArray does not yet support isbits-union arrays")
  if oneL0.module_properties(device()).fp16flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP16 !=
      oneL0.ZE_DEVICE_MODULE_FLAG_FP16
    contains_eltype(T, Float16) && error("Float16 is not supported on this device")
  end
  if oneL0.module_properties(device()).fp64flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 !=
      oneL0.ZE_DEVICE_MODULE_FLAG_FP64
    contains_eltype(T, Float64) && error("Float64 is not supported on this device")
  end
end

mutable struct oneArray{T,N,B} <: AbstractGPUArray{T,N}
  data::DataRef{B}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int   # offset of the data in the buffer, in number of elements
  dims::Dims{N}

  function oneArray{T,N,B}(::UndefInitializer, dims::Dims{N}) where {T,N,B}
    check_eltype(T)
    maxsize = prod(dims) * sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end

    ctx = context()
    dev = device()
    buf = allocate(B, ctx, dev, bufsize, Base.datatype_alignment(T))
    data = DataRef(buf) do buf
      release(buf)
    end
    obj = new{T,N,B}(data, maxsize, 0, dims)
    finalizer(unsafe_free!, obj)
  end

  function oneArray{T,N}(data::DataRef{B}, dims::Dims{N};
                         maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,B}
    check_eltype(T)
    obj = new{T,N,B}(copy(data), maxsize, offset, dims)
    finalizer(unsafe_free!, obj)
  end
end

unsafe_free!(a::oneArray) = GPUArrays.unsafe_free!(a.data)


## alias detection

Base.dataids(A::oneArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::oneArray) = copy(A)

function Base.mightalias(A::oneArray, B::oneArray)
  rA = pointer(A):pointer(A)+sizeof(A)
  rB = pointer(B):pointer(B)+sizeof(B)
  return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

const oneVector{T} = oneArray{T,1}
const oneMatrix{T} = oneArray{T,2}
const oneVecOrMat{T} = Union{oneVector{T},oneMatrix{T}}

# default to non-unified memory
oneArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  oneArray{T,N,oneL0.DeviceBuffer}(undef, dims)

# buffer, type and dimensionality specified
oneArray{T,N,B}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,B} =
  oneArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))
oneArray{T,N,B}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,B} =
  oneArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
oneArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  oneArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
oneArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  oneArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# only type specified
oneArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  oneArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
oneArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  oneArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
oneArray{T,1,B}() where {T,B} = oneArray{T,1,B}(undef, 0)
oneArray{T,1}() where {T} = oneArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:oneArray => (),
                      :(oneArray{T}) => (:T,),
                      :(oneArray{T,N}) => (:T, :N),
                      :(oneArray{T,N,B}) => (:T, :N, :B))
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

Base.similar(a::oneArray{T,N,B}) where {T,N,B} =
  oneArray{T,N,B}(undef, size(a))
Base.similar(a::oneArray{T,<:Any,B}, dims::Base.Dims{N}) where {T,N,B} =
  oneArray{T,N,B}(undef, dims)
Base.similar(a::oneArray{<:Any,<:Any,B}, ::Type{T}, dims::Base.Dims{N}) where {T,N,B} =
  oneArray{T,N,B}(undef, dims)

function Base.copy(a::oneArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end


## array interface

Base.elsize(::Type{<:oneArray{T}}) where {T} = sizeof(T)

Base.size(x::oneArray) = x.dims
Base.sizeof(x::oneArray) = Base.elsize(x) * length(x)

function context(A::oneArray)
  return oneL0.context(A.data[])
end

function device(A::oneArray)
  return oneL0.device(A.data[])
end


## derived types

export oneDenseArray, oneDenseVector, oneDenseMatrix, oneDenseVecOrMat,
       oneStridedArray, oneStridedVector, oneStridedMatrix, oneStridedVecOrMat,
       oneWrappedArray, oneWrappedVector, oneWrappedMatrix, oneWrappedVecOrMat

# dense arrays: stored contiguously in memory
#
# all common dense wrappers are currently represented as oneArray objects.
# this simplifies common use cases, and greatly improves load time.
const oneDenseArray{T,N} = oneArray{T,N}
const oneDenseVector{T} = oneDenseArray{T,1}
const oneDenseMatrix{T} = oneDenseArray{T,2}
const oneDenseVecOrMat{T} = Union{oneDenseVector{T}, oneDenseMatrix{T}}
# XXX: these dummy aliases (oneDenseArray=oneArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
const oneStridedSubArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                             Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:oneArray,I}
const oneStridedArray{T,N} = Union{oneArray{T,N}, oneStridedSubArray{T,N}}
const oneStridedVector{T} = oneStridedArray{T,1}
const oneStridedMatrix{T} = oneStridedArray{T,2}
const oneStridedVecOrMat{T} = Union{oneStridedVector{T}, oneStridedMatrix{T}}

Base.pointer(x::oneStridedArray{T}) where {T} = Base.unsafe_convert(ZePtr{T}, x)
@inline function Base.pointer(x::oneStridedArray{T}, i::Integer) where T
    Base.unsafe_convert(ZePtr{T}, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a oneArray
const oneWrappedArray{T,N} = Union{oneArray{T,N}, WrappedArray{T,N,oneArray,oneArray{T,N}}}
const oneWrappedVector{T} = oneWrappedArray{T,1}
const oneWrappedMatrix{T} = oneWrappedArray{T,2}
const oneWrappedVecOrMat{T} = Union{oneWrappedVector{T}, oneWrappedMatrix{T}}


## interop with other arrays

@inline function oneArray{T,N,B}(xs::AbstractArray{<:Any,N}) where {T,N,B}
  A = oneArray{T,N,B}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

@inline oneArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N} =
  oneArray{T,N,oneL0.DeviceBuffer}(xs)

@inline oneArray{T,N}(xs::oneArray{<:Any,N,B}) where {T,N,B} =
  oneArray{T,N,B}(xs)

# underspecified constructors
oneArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = oneArray{T,N}(xs)
(::Type{oneArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = oneArray{S,N}(x)
oneArray(A::AbstractArray{T,N}) where {T,N} = oneArray{T,N}(A)

# idempotency
oneArray{T,N,B}(xs::oneArray{T,N,B}) where {T,N,B} = xs
oneArray{T,N}(xs::oneArray{T,N,B}) where {T,N,B} = xs

# Level Zero references
oneL0.ZeRef(x::Any) = oneL0.ZeRefArray(oneArray([x]))
oneL0.ZeRef{T}(x) where {T} = oneL0.ZeRefArray{T}(oneArray(T[x]))
oneL0.ZeRef{T}() where {T} = oneL0.ZeRefArray(oneArray{T}(undef, 1))


## conversions

Base.convert(::Type{T}, x::T) where T <: oneArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::oneArray{T}) where {T} =
  throw(ArgumentError("cannot take the host address of a $(typeof(x))"))
Base.unsafe_convert(::Type{ZePtr{T}}, x::oneArray{T}) where {T} =
  convert(ZePtr{T}, x.data[]) + x.offset*Base.elsize(x)


## interop with GPU arrays

function Base.unsafe_convert(::Type{oneDeviceArray{T,N,AS.Global}}, a::oneArray{T,N}) where {T,N}
  oneDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)),
                                a.maxsize - a.offset*Base.elsize(a))
end

Adapt.adapt_storage(::KernelAdaptor, xs::oneArray{T,N}) where {T,N} =
  Base.unsafe_convert(oneDeviceArray{T,N,AS.Global}, xs)


## memory copying

typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
typetagdata(a::oneArray, i=1) =
  convert(ZePtr{UInt8}, a.data[]) + a.maxsize + a.offset + i - 1

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
    error("oneArray does not yet support isbits-union arrays")
  end
  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice,
                             dest::Array{T}, doffs, src::oneDenseArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("oneArray does not yet support isbits-union arrays")
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
    error("oneArray does not yet support isbits-union arrays")
  end
  return dest
end


## gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{oneArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(oneArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:oneArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(oneArray{T}, xs)


## utilities

zeros(T::Type, dims...) = fill!(oneArray{T}(undef, dims...), zero(T))
ones(T::Type, dims...) = fill!(oneArray{T}(undef, dims...), one(T))
zeros(dims...) = zeros(Float64, dims...)
ones(dims...) = ones(Float64, dims...)
fill(v, dims...) = fill!(oneArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(oneArray{typeof(v)}(undef, dims...), v)

function Base.fill!(A::oneDenseArray{T}, val) where T
  B = [convert(T, val)]
  unsafe_fill!(context(A), device(A), pointer(A), pointer(B), length(A))
  A
end


## derived arrays

function GPUArrays.derive(::Type{T}, N::Int, a::oneArray, dims::Dims, offset::Int) where {T}
  offset = (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
  oneArray{T,N}(a.data, dims; a.maxsize, offset)
end


## views

device(a::SubArray) = device(parent(a))
context(a::SubArray) = context(parent(a))

# pointer conversions
function Base.unsafe_convert(::Type{ZePtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(ZePtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end
function Base.unsafe_convert(::Type{ZePtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(ZePtr{T}, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## PermutedDimsArray

device(a::Base.PermutedDimsArray) = device(parent(a))
context(a::Base.PermutedDimsArray) = context(parent(a))

Base.unsafe_convert(::Type{ZePtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(ZePtr{T}, parent(A))


## unsafe_wrap

"""
    unsafe_wrap(Array, arr::oneArray{_,_,oneL0.SharedBuffer})

Wrap a Julia `Array` around the buffer that backs a `oneArray`. This is only possible if the
GPU array is backed by a shared buffer, i.e. if it was created with `oneArray{T}(undef, ...)`.
"""
function Base.unsafe_wrap(::Type{Array}, arr::oneArray{T,N,oneL0.SharedBuffer}) where {T,N}
  # TODO: can we make this more convenient by increasing the buffer's refcount and using
  #       a finalizer on the Array? does that work when taking views etc of the Array?
  ptr = reinterpret(Ptr{T}, pointer(arr))
  unsafe_wrap(Array, ptr, size(arr))
end
