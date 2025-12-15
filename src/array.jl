export oneArray, oneVector, oneMatrix, oneVecOrMat,
       is_device, is_shared, is_host


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

"""
    oneArray{T,N,B} <: AbstractGPUArray{T,N}

N-dimensional dense array type for Intel GPU programming using oneAPI and Level Zero.

# Type Parameters
- `T`: Element type (must be stored inline, no isbits-unions)
- `N`: Number of dimensions
- `B`: Buffer type, one of:
  - `oneL0.DeviceBuffer`: GPU device memory (default, not CPU-accessible)
  - `oneL0.SharedBuffer`: Unified shared memory (CPU and GPU accessible)
  - `oneL0.HostBuffer`: Pinned host memory (CPU-accessible, GPU-visible)

# Memory Types

- **Device memory** (default): Fastest GPU access, not directly accessible from CPU
- **Shared memory**: Accessible from both CPU and GPU, with unified virtual addressing
- **Host memory**: CPU memory that's visible to the GPU, useful for staging

Use [`is_device`](@ref), [`is_shared`](@ref), [`is_host`](@ref) to query memory type.

# Examples
```julia
# Create arrays with different memory types
A = oneArray{Float32,2}(undef, 10, 10)                    # Device memory (default)
B = oneArray{Float32,2,oneL0.SharedBuffer}(undef, 10, 10) # Shared memory
C = oneArray{Float32,2,oneL0.HostBuffer}(undef, 10, 10)   # Host memory

# From existing array
D = oneArray(rand(Float32, 10, 10))  # Creates device memory array

# Using do-block for automatic cleanup
result = oneArray{Float32}(100) do arr
    # Use arr...
    Array(arr)  # Copy result back before cleanup
end
```

See also: [`oneVector`](@ref), [`oneMatrix`](@ref), [`is_device`](@ref), [`is_shared`](@ref)
"""
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
    alignment = Base.datatype_alignment(T)
    data = GPUArrays.cached_alloc((oneArray, B, ctx, dev, bufsize, alignment)) do
        buf = allocate(B, ctx, dev, bufsize, alignment)
        data = DataRef(buf) do buf
          release(buf)
        end
    end
    obj = new{T,N,B}(data, maxsize, 0, dims)
    finalizer(unsafe_free!, obj)
  end

  function oneArray{T,N}(data::DataRef{B}, dims::Dims{N};
                         maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,B}
    check_eltype(T)
    if sizeof(T) == 0
      offset == 0 || error("Singleton arrays cannot have a nonzero offset")
      maxsize == 0 || error("Singleton arrays cannot have a size")
    end
    obj = new{T,N,B}(copy(data), maxsize, offset, dims)
    finalizer(unsafe_free!, obj)
  end
end

GPUArrays.storage(a::oneArray) = a.data


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

buftype(x::oneArray) = buftype(typeof(x))
buftype(::Type{<:oneArray{<:Any,<:Any,B}}) where {B} = @isdefined(B) ? B : Any

"""
    is_device(a::oneArray) -> Bool

Check if the array is stored in device memory (not directly CPU-accessible).

Device memory provides the fastest GPU access but cannot be directly accessed from the CPU.

See also: [`is_shared`](@ref), [`is_host`](@ref)
"""
is_device(a::oneArray) = isa(a.data[], oneL0.DeviceBuffer)

"""
    is_shared(a::oneArray) -> Bool

Check if the array is stored in shared (unified) memory.

Shared memory is accessible from both CPU and GPU with unified virtual addressing.

See also: [`is_device`](@ref), [`is_host`](@ref)
"""
is_shared(a::oneArray) = isa(a.data[], oneL0.SharedBuffer)

"""
    is_host(a::oneArray) -> Bool

Check if the array is stored in pinned host memory.

Host memory resides on the CPU but is visible to the GPU, useful for staging data.

See also: [`is_device`](@ref), [`is_shared`](@ref)
"""
is_host(a::oneArray) = isa(a.data[], oneL0.HostBuffer)

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

@inline function Base.pointer(x::oneStridedArray{T}, i::Integer=1; type=oneL0.DeviceBuffer) where T
    PT = if type == oneL0.DeviceBuffer
      ZePtr{T}
    elseif type == oneL0.HostBuffer
      Ptr{T}
    else
      error("unknown memory type")
    end
    Base.unsafe_convert(PT, x) + Base._memory_offset(x, i)
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


## interop with libraries

function Base.unsafe_convert(::Type{Ptr{T}}, x::oneArray{T}) where {T}
  buf = x.data[]
  if is_device(x)
    throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
  end
  convert(Ptr{T}, x.data[]) + x.offset*Base.elsize(x)
end

function Base.unsafe_convert(::Type{ZePtr{T}}, x::oneArray{T}) where {T}
  convert(ZePtr{T}, x.data[]) + x.offset*Base.elsize(x)
end


## indexing

# Host-accessible arrays can be indexed from CPU, bypassing GPUArrays restrictions
function Base.getindex(x::oneArray{<:Any, <:Any, <:Union{oneL0.HostBuffer, oneL0.SharedBuffer}}, I::Int)
    @boundscheck checkbounds(x, I)
    return unsafe_load(pointer(x, I; type = oneL0.HostBuffer))
end

function Base.setindex!(x::oneArray{<:Any, <:Any, <:Union{oneL0.HostBuffer, oneL0.SharedBuffer}}, v, I::Int)
    @boundscheck checkbounds(x, I)
    return unsafe_store!(pointer(x, I; type = oneL0.HostBuffer), v)
end


## interop with GPU arrays

function Base.unsafe_convert(::Type{oneDeviceArray{T,N,AS.CrossWorkgroup}}, a::oneArray{T,N}) where {T,N}
  oneDeviceArray{T,N,AS.CrossWorkgroup}(size(a), reinterpret(LLVMPtr{T,AS.CrossWorkgroup}, pointer(a)),
                                a.maxsize - a.offset*Base.elsize(a))
end


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
  unsafe_copyto!(context(dest), device(), dest, doffs, src, soffs, n)
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
  unsafe_copyto!(context(src), device(), dest, doffs, src, soffs, n)
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
  @assert context(dest) == context(src)
  unsafe_copyto!(context(dest), device(), dest, doffs, src, soffs, n)
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
  synchronize(global_queue(context(src), device()))

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

# between Array and host-accessible oneArray

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice,
                             dest::oneDenseArray{T,<:Any,<:Union{oneL0.SharedBuffer,oneL0.HostBuffer}}, doffs, src::Array{T}, soffs, n) where T
  # maintain queue-ordered semantics
  synchronize(global_queue(ctx, dev))

  if Base.isbitsunion(T)
    # copy selector bytes
    error("oneArray does not yet support isbits-union arrays")
  end
  GC.@preserve src dest begin
    ptr = pointer(dest, doffs)
    unsafe_copyto!(pointer(dest, doffs; type=oneL0.HostBuffer), pointer(src, soffs), n)
    if Base.isbitsunion(T)
      # copy selector bytes
      error("oneArray does not yet support isbits-union arrays")
    end
  end

  return dest
end

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice,
                             dest::Array{T}, doffs, src::oneDenseArray{T,<:Any,<:Union{oneL0.SharedBuffer,oneL0.HostBuffer}}, soffs, n) where T
  # maintain queue-ordered semantics
  synchronize(global_queue(ctx, dev))

  if Base.isbitsunion(T)
    # copy selector bytes
    error("oneArray does not yet support isbits-union arrays")
  end
  GC.@preserve src dest begin
    ptr = pointer(dest, doffs)
    unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs; type=oneL0.HostBuffer), n)
    if Base.isbitsunion(T)
      # copy selector bytes
      error("oneArray does not yet support isbits-union arrays")
    end
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
  length(A) == 0 && return A
  val = convert(T, val)
  sizeof(T) == 0 && return A

  # execute! is async, so we need to allocate the pattern in USM memory
  # and keep it alive until the operation completes.
  buf = oneL0.host_alloc(context(A), sizeof(T), Base.datatype_alignment(T))
  unsafe_store!(convert(Ptr{T}, buf), val)
  unsafe_fill!(context(A), device(), pointer(A), convert(ZePtr{T}, buf), length(A))
  synchronize(global_queue(context(A), device()))
  oneL0.free(buf)
  A
end


## derived arrays

function GPUArrays.derive(::Type{T}, a::oneArray, dims::Dims{N}, offset::Int) where {T,N}
  offset = if sizeof(T) == 0
    Base.elsize(a) == 0 || error("Cannot derive a singleton array from non-singleton inputs")
    offset
  else
    (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
  end
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


## resizing

"""
  resize!(a::oneVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(a::oneVector{T}, n::Integer) where {T}
    # TODO: add additional space to allow for quicker resizing
    maxsize = n * sizeof(T)
    bufsize = if isbitstype(T)
        maxsize
    else
        # type tag array past the data
        maxsize + n
    end

    # replace the data with a new one. this 'unshares' the array.
    # as a result, we can safely support resizing unowned buffers.
    ctx = context(a)
    dev = device(a)
    buf = allocate(buftype(a), ctx, dev, bufsize, Base.datatype_alignment(T))
    ptr = convert(ZePtr{T}, buf)
    m = min(length(a), n)
    if m > 0
        unsafe_copyto!(ctx, dev, ptr, pointer(a), m)
    end
    new_data = DataRef(buf) do buf
        free(buf)
    end
    unsafe_free!(a)

    a.data = new_data
    a.dims = (n,)
    a.maxsize = maxsize
    a.offset = 0

    a
end
