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

function _device_supports_bfloat16(dev=device())
    # check the driver extension first
    if haskey(
            oneL0.extension_properties(dev.driver),
            oneL0.ZE_BFLOAT16_CONVERSIONS_EXT_NAME
        )
        return true
    end
    # some drivers (e.g. older versions on PVC/Max) don't advertise the extension,
    # but the hardware supports BFloat16 natively. fall back to checking device ID.
    dev_id = oneL0.properties(dev).deviceId
    # Intel Data Center GPU Max (Ponte Vecchio): device IDs 0x0BD0-0x0BDB
    if 0x0BD0 <= dev_id <= 0x0BDB
        return true
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
    return @static if isdefined(Core, :BFloat16)
        if !_device_supports_bfloat16()
            contains_eltype(T, Core.BFloat16) && error("BFloat16 is not supported on this device")
        end
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
  offset::Int   # offset of the data in the buffer, in bytes
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
  convert(Ptr{T}, x.data[]) + x.offset
end

function Base.unsafe_convert(::Type{ZePtr{T}}, x::oneArray{T}) where {T}
  convert(ZePtr{T}, x.data[]) + x.offset
end


## indexing

# Host-accessible arrays can be indexed from CPU, bypassing GPUArrays restrictions.
# Wait for queued work first, e.g., the kernel computing the result of a reduction. This
# only synchronizes the current task's stream; work submitted by other tasks, or to an
# explicitly created queue, needs to be synchronized explicitly.
function Base.getindex(x::oneArray{<:Any, <:Any, <:Union{oneL0.HostBuffer, oneL0.SharedBuffer}}, I::Int)
    @boundscheck checkbounds(x, I)
    synchronize(global_stream(context(x), device()))
    return unsafe_load(pointer(x, I; type = oneL0.HostBuffer))
end

function Base.setindex!(x::oneArray{<:Any, <:Any, <:Union{oneL0.HostBuffer, oneL0.SharedBuffer}}, v, I::Int)
    @boundscheck checkbounds(x, I)
    synchronize(global_stream(context(x), device()))
    return unsafe_store!(pointer(x, I; type = oneL0.HostBuffer), v)
end


## interop with GPU arrays

function Base.unsafe_convert(::Type{oneDeviceArray{T,N,AS.CrossWorkgroup}}, a::oneArray{T,N}) where {T,N}
  oneDeviceArray{T,N,AS.CrossWorkgroup}(size(a), reinterpret(LLVMPtr{T,AS.CrossWorkgroup}, pointer(a)),
                                a.maxsize - a.offset)
end


## memory copying

typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
function typetagdata(a::oneArray, i=1)
  # for zero-size element types (e.g. singleton unions), the byte offset
  # is always zero, so the corresponding element offset is also zero
  elem_offset = iszero(Base.elsize(a)) ? 0 : a.offset ÷ Base.elsize(a)
  return convert(ZePtr{UInt8}, a.data[]) + a.maxsize + elem_offset + i - 1
end

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
  GC.@preserve src dest begin
    unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)

    # Keep pageable host memory alive until the queued copy completes.
    synchronize(global_stream(ctx, dev))
  end
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
  synchronize(global_stream(context(src), device()))

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
  synchronize(global_stream(ctx, dev))

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
  synchronize(global_stream(ctx, dev))

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

# NOTE: `Base.fill!` is deliberately not specialized here. GPUArrays' generic definition
# lowers to a single fill kernel, whereas the Level Zero memory-fill command requires the
# pattern to live in USM memory: a host allocation, a residency call and a free around
# every call, plus a full queue synchronize to keep the pattern alive until the
# asynchronous fill has read it.


## derived arrays

function GPUArrays.derive(::Type{T}, a::oneArray, dims::Dims{N}, offset::Int) where {T,N}
  if sizeof(T) == 0
    Base.elsize(a) == 0 || error("Cannot derive a singleton array from non-singleton inputs")
  end
  offset = a.offset + offset * sizeof(T)
  # The derived array constructor copies `a.data`, but merely loading that field does not
  # keep `a` alive. Without this preserve, `a` may be finalized between the field load and
  # the DataRef copy, causing its finalizer to mark the reference as freed.
  GC.@preserve a oneArray{T,N}(a.data, dims; a.maxsize, offset)
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
    unsafe_wrap(Array, arr::oneArray{_,_,<:Union{oneL0.SharedBuffer,oneL0.HostBuffer}})

Wrap a Julia `Array` around the buffer that backs a `oneArray`, without copying. This is
only possible if the GPU array is backed by memory that is accessible from the host, i.e.,
a shared buffer (as created by `oneArray{T}(undef, ...)`), a host buffer, or host memory
that was itself wrapped using `unsafe_wrap(oneArray, ...)`.

!!! warning

    The returned `Array` does **not** keep `arr` alive. The caller has to keep a reference
    to `arr` for as long as the `Array`, or anything derived from it, is used; otherwise
    the `Array` may end up referring to freed memory. Device operations execute
    asynchronously, so call `synchronize()` before accessing the returned array after
    using `arr` on the device.
"""
function Base.unsafe_wrap(::Type{Array},
                          arr::oneArray{T,N,<:Union{oneL0.SharedBuffer,oneL0.HostBuffer}}) where {T,N}
  # TODO: can we make this more convenient by increasing the buffer's refcount and using
  #       a finalizer on the Array? does that work when taking views etc of the Array?
  ptr = reinterpret(Ptr{T}, pointer(arr))
  unsafe_wrap(Array, ptr, size(arr))
end

"""
    unsafe_wrap(oneArray, a::Array)
    unsafe_wrap(oneArray, ptr::Ptr{T}, dims)
    unsafe_wrap(oneArray{T,N,oneL0.HostBuffer}, ...)

Wrap a `oneArray` around host memory, without copying, so that it can be used on the
device, e.g., in kernels or broadcasts. Changes made through the `oneArray` are visible in
the original array, and vice versa. The resulting array is backed by a host buffer.

This requires a driver that supports mapping system memory (the
`ZE_extension_external_memmap_sysmem` extension); otherwise an `ArgumentError` is thrown,
and `oneArray(a)` can be used to copy the data instead.

When wrapping an `Array`, the returned `oneArray` keeps it alive. When wrapping a pointer,
the caller has to make sure the memory stays valid for as long as the `oneArray` is used.
In both cases, the memory must not be freed or reallocated while it is wrapped (e.g., by
calling `resize!` on the original array), and resizing the wrapper is not supported.

Device operations execute asynchronously, so call `synchronize()` before accessing the
original memory on the host.

!!! warning

    Level Zero maps host memory in whole pages, and does not support overlapping mappings.
    Memory within pages that are already mapped for another wrapper can be wrapped as long
    as it lies entirely within that mapping. Memory whose pages only partially overlap an
    existing mapping cannot be wrapped, and results in an `ArgumentError`, until the other
    wrappers are freed. As a result, whether wrapping succeeds can depend on the order in
    which neighboring arrays are wrapped: wrapping a large array first allows wrapping
    smaller arrays inside of it, but not the other way around.

```julia
a = rand(Float32, 1024)
b = unsafe_wrap(oneArray, a)
b .= sin.(b)        # executes on the device, updating `a`
synchronize()
```
"""
unsafe_wrap(::Type{<:oneArray}, ::Any, ::Any...)

# Level Zero can only map whole pages of system memory, and does not support overlapping
# mappings (e.g., of two small arrays that share a page). Each mapping covers a range of
# pages, and is shared by all wrappers of memory within that range. Memory that only
# partially overlaps an existing mapping cannot be wrapped: that would require replacing
# the mapping, which the driver cannot do while it is in use. Mappings are never replaced
# or freed while wrappers use them.
mutable struct SystemMapping
    const ctx::ZeContext
    const lo::UInt
    const hi::UInt
    const buf::oneL0.HostBuffer
    # devices the mapping has been made resident on
    const resident::Vector{ZeDevice}
    # owners of memory within the mapping, which have to outlive it. they are kept until
    # the mapping is released, not just until their own wrapper is freed, because other
    # wrappers may keep the mapping alive.
    const owners::Base.IdSet{Any}
    refcount::Int
    # set when freeing the mapping failed; its pages then remain occupied
    broken::Bool
end

const system_mappings = SystemMapping[]
const system_mappings_lock = ReentrantLock()

page_range(ptr::Ptr, bytesize::Integer) = let pagesize = UInt(ccall(:getpagesize, Cint, ()))
    lo = UInt(ptr) & ~(pagesize - 1)
    hi = Base.checked_add(UInt(ptr), UInt(bytesize), pagesize - 1) & ~(pagesize - 1)
    lo, hi
end

function map_system_memory(ctx::ZeContext, dev::ZeDevice, ptr::Ptr, bytesize::Integer, owner)
    lo, hi = page_range(ptr, bytesize)
    @lock system_mappings_lock begin
        for m in system_mappings
            m.ctx == ctx && m.lo < hi && lo < m.hi || continue
            if m.broken || !(m.lo <= lo && hi <= m.hi)
                throw(ArgumentError("""Cannot wrap host memory at $ptr: its pages ($(repr(lo))-$(repr(hi))) partially overlap memory that is already wrapped (pages $(repr(m.lo))-$(repr(m.hi))).
                                       Level Zero maps host memory in whole pages and does not support overlapping mappings.
                                       Free the other wrapper first, wrap memory in a different order, or use `oneArray(a)` to copy the data instead."""))
            end
            if !(dev in m.resident)
                make_resident(ctx, dev, m.buf)
                push!(m.resident, dev)
            end
            m.refcount += 1
            owner === nothing || push!(m.owners, owner)
            return m
        end

        buf = oneL0.host_memmap(ctx, Ptr{Cvoid}(lo), hi - lo)
        try
            # like any host allocation, the mapping has to be resident to be usable by kernels
            make_resident(ctx, dev, buf)
        catch
            release(buf)
            rethrow()
        end
        owners = Base.IdSet{Any}()
        owner === nothing || push!(owners, owner)
        m = SystemMapping(ctx, lo, hi, buf, [dev], owners, 1, false)
        push!(system_mappings, m)
        return m
    end
end

# a wrapper's claim on a mapping. records are linked into a queue when they are freed from a
# finalizer, which keeps the mapping (and the owners it roots) alive until it is released.
mutable struct SystemWrapper
    const mapping::SystemMapping
    next::Union{Nothing,SystemWrapper}
end

function unmap_system_memory(m::SystemMapping)
    @lock system_mappings_lock begin
        m.refcount -= 1
        m.refcount == 0 || return
        try
            # this waits for outstanding work using the mapping
            release(m.buf)
        catch err
            # keep the mapping registered, which keeps its pages occupied and its owners
            # alive, rather than risking mapping them again or freeing memory that the
            # device may still access
            m.broken = true
            @error "Failed to release a mapping of host memory; its memory will be leaked" exception=(err, catch_backtrace())
            return
        end
        filter!(x -> x !== m, system_mappings)
    end
    return
end

# finalizers must not block or call into the driver, so wrappers that are freed from a
# finalizer are queued (without allocating or yielding while holding the lock), and
# released by a task that is woken up through an async condition.
const pending_wrappers = Ref{Union{Nothing,SystemWrapper}}(nothing)
const pending_lock = Threads.SpinLock()
const release_condition = Ref{Base.AsyncCondition}()
const release_task_lock = ReentrantLock()

function start_release_task()
    isassigned(release_condition) && return
    @lock release_task_lock begin
        isassigned(release_condition) && return
        cond = Base.AsyncCondition()
        errormonitor(@async while true
            wait(cond)
            process_pending_wrappers()
        end)
        release_condition[] = cond
    end
    return
end

function process_pending_wrappers()
    lock(pending_lock)
    w = pending_wrappers[]
    pending_wrappers[] = nothing
    unlock(pending_lock)
    while w !== nothing
        next = w.next
        w.next = nothing
        unmap_system_memory(w.mapping)
        w = next
    end
end

function free_system_wrapper(w::SystemWrapper)
    if ccall(:jl_gc_is_in_finalizer, Int8, ()) == 0
        unmap_system_memory(w.mapping)
    else
        lock(pending_lock)
        w.next = pending_wrappers[]
        pending_wrappers[] = w
        unlock(pending_lock)
        ccall(:uv_async_send, Cint, (Ptr{Cvoid},), release_condition[].handle)
    end
    return
end

system_memmap_supported(drv::ZeDriver) =
    haskey(oneL0.extension_properties(drv), "ZE_extension_external_memmap_sysmem")

# `owner` is kept alive for as long as the wrapper
function wrap_system_memory(::Type{oneArray{T,N,B}}, ptr::Ptr{T}, dims::NTuple{N,Int},
                            owner=nothing) where {T,N,B}
    B == oneL0.HostBuffer ||
        throw(ArgumentError("Cannot wrap host memory as $B; use oneL0.HostBuffer"))
    check_eltype(T)
    isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
    all(>=(0), dims) || throw(ArgumentError("Invalid dimensions $dims"))
    bytesize = Base.checked_mul(foldl(Base.checked_mul, dims; init=1), sizeof(T))
    if bytesize > 0 && ptr == C_NULL
        throw(ArgumentError("Cannot wrap a NULL pointer"))
    end
    if !iszero(UInt(ptr) % Base.datatype_alignment(T))
        throw(ArgumentError("Pointer $ptr is not sufficiently aligned for elements of type $T"))
    end
    if Base.Checked.add_with_overflow(UInt(ptr), UInt(bytesize) + UInt(ccall(:getpagesize, Cint, ())))[2]
        throw(ArgumentError("Memory range of $bytesize bytes at $ptr exceeds the address space"))
    end
    if !system_memmap_supported(driver())
        throw(ArgumentError("""The Level Zero driver does not support mapping host memory, which is required to wrap it as a oneArray.
                               Use `oneArray(a)` to copy the data instead."""))
    end

    ctx = context()
    buf = oneL0.HostBuffer(Ptr{Cvoid}(ptr), bytesize, ctx)
    bytesize == 0 && return oneArray{T,N}(DataRef(Returns(nothing), buf), dims)

    start_release_task()
    m = map_system_memory(ctx, device(), ptr, bytesize, owner)
    arr, data = try
        w = SystemWrapper(m, nothing)
        data = DataRef(buf) do _
            free_system_wrapper(w)
        end
        oneArray{T,N}(data, dims), data
    catch
        # give up our claim on the mapping (which roots the owner until it is released)
        unmap_system_memory(m)
        rethrow()
    end
    # the array holds its own reference
    unsafe_free!(data)
    return arr
end
wrap_system_memory(::Type{oneArray{T,N}}, ptr::Ptr{T}, dims::NTuple{N,Int}, owner=nothing) where {T,N} =
    wrap_system_memory(oneArray{T,N,oneL0.HostBuffer}, ptr, dims, owner)

Base.unsafe_wrap(::Union{Type{oneArray},Type{oneArray{T}},Type{oneArray{T,N}}},
                 ptr::Ptr{T}, dims::NTuple{N,Int}) where {T,N} =
    wrap_system_memory(oneArray{T,N}, ptr, dims)
Base.unsafe_wrap(::Type{oneArray{T,N,B}}, ptr::Ptr{T}, dims::NTuple{N,Int}) where {T,N,B} =
    wrap_system_memory(oneArray{T,N,B}, ptr, dims)

# integer size input
Base.unsafe_wrap(::Union{Type{oneArray},Type{oneArray{T}},Type{oneArray{T,1}}},
                 ptr::Ptr{T}, dim::Integer) where {T} =
    unsafe_wrap(oneArray{T,1}, ptr, (Int(dim),))
Base.unsafe_wrap(::Type{oneArray{T,1,B}}, ptr::Ptr{T}, dim::Integer) where {T,B} =
    unsafe_wrap(oneArray{T,1,B}, ptr, (Int(dim),))

# array input: keep the array alive for as long as the wrapper
Base.unsafe_wrap(::Union{Type{oneArray},Type{oneArray{T}},Type{oneArray{T,N}}},
                 a::Array{T,N}) where {T,N} =
    wrap_system_memory(oneArray{T,N}, pointer(a), size(a), a)
Base.unsafe_wrap(::Type{oneArray{T,N,B}}, a::Array{T,N}) where {T,N,B} =
    wrap_system_memory(oneArray{T,N,B}, pointer(a), size(a), a)

# whether an array wraps host memory using `unsafe_wrap`
function is_system(a::oneArray)
    buf = a.data[]
    buf isa oneL0.HostBuffer && sizeof(buf) > 0 || return false
    lo, hi = page_range(pointer(buf), sizeof(buf))
    @lock system_mappings_lock begin
        any(m -> m.ctx == context(a) && m.lo <= lo && hi <= m.hi, system_mappings)
    end
end


## resizing

"""
  resize!(a::oneVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(a::oneVector{T}, n::Integer) where {T}
    # resizing would detach the array from the host memory it wraps
    is_system(a) && throw(ArgumentError("Cannot resize a oneArray that wraps host memory"))

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
