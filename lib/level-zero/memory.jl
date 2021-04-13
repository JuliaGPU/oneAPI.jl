# Raw memory management

export device_alloc, host_alloc, shared_alloc, free, properties, lookup_alloc


#
# untyped buffers
#

abstract type AbstractBuffer end

Base.convert(T::Type{<:Union{Ptr,ZePtr}}, buf::AbstractBuffer) =
    throw(ArgumentError("Illegal conversion of a $(typeof(buf)) to a $T"))

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(P::Type{<:Union{Ptr,ZePtr}}, buf::AbstractBuffer) = convert(P, buf)

function free(buf::AbstractBuffer)
    zeMemFree(context(buf), buf)
end


## device buffer

"""
    DeviceBuffer

A buffer of device memory, owned by a specific device. Generally, may only be accessed by
the device that owns it.
"""
struct DeviceBuffer <: AbstractBuffer
    ptr::ZePtr{Cvoid}
    bytesize::Int
    context::ZeContext
    device::ZeDevice
end

function device_alloc(ctx::ZeContext, dev::ZeDevice, bytesize::Integer, alignment::Integer=1;
                      flags=0, ordinal::Integer=0)
    desc_ref = Ref(ze_device_mem_alloc_desc_t(; flags, ordinal))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeMemAllocDevice(ctx, desc_ref, bytesize, alignment, dev, ptr_ref)

    return DeviceBuffer(reinterpret(ZePtr{Cvoid}, ptr_ref[]), bytesize, ctx, dev)
end

Base.pointer(buf::DeviceBuffer) = buf.ptr
Base.sizeof(buf::DeviceBuffer) = buf.bytesize
context(buf::DeviceBuffer) = buf.context

Base.show(io::IO, buf::DeviceBuffer) =
    @printf(io, "DeviceBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), pointer(buf))

Base.convert(::Type{ZePtr{T}}, buf::DeviceBuffer) where {T} =
    convert(ZePtr{T}, pointer(buf))


## host buffer

"""
    HostBuffer

A buffer of memory on the host. May be accessed by the host, and all devices within the
host driver. Frequently used as staging areas to transfer data to or from devices.

Note that these buffers need to be made resident to the device, e.g., by using the
ZE_KERNEL_FLAG_FORCE_RESIDENCY module flag, the ZE_KERNEL_SET_ATTR_INDIRECT_HOST_ACCESS
kernel attribute, or by calling zeDeviceMakeMemoryResident.
"""
struct HostBuffer <: AbstractBuffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    context::ZeContext
end

function host_alloc(ctx::ZeContext, bytesize::Integer, alignment::Integer=1; flags=0)
    desc_ref = Ref(ze_host_mem_alloc_desc_t(; flags))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeMemAllocHost(ctx, desc_ref, bytesize, alignment, ptr_ref)

    return HostBuffer(ptr_ref[], bytesize, ctx)
end

Base.pointer(buf::HostBuffer) = buf.ptr
Base.sizeof(buf::HostBuffer) = buf.bytesize
context(buf::HostBuffer) = buf.context

Base.show(io::IO, buf::HostBuffer) =
    @printf(io, "HostBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buf))

Base.convert(::Type{ZePtr{T}}, buf::HostBuffer) where {T} =
    reinterpret(ZePtr{T}, pointer(buf))


## shared buffer

"""
    SharedBuffer

A managed buffer that is shared between the host and one or more devices.
"""
struct SharedBuffer <: AbstractBuffer
    ptr::ZePtr{Cvoid}
    bytesize::Int
    context::ZeContext
    device::Union{Nothing,ZeDevice}
end

function shared_alloc(ctx::ZeContext, dev::Union{Nothing,ZeDevice}, bytesize::Integer,
                      alignment::Integer=1; host_flags=0,
                      device_flags=0, ordinal::Integer=0)
    device_desc_ref = Ref(ze_device_mem_alloc_desc_t(; flags=device_flags, ordinal))
    host_desc_ref = Ref(ze_host_mem_alloc_desc_t(; flags=host_flags))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeMemAllocShared(ctx, device_desc_ref, host_desc_ref, bytesize, alignment,
                     something(dev, C_NULL), ptr_ref)

    return SharedBuffer(reinterpret(ZePtr{Cvoid}, ptr_ref[]), bytesize, ctx, dev)
end

Base.pointer(buf::SharedBuffer) = buf.ptr
Base.sizeof(buf::SharedBuffer) = buf.bytesize
context(buf::SharedBuffer) = buf.context

Base.show(io::IO, buf::SharedBuffer) =
    @printf(io, "SharedBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::SharedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buf)))

Base.convert(::Type{ZePtr{T}}, buf::SharedBuffer) where {T} =
    convert(ZePtr{T}, pointer(buf))


## properties

function properties(buf::AbstractBuffer)
    props_ref = Ref(ze_memory_allocation_properties_t())
    dev_ref = Ref(ze_device_handle_t())
    zeMemGetAllocProperties(buf.context, pointer(buf), props_ref, dev_ref)

    props = props_ref[]
    return (
        device=ZeDevice(dev_ref[], buf.context.driver),
        type=props.type,
        id=props.id,
    )
end

struct UnknownBuffer <: AbstractBuffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    context::ZeContext
end

Base.pointer(buf::UnknownBuffer) = buf.ptr
Base.sizeof(buf::UnknownBuffer) = buf.bytesize
context(buf::UnknownBuffer) = buf.context

Base.show(io::IO, buf::UnknownBuffer) =
    @printf(io, "UnknownBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

function lookup_alloc(ctx::ZeContext, ptr::Union{Ptr,ZePtr})
    base_ref = Ref{Ptr{Cvoid}}()
    bytesize_ref = Ref{Csize_t}()
    zeMemGetAddressRange(ctx, ptr, base_ref, bytesize_ref)

    buf = UnknownBuffer(base_ref[], bytesize_ref[], ctx)
    props = properties(buf)
    return if props.type == ZE_MEMORY_TYPE_HOST
        HostBuffer(pointer(buf), sizeof(buf), ctx)
    elseif props.type == ZE_MEMORY_TYPE_DEVICE
        DeviceBuffer(reinterpret(ZePtr{Cvoid}, pointer(buf)), sizeof(buf), ctx, props.device)
    elseif props.type == ZE_MEMORY_TYPE_SHARED
        SharedBuffer(reinterpret(ZePtr{Cvoid}, pointer(buf)), sizeof(buf), ctx, props.device)
    else
        buf
    end
end
