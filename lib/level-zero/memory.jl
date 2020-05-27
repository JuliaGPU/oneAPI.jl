# Raw memory management

export device_alloc, host_alloc, shared_alloc, free, properties, lookup_alloc


#
# untyped buffers
#

abstract type Buffer end

# expected interface:
# - similar()
# - ptr, bytesize and drv fields
# - convert() to certain pointers

Base.pointer(buf::Buffer) = buf.ptr

Base.sizeof(buf::Buffer) = buf.bytesize

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(P::Type{<:Ptr},   buf::Buffer) = convert(P, buf)
Base.unsafe_convert(P::Type{<:ZePtr}, buf::Buffer) = convert(P, buf)

free(buf::Buffer) = zeDriverFreeMem(buf.drv, pointer(buf))

function Base.show(io::IO, ::MIME"text/plain", buf::Buffer)
    print(io, Base.format_bytes(sizeof(buf)), " ", nameof(typeof(buf)),
          " at 0x", string(UInt(pointer(buf)), base=16))
end


## device buffer

"""
    DeviceBuffer

A buffer of device memory, owned by a specific device. Generally, may only be accessed by
the device that owns it.
"""
struct DeviceBuffer <: Buffer
    ptr::ZePtr{Cvoid}
    bytesize::Int
    drv::ZeDriver
end

Base.similar(buf::DeviceBuffer, ptr::ZePtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), drv::ZeDriver=buf.drv) =
    DeviceBuffer(ptr, bytesize, ctx)

Base.convert(::Type{<:Ptr}, buf::DeviceBuffer) =
    throw(ArgumentError("cannot take the host address of a device buffer"))

Base.convert(::Type{ZePtr{T}}, buf::DeviceBuffer) where {T} =
    convert(ZePtr{T}, pointer(buf))


function device_alloc(dev::ZeDevice, bytesize::Integer, alignment::Integer=1;
                      flags=ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, ordinal::Integer=0)
    desc_ref = Ref(ze_device_mem_alloc_desc_t(
        ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT,
        flags, ordinal
    ))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeDriverAllocDeviceMem(dev.driver, desc_ref, bytesize, alignment, dev, ptr_ref)

    return DeviceBuffer(reinterpret(ZePtr{Cvoid}, ptr_ref[]), bytesize, dev.driver)
end


## host buffer

"""
    HostBuffer

A buffer of memory on the host. May be accessed by the host, and all devices within the
host driver. Frequently used as staging areas to transfer data to or from devices.

Note that these buffers need to be made resident to the device, e.g., by using the
ZE_KERNEL_FLAG_FORCE_RESIDENCY module flag, the ZE_KERNEL_SET_ATTR_INDIRECT_HOST_ACCESS
kernel attribute, or by calling zeDeviceMakeMemoryResident.
"""
struct HostBuffer <: Buffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    drv::ZeDriver
end

Base.similar(buf::HostBuffer, ptr::Ptr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), drv::ZeDriver=buf.drv) =
    HostBuffer(ptr, bytesize, ctx, mapped)

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buf))

Base.convert(::Type{ZePtr{T}}, buf::HostBuffer) where {T} =
    reinterpret(ZePtr{T}, pointer(buf))


function host_alloc(drv::ZeDriver, bytesize::Integer, alignment::Integer=1;
                    flags=ZE_HOST_MEM_ALLOC_FLAG_DEFAULT)
    desc_ref = Ref(ze_host_mem_alloc_desc_t(
        ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT,
        flags
    ))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeDriverAllocHostMem(drv, desc_ref, bytesize, alignment, ptr_ref)

    return HostBuffer(ptr_ref[], bytesize, drv)
end


## shared buffer

"""
    SharedBuffer

A managed buffer that is shared between the host and one or more devices.
"""
struct SharedBuffer <: Buffer
    ptr::ZePtr{Cvoid}
    bytesize::Int
    drv::ZeDriver
end

Base.similar(buf::SharedBuffer, ptr::ZePtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), dev::ZeDriver=buf.dev) =
    SharedBuffer(ptr, bytesize, ctx)

Base.convert(::Type{Ptr{T}}, buf::SharedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buf)))

Base.convert(::Type{ZePtr{T}}, buf::SharedBuffer) where {T} =
    convert(ZePtr{T}, pointer(buf))


function shared_alloc(drv::ZeDriver, dev::Union{Nothing,ZeDevice}, bytesize::Integer,
                      alignment::Integer=1; host_flags=ZE_HOST_MEM_ALLOC_FLAG_DEFAULT,
                      device_flags=ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, ordinal::Integer=0)
    device_desc_ref = Ref(ze_device_mem_alloc_desc_t(
        ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT,
        device_flags, ordinal
    ))
    host_desc_ref = Ref(ze_host_mem_alloc_desc_t(
        ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT,
        host_flags
    ))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeDriverAllocSharedMem(drv, device_desc_ref, host_desc_ref, bytesize, alignment,
                           something(dev, C_NULL), ptr_ref)

    return SharedBuffer(reinterpret(ZePtr{Cvoid}, ptr_ref[]), bytesize, drv)
end


## properties

function properties(buf::Buffer)
    props_ref = Ref{ze_memory_allocation_properties_t}()
    dev_ref = Ref{ze_device_handle_t}(C_NULL)
    unsafe_store!(convert(Ptr{ze_memory_allocation_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_MEMORY_ALLOCATION_PROPERTIES_VERSION_CURRENT)
    zeDriverGetMemAllocProperties(buf.drv, pointer(buf), props_ref, dev_ref)

    props = props_ref[]
    return (
        device=ZeDevice(dev_ref[], buf.drv),
        type=props.type,
        id=props.id,
    )
end

struct UnknownBuffer <: Buffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    drv::ZeDriver
end

function lookup_alloc(drv::ZeDriver, ptr::Union{Ptr,ZePtr})
    base_ref = Ref{Ptr{Cvoid}}()
    bytesize_ref = Ref{Csize_t}()
    zeDriverGetMemAddressRange(drv, ptr, base_ref, bytesize_ref)

    buf = UnknownBuffer(base_ref[], bytesize_ref[], drv)
    props = properties(buf)
    return if props.type == ZE_MEMORY_TYPE_HOST
        HostBuffer(pointer(buf), sizeof(buf), buf.drv)
    elseif props.type == ZE_MEMORY_TYPE_DEVICE
        DeviceBuffer(reinterpret(ZePtr{Cvoid}, pointer(buf)), sizeof(buf), buf.drv)
    elseif props.type == ZE_MEMORY_TYPE_SHARED
        SharedBuffer(reinterpret(ZePtr{Cvoid}, pointer(buf)), sizeof(buf), buf.drv)
    else
        buf
    end
end
