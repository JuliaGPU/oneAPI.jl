export ZeDevice, properties, compute_properties, module_properties, memory_properties, memory_access_properties, cache_properties, image_properties, p2p_properties

struct ZeDevice
    handle::ze_device_handle_t

    driver::ZeDriver

    # only accept handles, don't convert
    ZeDevice(handle::ze_device_handle_t, driver::ZeDriver) = new(handle, driver)
end

Base.unsafe_convert(::Type{ze_device_handle_t}, dev::ZeDevice) = dev.handle

Base.:(==)(a::ZeDevice, b::ZeDevice) = a.handle == b.handle
Base.hash(e::ZeDevice, h::UInt) = hash(e.handle, h)

function Base.show(io::IO, dev::ZeDevice)
    props = properties(dev)
    print(io, "ZeDevice(")
    if props.type == ZE_DEVICE_TYPE_GPU
        print(io, "GPU")
    elseif props.type == ZE_DEVICE_TYPE_FPGA
        print(io, "FPGA")
    end
    print(io, ", vendor ")
    show(io, props.vendorId)
    print(io, ", device ")
    show(io, props.deviceId)
    if props.subdeviceId !== nothing
        print(io, ", sub-device ")
        show(io, props.subdeviceId)
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", dev::ZeDevice)
    show(io, dev)
    props = properties(dev)
    print(io, ": $(props.name)")
end


## properties

function properties(dev::ZeDevice)
    props_ref = Ref(ze_device_properties_t())
    zeDeviceGetProperties(dev, props_ref)

    props = props_ref[]
    return (
        type=props.type,
        vendorId=UInt16(props.vendorId),
        deviceId=UInt16(props.deviceId),
        flags=props.flags,
        subdeviceId=(props.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE == 0) ? nothing : props.subdeviceId,
        coreClockRate=Int(props.coreClockRate),
        maxMemAllocSize=Int(props.maxMemAllocSize),
        maxHardwareContexts=Int(props.maxHardwareContexts),
        maxCommandQueuePriority=Int(props.maxCommandQueuePriority),
        numThreadsPerEU=Int(props.numThreadsPerEU),
        physicalEUSimdWidth=Int(props.physicalEUSimdWidth),
        numEUsPerSubslice=Int(props.numEUsPerSubslice),
        numSubslicesPerSlice=Int(props.numSubslicesPerSlice),
        numSlices=Int(props.numSlices),
        timerResolution=Int(props.timerResolution),
        timestampValidBits=Int(props.timestampValidBits),
        kernelTimestampValidBits=Int(props.kernelTimestampValidBits),
        uuid=Base.UUID(reinterpret(UInt128, [props.uuid.id...])[1]),
        name=String(UInt8[props.name[1:findfirst(isequal(0), props.name)-1]...]),
    )
end

function compute_properties(dev::ZeDevice)
    props_ref = Ref(ze_device_compute_properties_t())
    zeDeviceGetComputeProperties(dev, props_ref)

    props = props_ref[]
    return (
        maxTotalGroupSize=Int(props.maxTotalGroupSize),
        maxGroupSizeX=Int(props.maxGroupSizeX),
        maxGroupSizeY=Int(props.maxGroupSizeY),
        maxGroupSizeZ=Int(props.maxGroupSizeZ),
        maxGroupCountX=Int(props.maxGroupCountX),
        maxGroupCountY=Int(props.maxGroupCountY),
        maxGroupCountZ=Int(props.maxGroupCountZ),
        maxSharedLocalMemory=Int(props.maxSharedLocalMemory),
        subGroupSizes=Int.(props.subGroupSizes[1:props.numSubGroupSizes]),
    )
end

function module_properties(dev::ZeDevice)
    props_ref = Ref(ze_device_module_properties_t())
    zeDeviceGetModuleProperties(dev, props_ref)

    props = props_ref[]
    return (
        spirvVersionSupported=props.spirvVersionSupported==0 ? nothing : unmake_version(props.spirvVersionSupported),
        flags=props.flags,
        fp16flags=props.fp16flags,
        fp32flags=props.fp32flags,
        fp64flags=props.fp64flags,
        maxArgumentsSize=Int(props.maxArgumentsSize),
        printfBufferSize=Int(props.printfBufferSize),
        nativeKernelSupported=Base.UUID(reinterpret(UInt128, [props.nativeKernelSupported.id...])[1]),
    )
end

function memory_properties(dev::ZeDevice)
    count_ref = Ref{UInt32}(0)
    zeDeviceGetMemoryProperties(dev, count_ref, C_NULL)

    all_props = fill(ze_device_memory_properties_t(), count_ref[])
    zeDeviceGetMemoryProperties(dev, count_ref, all_props)

    return [(maxClockRate=Int(props.maxClockRate),
             maxBusWidth=Int(props.maxBusWidth),
             totalSize=Int(props.totalSize),
            ) for props in all_props[1:count_ref[]]]
end

function memory_access_properties(dev::ZeDevice)
    props_ref = Ref(ze_device_memory_access_properties_t())
    zeDeviceGetMemoryAccessProperties(dev, props_ref)

    props = props_ref[]
    return (
        hostAllocCapabilities=Int(props.hostAllocCapabilities),
        deviceAllocCapabilities=Int(props.deviceAllocCapabilities),
        sharedSingleDeviceAllocCapabilities=Int(props.sharedSingleDeviceAllocCapabilities),
        sharedCrossDeviceAllocCapabilities=Int(props.sharedCrossDeviceAllocCapabilities),
        sharedSystemAllocCapabilities=Int(props.sharedSystemAllocCapabilities),
    )
end

function cache_properties(dev::ZeDevice)
    count_ref = Ref{UInt32}(0)
    zeDeviceGetCacheProperties(dev, count_ref, C_NULL)

    all_props = fill(ze_device_cache_properties_t(), count_ref[])
    zeDeviceGetCacheProperties(dev, count_ref, all_props)

    return [(flags=props.flags,
             cacheSize=Int(props.cacheSize),
            ) for props in all_props[1:count_ref[]]]
end

function image_properties(dev::ZeDevice)
    props_ref = Ref(ze_device_image_properties_t())
    zeDeviceGetImageProperties(dev, props_ref)

    props = props_ref[]
    return (
        maxImageDims1D=Int(props.maxImageDims1D),
        maxImageDims2D=Int(props.maxImageDims2D),
        maxImageDims3D=Int(props.maxImageDims3D),
        maxImageBufferSize=Int(props.maxImageBufferSize),
        maxImageArraySlices=Int(props.maxImageArraySlices),
        maxSamplers=Int(props.maxSamplers),
        maxReadImageArgs=Int(props.maxReadImageArgs),
        maxWriteImageArgs=Int(props.maxWriteImageArgs),
    )
end

function p2p_properties(dev1, dev2::ZeDevice)
    props_ref = Ref(ze_device_p2p_properties_t())
    zeDeviceGetP2PProperties(dev1, dev2, props_ref)

    props = props_ref[]
    return (
        flags=props.flags,
    )
end


## device iteration

export devices

struct ZeDevices
    handles::Vector{ze_device_handle_t}

    driver::ZeDriver

    function ZeDevices(drv::ZeDriver)
        count_ref = Ref{UInt32}(0)
        zeDeviceGet(drv, count_ref, C_NULL)

        handles = fill(ze_device_handle_t(), count_ref[])
        zeDeviceGet(drv, count_ref, handles)

        new(handles, drv)
    end
end

devices(drv::ZeDriver) = ZeDevices(drv)

Base.eltype(::ZeDevices) = ZeDevice

function Base.iterate(iter::ZeDevices, i=1)
    i >= length(iter) + 1 ? nothing : (ZeDevice(iter.handles[i], iter.driver), i+1)
end

Base.length(iter::ZeDevices) = length(iter.handles)

Base.IteratorSize(::ZeDevices) = Base.HasLength()

function Base.show(io::IO, ::MIME"text/plain", iter::ZeDevices)
    print(io, "ZeDevice iterator for $(length(iter)) devices")
    if !isempty(iter)
        print(io, ":")
        for (i,dev) in enumerate(iter)
            print(io, "\n$(i). $(properties(dev).name)")
        end
    end
end

Base.getindex(iter::ZeDevices, i::Integer) =
    ZeDevice(iter.handles[i], iter.driver)
