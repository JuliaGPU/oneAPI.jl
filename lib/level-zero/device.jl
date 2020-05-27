export ZeDevice, properties, compute_properties, kernel_properties, memory_properties, memory_access_properties, cache_properties, image_properties, p2p_properties

struct ZeDevice
    handle::ze_device_handle_t

    driver::ZeDriver
end

Base.unsafe_convert(::Type{ze_device_handle_t}, dev::ZeDevice) = dev.handle

function Base.show(io::IO, ::MIME"text/plain", dev::ZeDevice)
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
    print(io, "): $(props.name)")
end


## properties

function properties(dev::ZeDevice)
    props_ref = Ref{ze_device_properties_t}()
    unsafe_store!(convert(Ptr{ze_device_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_PROPERTIES_VERSION_CURRENT)
    zeDeviceGetProperties(dev, props_ref)

    props = props_ref[]
    return (
        type=props.type,
        vendorId=UInt16(props.vendorId),
        deviceId=UInt16(props.deviceId),
        uuid=Base.UUID(reinterpret(UInt128, [props.uuid.id...])[1]),
        subdeviceId=Bool(props.isSubdevice) ? props.subdeviceId : nothing,
        coreClockRate=Int(props.coreClockRate),
        unifiedMemorySupported=Bool(props.unifiedMemorySupported),
        eccMemorySupported=Bool(props.eccMemorySupported),
        onDemandPageFaultsSupported=Bool(props.onDemandPageFaultsSupported),
        maxCommandQueues=Int(props.maxCommandQueues),
        numAsyncComputeEngines=Int(props.numAsyncComputeEngines),
        numAsyncCopyEngines=Int(props.numAsyncCopyEngines),
        maxCommandQueuePriority=Int(props.maxCommandQueuePriority),
        numThreadsPerEU=Int(props.numThreadsPerEU),
        physicalEUSimdWidth=Int(props.physicalEUSimdWidth),
        numEUsPerSubslice=Int(props.numEUsPerSubslice),
        numSubslicesPerSlice=Int(props.numSubslicesPerSlice),
        numSlices=Int(props.numSlices),
        timerResolution=Int(props.timerResolution),
        name=String([props.name[1:findfirst(isequal(0), props.name)-1]...]),
    )
end

function compute_properties(dev::ZeDevice)
    props_ref = Ref{ze_device_compute_properties_t}()
    unsafe_store!(convert(Ptr{ze_device_compute_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_COMPUTE_PROPERTIES_VERSION_CURRENT)
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

function kernel_properties(dev::ZeDevice)
    props_ref = Ref{ze_device_kernel_properties_t}()
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), props_ref, 0,
          sizeof(ze_device_kernel_properties_t)) # oneapi-src/level-zero#8
    unsafe_store!(convert(Ptr{ze_device_kernel_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_KERNEL_PROPERTIES_VERSION_CURRENT)
    zeDeviceGetKernelProperties(dev, props_ref)

    props = props_ref[]
    return (
        spirvVersionSupported=props.spirvVersionSupported==0 ? nothing : unmake_version(props.spirvVersionSupported),
        nativeKernelSupported=Base.UUID(reinterpret(UInt128, [props.nativeKernelSupported.id...])[1]),
        fp16Supported=Bool(props.fp16Supported),
        fp64Supported=Bool(props.fp64Supported),
        int64AtomicsSupported=Bool(props.int64AtomicsSupported),
        dp4aSupported=Bool(props.dp4aSupported),
        halfFpCapabilities=Int(props.halfFpCapabilities),
        singleFpCapabilities=Int(props.singleFpCapabilities),
        doubleFpCapabilities=Int(props.doubleFpCapabilities),
        maxArgumentsSize=Int(props.maxArgumentsSize),
        printfBufferSize=Int(props.printfBufferSize),
    )
end

function memory_properties(dev::ZeDevice)
    count_ref = Ref{UInt32}(0)
    zeDeviceGetMemoryProperties(dev, count_ref, C_NULL)

    all_props = Vector{ze_device_memory_properties_t}(undef, count_ref[])
    for i in 1:count_ref[]
        unsafe_store!(convert(Ptr{ze_device_memory_properties_version_t},
                              pointer(all_props, i)),
                      ZE_DEVICE_MEMORY_PROPERTIES_VERSION_CURRENT)
    end
    zeDeviceGetMemoryProperties(dev, count_ref, all_props)

    return [(maxClockRate=Int(props.maxClockRate),
             maxBusWidth=Int(props.maxBusWidth),
             totalSize=Int(props.totalSize),
            ) for props in all_props[1:count_ref[]]]
end

function memory_access_properties(dev::ZeDevice)
    props_ref = Ref{ze_device_memory_access_properties_t}()
    unsafe_store!(convert(Ptr{ze_device_memory_access_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_MEMORY_ACCESS_PROPERTIES_VERSION_CURRENT)
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
    props_ref = Ref{ze_device_cache_properties_t}()
    unsafe_store!(convert(Ptr{ze_device_cache_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_CACHE_PROPERTIES_VERSION_CURRENT)
    zeDeviceGetCacheProperties(dev, props_ref)

    props = props_ref[]
    return (
        intermediateCacheControlSupported=Bool(props.intermediateCacheControlSupported),
        intermediateCacheSize=Int(props.intermediateCacheSize),
        intermediateCachelineSize=Int(props.intermediateCachelineSize),
        lastLevelCacheSizeControlSupported=Bool(props.lastLevelCacheSizeControlSupported),
        lastLevelCacheSize=Int(props.lastLevelCacheSize),
        lastLevelCachelineSize=Int(props.lastLevelCachelineSize),
    )
end

function image_properties(dev::ZeDevice)
    props_ref = Ref{ze_device_image_properties_t}()
    unsafe_store!(convert(Ptr{ze_device_image_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_IMAGE_PROPERTIES_VERSION_CURRENT)
    zeDeviceGetImageProperties(dev, props_ref)

    props = props_ref[]
    return (
        supported=Bool(props.supported),
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
    props_ref = Ref{ze_device_p2p_properties_t}()
    unsafe_store!(convert(Ptr{ze_device_p2p_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_DEVICE_P2P_PROPERTIES_VERSION_CURRENT)
    zeDeviceGetP2PProperties(dev1, dev2, props_ref)

    props = props_ref[]
    return (
        accessSupported=Bool(props.accessSupported),
        atomicsSupported=Bool(props.atomicsSupported),
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

        handles = Vector{ze_device_handle_t}(undef, count_ref[])
        zeDeviceGet(drv, count_ref, handles)

        new(handles, drv)
    end
end

devices(drv) = ZeDevices(drv)

Base.eltype(::ZeDevices) = ZeDevice

function Base.iterate(iter::ZeDevices, i=1)
    i >= length(iter) + 1 ? nothing : (ZeDevice(iter.handles[i], iter.driver), i+1)
end

Base.length(iter::ZeDevices) = length(iter.handles)

Base.IteratorSize(::ZeDevices) = Base.HasLength()
