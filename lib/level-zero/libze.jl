using CEnum

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == RESULT_ERROR_OUT_OF_HOST_MEMORY || res == RESULT_ERROR_OUT_OF_DEVICE_MEMORY
        throw(OutOfGPUMemoryError())
    else
        throw(ZeError(res))
    end
end

function check(f)
    res = retry_reclaim(err -> err == RESULT_ERROR_OUT_OF_HOST_MEMORY ||
                               err == RESULT_ERROR_OUT_OF_DEVICE_MEMORY) do
        return f()
    end

    if res != RESULT_SUCCESS
        throw_api_error(res)
    end

    return
end

const ze_bool_t = UInt8

mutable struct _ze_driver_handle_t end

const ze_driver_handle_t = Ptr{_ze_driver_handle_t}

mutable struct _ze_device_handle_t end

const ze_device_handle_t = Ptr{_ze_device_handle_t}

mutable struct _ze_context_handle_t end

const ze_context_handle_t = Ptr{_ze_context_handle_t}

mutable struct _ze_command_queue_handle_t end

const ze_command_queue_handle_t = Ptr{_ze_command_queue_handle_t}

mutable struct _ze_command_list_handle_t end

const ze_command_list_handle_t = Ptr{_ze_command_list_handle_t}

mutable struct _ze_fence_handle_t end

const ze_fence_handle_t = Ptr{_ze_fence_handle_t}

mutable struct _ze_event_pool_handle_t end

const ze_event_pool_handle_t = Ptr{_ze_event_pool_handle_t}

mutable struct _ze_event_handle_t end

const ze_event_handle_t = Ptr{_ze_event_handle_t}

mutable struct _ze_image_handle_t end

const ze_image_handle_t = Ptr{_ze_image_handle_t}

mutable struct _ze_module_handle_t end

const ze_module_handle_t = Ptr{_ze_module_handle_t}

mutable struct _ze_module_build_log_handle_t end

const ze_module_build_log_handle_t = Ptr{_ze_module_build_log_handle_t}

mutable struct _ze_kernel_handle_t end

const ze_kernel_handle_t = Ptr{_ze_kernel_handle_t}

mutable struct _ze_sampler_handle_t end

const ze_sampler_handle_t = Ptr{_ze_sampler_handle_t}

mutable struct _ze_physical_mem_handle_t end

const ze_physical_mem_handle_t = Ptr{_ze_physical_mem_handle_t}

mutable struct _ze_fabric_vertex_handle_t end

const ze_fabric_vertex_handle_t = Ptr{_ze_fabric_vertex_handle_t}

mutable struct _ze_fabric_edge_handle_t end

const ze_fabric_edge_handle_t = Ptr{_ze_fabric_edge_handle_t}

struct _ze_ipc_mem_handle_t
    data::NTuple{64,Cchar}
end

const ze_ipc_mem_handle_t = _ze_ipc_mem_handle_t

struct _ze_ipc_event_pool_handle_t
    data::NTuple{64,Cchar}
end

const ze_ipc_event_pool_handle_t = _ze_ipc_event_pool_handle_t

@cenum _ze_result_t::UInt32 begin
    ZE_RESULT_SUCCESS = 0
    ZE_RESULT_NOT_READY = 1
    ZE_RESULT_ERROR_DEVICE_LOST = 1879048193
    ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 1879048194
    ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 1879048195
    ZE_RESULT_ERROR_MODULE_BUILD_FAILURE = 1879048196
    ZE_RESULT_ERROR_MODULE_LINK_FAILURE = 1879048197
    ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET = 1879048198
    ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 1879048199
    ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX = 2146435073
    ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE = 2146435074
    ZE_RESULT_EXP_ERROR_REMOTE_DEVICE = 2146435075
    ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE = 2146435076
    ZE_RESULT_EXP_RTAS_BUILD_RETRY = 2146435077
    ZE_RESULT_EXP_RTAS_BUILD_DEFERRED = 2146435078
    ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 1879113728
    ZE_RESULT_ERROR_NOT_AVAILABLE = 1879113729
    ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 1879179264
    ZE_RESULT_WARNING_DROPPED_DATA = 1879179265
    ZE_RESULT_ERROR_UNINITIALIZED = 2013265921
    ZE_RESULT_ERROR_UNSUPPORTED_VERSION = 2013265922
    ZE_RESULT_ERROR_UNSUPPORTED_FEATURE = 2013265923
    ZE_RESULT_ERROR_INVALID_ARGUMENT = 2013265924
    ZE_RESULT_ERROR_INVALID_NULL_HANDLE = 2013265925
    ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE = 2013265926
    ZE_RESULT_ERROR_INVALID_NULL_POINTER = 2013265927
    ZE_RESULT_ERROR_INVALID_SIZE = 2013265928
    ZE_RESULT_ERROR_UNSUPPORTED_SIZE = 2013265929
    ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 2013265930
    ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 2013265931
    ZE_RESULT_ERROR_INVALID_ENUMERATION = 2013265932
    ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION = 2013265933
    ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 2013265934
    ZE_RESULT_ERROR_INVALID_NATIVE_BINARY = 2013265935
    ZE_RESULT_ERROR_INVALID_GLOBAL_NAME = 2013265936
    ZE_RESULT_ERROR_INVALID_KERNEL_NAME = 2013265937
    ZE_RESULT_ERROR_INVALID_FUNCTION_NAME = 2013265938
    ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION = 2013265939
    ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 2013265940
    ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 2013265941
    ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 2013265942
    ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 2013265943
    ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED = 2013265944
    ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 2013265945
    ZE_RESULT_ERROR_OVERLAPPING_REGIONS = 2013265946
    ZE_RESULT_WARNING_ACTION_REQUIRED = 2013265947
    ZE_RESULT_ERROR_INVALID_KERNEL_HANDLE = 2013265948
    ZE_RESULT_EXT_RTAS_BUILD_RETRY = 2013265949
    ZE_RESULT_EXT_RTAS_BUILD_DEFERRED = 2013265950
    ZE_RESULT_EXT_ERROR_OPERANDS_INCOMPATIBLE = 2013265951
    ZE_RESULT_ERROR_SURVIVABILITY_MODE_DETECTED = 2013265952
    ZE_RESULT_ERROR_UNKNOWN = 2147483646
    ZE_RESULT_FORCE_UINT32 = 2147483647
end

const ze_result_t = _ze_result_t

@cenum _ze_structure_type_t::UInt32 begin
    ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 1
    ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES = 2
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 3
    ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 4
    ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 5
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 6
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 7
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES = 8
    ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES = 9
    ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES = 10
    ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES = 11
    ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 12
    ZE_STRUCTURE_TYPE_CONTEXT_DESC = 13
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 14
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 15
    ZE_STRUCTURE_TYPE_EVENT_POOL_DESC = 16
    ZE_STRUCTURE_TYPE_EVENT_DESC = 17
    ZE_STRUCTURE_TYPE_FENCE_DESC = 18
    ZE_STRUCTURE_TYPE_IMAGE_DESC = 19
    ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES = 20
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 21
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 22
    ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 23
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 24
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD = 25
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD = 26
    ZE_STRUCTURE_TYPE_MODULE_DESC = 27
    ZE_STRUCTURE_TYPE_MODULE_PROPERTIES = 28
    ZE_STRUCTURE_TYPE_KERNEL_DESC = 29
    ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES = 30
    ZE_STRUCTURE_TYPE_SAMPLER_DESC = 31
    ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 32
    ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES = 33
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32 = 34
    ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32 = 35
    ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES = 65537
    ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC = 65538
    ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES = 65539
    ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC = 65540
    ZE_STRUCTURE_TYPE_EU_COUNT_EXT = 65541
    ZE_STRUCTURE_TYPE_SRGB_EXT_DESC = 65542
    ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC = 65543
    ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES = 65544
    ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES = 65545
    ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC = 65546
    ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC = 65547
    ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES = 65548
    ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES = 65549
    ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES = 65550
    ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT = 65551
    ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC = 65552
    ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES = 65553
    ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES = 65554
    ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES = 65555
    ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 131073
    ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC = 131074
    ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES = 131075
    ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC = 131076
    ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC = 131077
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 = 131078
    ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES = 131079
    ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC = 131080
    ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES = 131081
    ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES = 131082
    ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES = 131083
    ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES = 131084
    ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES = 131085
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC = 131086
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC = 131087
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES = 131088
    ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES = 131089
    ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES = 131090
    ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS = 131091
    ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC = 131092
    ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_PROPERTIES = 131093
    ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC = 131094
    ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC = 131095
    ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC = 131096
    ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC = 131097
    ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC = 131098
    ZE_STRUCTURE_TYPE_MUTABLE_GROUP_SIZE_EXP_DESC = 131099
    ZE_STRUCTURE_TYPE_MUTABLE_GLOBAL_OFFSET_EXP_DESC = 131100
    ZE_STRUCTURE_TYPE_PITCHED_ALLOC_DEVICE_EXP_PROPERTIES = 131101
    ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC = 131102
    ZE_STRUCTURE_TYPE_PITCHED_IMAGE_EXP_DESC = 131103
    ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC = 131104
    ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC = 131105
    ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_EXT_DESC = 131106
    ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WIN32_EXT_DESC = 131107
    ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_FD_EXT_DESC = 131108
    ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXT = 131109
    ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXT = 131110
    ZE_STRUCTURE_TYPE_DRIVER_DDI_HANDLES_EXT_PROPERTIES = 131111
    ZE_STRUCTURE_TYPE_DEVICE_CACHELINE_SIZE_EXT = 131112
    ZE_STRUCTURE_TYPE_DEVICE_VECTOR_WIDTH_PROPERTIES_EXT = 131113
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXT_DESC = 131120
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXT_DESC = 131121
    ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXT_PROPERTIES = 131122
    ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXT_PROPERTIES = 131123
    ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXT_PROPERTIES = 131124
    ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXT_CB_PARAMS = 131125
    ZE_STRUCTURE_TYPE_FORCE_UINT32 = 2147483647
end

const ze_structure_type_t = _ze_structure_type_t

const ze_external_memory_type_flags_t = UInt32

@cenum _ze_external_memory_type_flag_t::UInt32 begin
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD = 1
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 2
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32 = 4
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT = 8
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE = 16
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D11_TEXTURE_KMT = 32
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_HEAP = 64
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE = 128
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_external_memory_type_flag_t = _ze_external_memory_type_flag_t

@cenum _ze_bandwidth_unit_t::UInt32 begin
    ZE_BANDWIDTH_UNIT_UNKNOWN = 0
    ZE_BANDWIDTH_UNIT_BYTES_PER_NANOSEC = 1
    ZE_BANDWIDTH_UNIT_BYTES_PER_CLOCK = 2
    ZE_BANDWIDTH_UNIT_FORCE_UINT32 = 2147483647
end

const ze_bandwidth_unit_t = _ze_bandwidth_unit_t

@cenum _ze_latency_unit_t::UInt32 begin
    ZE_LATENCY_UNIT_UNKNOWN = 0
    ZE_LATENCY_UNIT_NANOSEC = 1
    ZE_LATENCY_UNIT_CLOCK = 2
    ZE_LATENCY_UNIT_HOP = 3
    ZE_LATENCY_UNIT_FORCE_UINT32 = 2147483647
end

const ze_latency_unit_t = _ze_latency_unit_t

struct _ze_uuid_t
    id::NTuple{16,UInt8}
end

const ze_uuid_t = _ze_uuid_t

struct _ze_base_cb_params_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
end

const ze_base_cb_params_t = _ze_base_cb_params_t

struct _ze_base_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
end

const ze_base_properties_t = _ze_base_properties_t

struct _ze_base_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
end

const ze_base_desc_t = _ze_base_desc_t

const ze_init_driver_type_flags_t = UInt32

struct _ze_init_driver_type_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_init_driver_type_flags_t
end

const ze_init_driver_type_desc_t = _ze_init_driver_type_desc_t

struct _ze_driver_uuid_t
    id::NTuple{16,UInt8}
end

const ze_driver_uuid_t = _ze_driver_uuid_t

struct _ze_driver_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    uuid::ze_driver_uuid_t
    driverVersion::UInt32
end

const ze_driver_properties_t = _ze_driver_properties_t

const ze_ipc_property_flags_t = UInt32

struct _ze_driver_ipc_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_ipc_property_flags_t
end

const ze_driver_ipc_properties_t = _ze_driver_ipc_properties_t

struct _ze_driver_extension_properties_t
    name::NTuple{256,Cchar}
    version::UInt32
end

const ze_driver_extension_properties_t = _ze_driver_extension_properties_t

struct _ze_device_uuid_t
    id::NTuple{16,UInt8}
end

const ze_device_uuid_t = _ze_device_uuid_t

@cenum _ze_device_type_t::UInt32 begin
    ZE_DEVICE_TYPE_GPU = 1
    ZE_DEVICE_TYPE_CPU = 2
    ZE_DEVICE_TYPE_FPGA = 3
    ZE_DEVICE_TYPE_MCA = 4
    ZE_DEVICE_TYPE_VPU = 5
    ZE_DEVICE_TYPE_FORCE_UINT32 = 2147483647
end

const ze_device_type_t = _ze_device_type_t

const ze_device_property_flags_t = UInt32

struct _ze_device_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    type::ze_device_type_t
    vendorId::UInt32
    deviceId::UInt32
    flags::ze_device_property_flags_t
    subdeviceId::UInt32
    coreClockRate::UInt32
    maxMemAllocSize::UInt64
    maxHardwareContexts::UInt32
    maxCommandQueuePriority::UInt32
    numThreadsPerEU::UInt32
    physicalEUSimdWidth::UInt32
    numEUsPerSubslice::UInt32
    numSubslicesPerSlice::UInt32
    numSlices::UInt32
    timerResolution::UInt64
    timestampValidBits::UInt32
    kernelTimestampValidBits::UInt32
    uuid::ze_device_uuid_t
    name::NTuple{256,Cchar}
end

const ze_device_properties_t = _ze_device_properties_t

struct _ze_device_thread_t
    slice::UInt32
    subslice::UInt32
    eu::UInt32
    thread::UInt32
end

const ze_device_thread_t = _ze_device_thread_t

struct _ze_device_compute_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    maxTotalGroupSize::UInt32
    maxGroupSizeX::UInt32
    maxGroupSizeY::UInt32
    maxGroupSizeZ::UInt32
    maxGroupCountX::UInt32
    maxGroupCountY::UInt32
    maxGroupCountZ::UInt32
    maxSharedLocalMemory::UInt32
    numSubGroupSizes::UInt32
    subGroupSizes::NTuple{8,UInt32}
end

const ze_device_compute_properties_t = _ze_device_compute_properties_t

struct _ze_native_kernel_uuid_t
    id::NTuple{16,UInt8}
end

const ze_native_kernel_uuid_t = _ze_native_kernel_uuid_t

const ze_device_module_flags_t = UInt32

const ze_device_fp_flags_t = UInt32

struct _ze_device_module_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    spirvVersionSupported::UInt32
    flags::ze_device_module_flags_t
    fp16flags::ze_device_fp_flags_t
    fp32flags::ze_device_fp_flags_t
    fp64flags::ze_device_fp_flags_t
    maxArgumentsSize::UInt32
    printfBufferSize::UInt32
    nativeKernelSupported::ze_native_kernel_uuid_t
end

const ze_device_module_properties_t = _ze_device_module_properties_t

const ze_command_queue_group_property_flags_t = UInt32

struct _ze_command_queue_group_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_command_queue_group_property_flags_t
    maxMemoryFillPatternSize::Csize_t
    numQueues::UInt32
end

const ze_command_queue_group_properties_t = _ze_command_queue_group_properties_t

const ze_device_memory_property_flags_t = UInt32

struct _ze_device_memory_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_device_memory_property_flags_t
    maxClockRate::UInt32
    maxBusWidth::UInt32
    totalSize::UInt64
    name::NTuple{256,Cchar}
end

const ze_device_memory_properties_t = _ze_device_memory_properties_t

const ze_memory_access_cap_flags_t = UInt32

struct _ze_device_memory_access_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    hostAllocCapabilities::ze_memory_access_cap_flags_t
    deviceAllocCapabilities::ze_memory_access_cap_flags_t
    sharedSingleDeviceAllocCapabilities::ze_memory_access_cap_flags_t
    sharedCrossDeviceAllocCapabilities::ze_memory_access_cap_flags_t
    sharedSystemAllocCapabilities::ze_memory_access_cap_flags_t
end

const ze_device_memory_access_properties_t = _ze_device_memory_access_properties_t

const ze_device_cache_property_flags_t = UInt32

struct _ze_device_cache_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_device_cache_property_flags_t
    cacheSize::Csize_t
end

const ze_device_cache_properties_t = _ze_device_cache_properties_t

struct _ze_device_image_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    maxImageDims1D::UInt32
    maxImageDims2D::UInt32
    maxImageDims3D::UInt32
    maxImageBufferSize::UInt64
    maxImageArraySlices::UInt32
    maxSamplers::UInt32
    maxReadImageArgs::UInt32
    maxWriteImageArgs::UInt32
end

const ze_device_image_properties_t = _ze_device_image_properties_t

struct _ze_device_external_memory_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    memoryAllocationImportTypes::ze_external_memory_type_flags_t
    memoryAllocationExportTypes::ze_external_memory_type_flags_t
    imageImportTypes::ze_external_memory_type_flags_t
    imageExportTypes::ze_external_memory_type_flags_t
end

const ze_device_external_memory_properties_t = _ze_device_external_memory_properties_t

const ze_device_p2p_property_flags_t = UInt32

struct _ze_device_p2p_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_device_p2p_property_flags_t
end

const ze_device_p2p_properties_t = _ze_device_p2p_properties_t

const ze_context_flags_t = UInt32

struct _ze_context_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_context_flags_t
end

const ze_context_desc_t = _ze_context_desc_t

const ze_command_queue_flags_t = UInt32

@cenum _ze_command_queue_mode_t::UInt32 begin
    ZE_COMMAND_QUEUE_MODE_DEFAULT = 0
    ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2
    ZE_COMMAND_QUEUE_MODE_FORCE_UINT32 = 2147483647
end

const ze_command_queue_mode_t = _ze_command_queue_mode_t

@cenum _ze_command_queue_priority_t::UInt32 begin
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
    ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW = 1
    ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2
    ZE_COMMAND_QUEUE_PRIORITY_FORCE_UINT32 = 2147483647
end

const ze_command_queue_priority_t = _ze_command_queue_priority_t

struct _ze_command_queue_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    ordinal::UInt32
    index::UInt32
    flags::ze_command_queue_flags_t
    mode::ze_command_queue_mode_t
    priority::ze_command_queue_priority_t
end

const ze_command_queue_desc_t = _ze_command_queue_desc_t

const ze_command_list_flags_t = UInt32

struct _ze_command_list_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    commandQueueGroupOrdinal::UInt32
    flags::ze_command_list_flags_t
end

const ze_command_list_desc_t = _ze_command_list_desc_t

struct _ze_copy_region_t
    originX::UInt32
    originY::UInt32
    originZ::UInt32
    width::UInt32
    height::UInt32
    depth::UInt32
end

const ze_copy_region_t = _ze_copy_region_t

struct _ze_image_region_t
    originX::UInt32
    originY::UInt32
    originZ::UInt32
    width::UInt32
    height::UInt32
    depth::UInt32
end

const ze_image_region_t = _ze_image_region_t

const ze_event_pool_flags_t = UInt32

struct _ze_event_pool_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_event_pool_flags_t
    count::UInt32
end

const ze_event_pool_desc_t = _ze_event_pool_desc_t

const ze_event_scope_flags_t = UInt32

struct _ze_event_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    index::UInt32
    signal::ze_event_scope_flags_t
    wait::ze_event_scope_flags_t
end

const ze_event_desc_t = _ze_event_desc_t

struct _ze_kernel_timestamp_data_t
    kernelStart::UInt64
    kernelEnd::UInt64
end

const ze_kernel_timestamp_data_t = _ze_kernel_timestamp_data_t

struct _ze_kernel_timestamp_result_t
    _global::ze_kernel_timestamp_data_t
    context::ze_kernel_timestamp_data_t
end

const ze_kernel_timestamp_result_t = _ze_kernel_timestamp_result_t

const ze_fence_flags_t = UInt32

struct _ze_fence_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_fence_flags_t
end

const ze_fence_desc_t = _ze_fence_desc_t

@cenum _ze_image_format_layout_t::UInt32 begin
    ZE_IMAGE_FORMAT_LAYOUT_8 = 0
    ZE_IMAGE_FORMAT_LAYOUT_16 = 1
    ZE_IMAGE_FORMAT_LAYOUT_32 = 2
    ZE_IMAGE_FORMAT_LAYOUT_8_8 = 3
    ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 = 4
    ZE_IMAGE_FORMAT_LAYOUT_16_16 = 5
    ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 = 6
    ZE_IMAGE_FORMAT_LAYOUT_32_32 = 7
    ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 = 8
    ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2 = 9
    ZE_IMAGE_FORMAT_LAYOUT_11_11_10 = 10
    ZE_IMAGE_FORMAT_LAYOUT_5_6_5 = 11
    ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1 = 12
    ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4 = 13
    ZE_IMAGE_FORMAT_LAYOUT_Y8 = 14
    ZE_IMAGE_FORMAT_LAYOUT_NV12 = 15
    ZE_IMAGE_FORMAT_LAYOUT_YUYV = 16
    ZE_IMAGE_FORMAT_LAYOUT_VYUY = 17
    ZE_IMAGE_FORMAT_LAYOUT_YVYU = 18
    ZE_IMAGE_FORMAT_LAYOUT_UYVY = 19
    ZE_IMAGE_FORMAT_LAYOUT_AYUV = 20
    ZE_IMAGE_FORMAT_LAYOUT_P010 = 21
    ZE_IMAGE_FORMAT_LAYOUT_Y410 = 22
    ZE_IMAGE_FORMAT_LAYOUT_P012 = 23
    ZE_IMAGE_FORMAT_LAYOUT_Y16 = 24
    ZE_IMAGE_FORMAT_LAYOUT_P016 = 25
    ZE_IMAGE_FORMAT_LAYOUT_Y216 = 26
    ZE_IMAGE_FORMAT_LAYOUT_P216 = 27
    ZE_IMAGE_FORMAT_LAYOUT_P8 = 28
    ZE_IMAGE_FORMAT_LAYOUT_YUY2 = 29
    ZE_IMAGE_FORMAT_LAYOUT_A8P8 = 30
    ZE_IMAGE_FORMAT_LAYOUT_IA44 = 31
    ZE_IMAGE_FORMAT_LAYOUT_AI44 = 32
    ZE_IMAGE_FORMAT_LAYOUT_Y416 = 33
    ZE_IMAGE_FORMAT_LAYOUT_Y210 = 34
    ZE_IMAGE_FORMAT_LAYOUT_I420 = 35
    ZE_IMAGE_FORMAT_LAYOUT_YV12 = 36
    ZE_IMAGE_FORMAT_LAYOUT_400P = 37
    ZE_IMAGE_FORMAT_LAYOUT_422H = 38
    ZE_IMAGE_FORMAT_LAYOUT_422V = 39
    ZE_IMAGE_FORMAT_LAYOUT_444P = 40
    ZE_IMAGE_FORMAT_LAYOUT_RGBP = 41
    ZE_IMAGE_FORMAT_LAYOUT_BRGP = 42
    ZE_IMAGE_FORMAT_LAYOUT_8_8_8 = 43
    ZE_IMAGE_FORMAT_LAYOUT_16_16_16 = 44
    ZE_IMAGE_FORMAT_LAYOUT_32_32_32 = 45
    ZE_IMAGE_FORMAT_LAYOUT_FORCE_UINT32 = 2147483647
end

const ze_image_format_layout_t = _ze_image_format_layout_t

@cenum _ze_image_format_type_t::UInt32 begin
    ZE_IMAGE_FORMAT_TYPE_UINT = 0
    ZE_IMAGE_FORMAT_TYPE_SINT = 1
    ZE_IMAGE_FORMAT_TYPE_UNORM = 2
    ZE_IMAGE_FORMAT_TYPE_SNORM = 3
    ZE_IMAGE_FORMAT_TYPE_FLOAT = 4
    ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32 = 2147483647
end

const ze_image_format_type_t = _ze_image_format_type_t

@cenum _ze_image_format_swizzle_t::UInt32 begin
    ZE_IMAGE_FORMAT_SWIZZLE_R = 0
    ZE_IMAGE_FORMAT_SWIZZLE_G = 1
    ZE_IMAGE_FORMAT_SWIZZLE_B = 2
    ZE_IMAGE_FORMAT_SWIZZLE_A = 3
    ZE_IMAGE_FORMAT_SWIZZLE_0 = 4
    ZE_IMAGE_FORMAT_SWIZZLE_1 = 5
    ZE_IMAGE_FORMAT_SWIZZLE_X = 6
    ZE_IMAGE_FORMAT_SWIZZLE_FORCE_UINT32 = 2147483647
end

const ze_image_format_swizzle_t = _ze_image_format_swizzle_t

struct _ze_image_format_t
    layout::ze_image_format_layout_t
    type::ze_image_format_type_t
    x::ze_image_format_swizzle_t
    y::ze_image_format_swizzle_t
    z::ze_image_format_swizzle_t
    w::ze_image_format_swizzle_t
end

const ze_image_format_t = _ze_image_format_t

const ze_image_flags_t = UInt32

@cenum _ze_image_type_t::UInt32 begin
    ZE_IMAGE_TYPE_1D = 0
    ZE_IMAGE_TYPE_1DARRAY = 1
    ZE_IMAGE_TYPE_2D = 2
    ZE_IMAGE_TYPE_2DARRAY = 3
    ZE_IMAGE_TYPE_3D = 4
    ZE_IMAGE_TYPE_BUFFER = 5
    ZE_IMAGE_TYPE_FORCE_UINT32 = 2147483647
end

const ze_image_type_t = _ze_image_type_t

struct _ze_image_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_image_flags_t
    type::ze_image_type_t
    format::ze_image_format_t
    width::UInt64
    height::UInt32
    depth::UInt32
    arraylevels::UInt32
    miplevels::UInt32
end

const ze_image_desc_t = _ze_image_desc_t

const ze_image_sampler_filter_flags_t = UInt32

struct _ze_image_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    samplerFilterFlags::ze_image_sampler_filter_flags_t
end

const ze_image_properties_t = _ze_image_properties_t

const ze_device_mem_alloc_flags_t = UInt32

struct _ze_device_mem_alloc_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_device_mem_alloc_flags_t
    ordinal::UInt32
end

const ze_device_mem_alloc_desc_t = _ze_device_mem_alloc_desc_t

const ze_host_mem_alloc_flags_t = UInt32

struct _ze_host_mem_alloc_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_host_mem_alloc_flags_t
end

const ze_host_mem_alloc_desc_t = _ze_host_mem_alloc_desc_t

@cenum _ze_memory_type_t::UInt32 begin
    ZE_MEMORY_TYPE_UNKNOWN = 0
    ZE_MEMORY_TYPE_HOST = 1
    ZE_MEMORY_TYPE_DEVICE = 2
    ZE_MEMORY_TYPE_SHARED = 3
    ZE_MEMORY_TYPE_FORCE_UINT32 = 2147483647
end

const ze_memory_type_t = _ze_memory_type_t

struct _ze_memory_allocation_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    type::ze_memory_type_t
    id::UInt64
    pageSize::UInt64
end

const ze_memory_allocation_properties_t = _ze_memory_allocation_properties_t

struct _ze_external_memory_export_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_external_memory_type_flags_t
end

const ze_external_memory_export_desc_t = _ze_external_memory_export_desc_t

struct _ze_external_memory_import_fd_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_external_memory_type_flags_t
    fd::Cint
end

const ze_external_memory_import_fd_t = _ze_external_memory_import_fd_t

struct _ze_external_memory_export_fd_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_external_memory_type_flags_t
    fd::Cint
end

const ze_external_memory_export_fd_t = _ze_external_memory_export_fd_t

struct _ze_external_memory_import_win32_handle_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_external_memory_type_flags_t
    handle::Ptr{Cvoid}
    name::Ptr{Cvoid}
end

const ze_external_memory_import_win32_handle_t = _ze_external_memory_import_win32_handle_t

struct _ze_external_memory_export_win32_handle_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_external_memory_type_flags_t
    handle::Ptr{Cvoid}
end

const ze_external_memory_export_win32_handle_t = _ze_external_memory_export_win32_handle_t

struct _ze_module_constants_t
    numConstants::UInt32
    pConstantIds::Ptr{UInt32}
    pConstantValues::Ptr{Ptr{Cvoid}}
end

const ze_module_constants_t = _ze_module_constants_t

@cenum _ze_module_format_t::UInt32 begin
    ZE_MODULE_FORMAT_IL_SPIRV = 0
    ZE_MODULE_FORMAT_NATIVE = 1
    ZE_MODULE_FORMAT_FORCE_UINT32 = 2147483647
end

const ze_module_format_t = _ze_module_format_t

struct _ze_module_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    format::ze_module_format_t
    inputSize::Csize_t
    pInputModule::Ptr{UInt8}
    pBuildFlags::Ptr{Cchar}
    pConstants::Ptr{ze_module_constants_t}
end

const ze_module_desc_t = _ze_module_desc_t

const ze_module_property_flags_t = UInt32

struct _ze_module_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_module_property_flags_t
end

const ze_module_properties_t = _ze_module_properties_t

const ze_kernel_flags_t = UInt32

struct _ze_kernel_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_kernel_flags_t
    pKernelName::Ptr{Cchar}
end

const ze_kernel_desc_t = _ze_kernel_desc_t

struct _ze_kernel_uuid_t
    kid::NTuple{16,UInt8}
    mid::NTuple{16,UInt8}
end

const ze_kernel_uuid_t = _ze_kernel_uuid_t

struct _ze_kernel_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    numKernelArgs::UInt32
    requiredGroupSizeX::UInt32
    requiredGroupSizeY::UInt32
    requiredGroupSizeZ::UInt32
    requiredNumSubGroups::UInt32
    requiredSubgroupSize::UInt32
    maxSubgroupSize::UInt32
    maxNumSubgroups::UInt32
    localMemSize::UInt32
    privateMemSize::UInt32
    spillMemSize::UInt32
    uuid::ze_kernel_uuid_t
end

const ze_kernel_properties_t = _ze_kernel_properties_t

struct _ze_kernel_preferred_group_size_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    preferredMultiple::UInt32
end

const ze_kernel_preferred_group_size_properties_t = _ze_kernel_preferred_group_size_properties_t

struct _ze_group_count_t
    groupCountX::UInt32
    groupCountY::UInt32
    groupCountZ::UInt32
end

const ze_group_count_t = _ze_group_count_t

struct _ze_module_program_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    count::UInt32
    inputSizes::Ptr{Csize_t}
    pInputModules::Ptr{Ptr{UInt8}}
    pBuildFlags::Ptr{Ptr{Cchar}}
    pConstants::Ptr{Ptr{ze_module_constants_t}}
end

const ze_module_program_exp_desc_t = _ze_module_program_exp_desc_t

const ze_device_raytracing_ext_flags_t = UInt32

struct _ze_device_raytracing_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_device_raytracing_ext_flags_t
    maxBVHLevels::UInt32
end

const ze_device_raytracing_ext_properties_t = _ze_device_raytracing_ext_properties_t

const ze_raytracing_mem_alloc_ext_flags_t = UInt32

struct _ze_raytracing_mem_alloc_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_raytracing_mem_alloc_ext_flags_t
end

const ze_raytracing_mem_alloc_ext_desc_t = _ze_raytracing_mem_alloc_ext_desc_t

@cenum _ze_sampler_address_mode_t::UInt32 begin
    ZE_SAMPLER_ADDRESS_MODE_NONE = 0
    ZE_SAMPLER_ADDRESS_MODE_REPEAT = 1
    ZE_SAMPLER_ADDRESS_MODE_CLAMP = 2
    ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3
    ZE_SAMPLER_ADDRESS_MODE_MIRROR = 4
    ZE_SAMPLER_ADDRESS_MODE_FORCE_UINT32 = 2147483647
end

const ze_sampler_address_mode_t = _ze_sampler_address_mode_t

@cenum _ze_sampler_filter_mode_t::UInt32 begin
    ZE_SAMPLER_FILTER_MODE_NEAREST = 0
    ZE_SAMPLER_FILTER_MODE_LINEAR = 1
    ZE_SAMPLER_FILTER_MODE_FORCE_UINT32 = 2147483647
end

const ze_sampler_filter_mode_t = _ze_sampler_filter_mode_t

struct _ze_sampler_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    addressMode::ze_sampler_address_mode_t
    filterMode::ze_sampler_filter_mode_t
    isNormalized::ze_bool_t
end

const ze_sampler_desc_t = _ze_sampler_desc_t

const ze_physical_mem_flags_t = UInt32

struct _ze_physical_mem_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_physical_mem_flags_t
    size::Csize_t
end

const ze_physical_mem_desc_t = _ze_physical_mem_desc_t

const ze_device_fp_atomic_ext_flags_t = UInt32

struct _ze_float_atomic_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    fp16Flags::ze_device_fp_atomic_ext_flags_t
    fp32Flags::ze_device_fp_atomic_ext_flags_t
    fp64Flags::ze_device_fp_atomic_ext_flags_t
end

const ze_float_atomic_ext_properties_t = _ze_float_atomic_ext_properties_t

const ze_relaxed_allocation_limits_exp_flags_t = UInt32

struct _ze_relaxed_allocation_limits_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_relaxed_allocation_limits_exp_flags_t
end

const ze_relaxed_allocation_limits_exp_desc_t = _ze_relaxed_allocation_limits_exp_desc_t

const ze_driver_ddi_handle_ext_flags_t = UInt32

struct _ze_driver_ddi_handles_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_driver_ddi_handle_ext_flags_t
end

const ze_driver_ddi_handles_ext_properties_t = _ze_driver_ddi_handles_ext_properties_t

const ze_external_semaphore_ext_flags_t = UInt32

struct _ze_external_semaphore_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_external_semaphore_ext_flags_t
end

const ze_external_semaphore_ext_desc_t = _ze_external_semaphore_ext_desc_t

struct _ze_external_semaphore_win32_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    handle::Ptr{Cvoid}
    name::Ptr{Cchar}
end

const ze_external_semaphore_win32_ext_desc_t = _ze_external_semaphore_win32_ext_desc_t

struct _ze_external_semaphore_fd_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    fd::Cint
end

const ze_external_semaphore_fd_ext_desc_t = _ze_external_semaphore_fd_ext_desc_t

struct _ze_external_semaphore_signal_params_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    value::UInt64
end

const ze_external_semaphore_signal_params_ext_t = _ze_external_semaphore_signal_params_ext_t

struct _ze_external_semaphore_wait_params_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    value::UInt64
end

const ze_external_semaphore_wait_params_ext_t = _ze_external_semaphore_wait_params_ext_t

struct _ze_device_cache_line_size_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    cacheLineSize::Csize_t
end

const ze_device_cache_line_size_ext_t = _ze_device_cache_line_size_ext_t

@cenum _ze_rtas_builder_ext_version_t::UInt32 begin
    ZE_RTAS_BUILDER_EXT_VERSION_1_0 = 65536
    ZE_RTAS_BUILDER_EXT_VERSION_CURRENT = 65536
    ZE_RTAS_BUILDER_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_ext_version_t = _ze_rtas_builder_ext_version_t

struct _ze_rtas_builder_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    builderVersion::ze_rtas_builder_ext_version_t
end

const ze_rtas_builder_ext_desc_t = _ze_rtas_builder_ext_desc_t

const ze_rtas_builder_ext_flags_t = UInt32

struct _ze_rtas_builder_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_rtas_builder_ext_flags_t
    rtasBufferSizeBytesExpected::Csize_t
    rtasBufferSizeBytesMaxRequired::Csize_t
    scratchBufferSizeBytes::Csize_t
end

const ze_rtas_builder_ext_properties_t = _ze_rtas_builder_ext_properties_t

const ze_rtas_parallel_operation_ext_flags_t = UInt32

struct _ze_rtas_parallel_operation_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_rtas_parallel_operation_ext_flags_t
    maxConcurrency::UInt32
end

const ze_rtas_parallel_operation_ext_properties_t = _ze_rtas_parallel_operation_ext_properties_t

const ze_rtas_device_ext_flags_t = UInt32

@cenum _ze_rtas_format_ext_t::UInt32 begin
    ZE_RTAS_FORMAT_EXT_INVALID = 0
    ZE_RTAS_FORMAT_EXT_MAX = 2147483646
    ZE_RTAS_FORMAT_EXT_FORCE_UINT32 = 2147483647
end

const ze_rtas_format_ext_t = _ze_rtas_format_ext_t

struct _ze_rtas_device_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_rtas_device_ext_flags_t
    rtasFormat::ze_rtas_format_ext_t
    rtasBufferAlignment::UInt32
end

const ze_rtas_device_ext_properties_t = _ze_rtas_device_ext_properties_t

struct _ze_rtas_float3_ext_t
    x::Cfloat
    y::Cfloat
    z::Cfloat
end

const ze_rtas_float3_ext_t = _ze_rtas_float3_ext_t

struct _ze_rtas_transform_float3x4_column_major_ext_t
    vx_x::Cfloat
    vx_y::Cfloat
    vx_z::Cfloat
    vy_x::Cfloat
    vy_y::Cfloat
    vy_z::Cfloat
    vz_x::Cfloat
    vz_y::Cfloat
    vz_z::Cfloat
    p_x::Cfloat
    p_y::Cfloat
    p_z::Cfloat
end

const ze_rtas_transform_float3x4_column_major_ext_t = _ze_rtas_transform_float3x4_column_major_ext_t

struct _ze_rtas_transform_float3x4_aligned_column_major_ext_t
    vx_x::Cfloat
    vx_y::Cfloat
    vx_z::Cfloat
    pad0::Cfloat
    vy_x::Cfloat
    vy_y::Cfloat
    vy_z::Cfloat
    pad1::Cfloat
    vz_x::Cfloat
    vz_y::Cfloat
    vz_z::Cfloat
    pad2::Cfloat
    p_x::Cfloat
    p_y::Cfloat
    p_z::Cfloat
    pad3::Cfloat
end

const ze_rtas_transform_float3x4_aligned_column_major_ext_t = _ze_rtas_transform_float3x4_aligned_column_major_ext_t

struct _ze_rtas_transform_float3x4_row_major_ext_t
    vx_x::Cfloat
    vy_x::Cfloat
    vz_x::Cfloat
    p_x::Cfloat
    vx_y::Cfloat
    vy_y::Cfloat
    vz_y::Cfloat
    p_y::Cfloat
    vx_z::Cfloat
    vy_z::Cfloat
    vz_z::Cfloat
    p_z::Cfloat
end

const ze_rtas_transform_float3x4_row_major_ext_t = _ze_rtas_transform_float3x4_row_major_ext_t

struct _ze_rtas_aabb_ext_t
    lower::ze_rtas_float3_ext_t
    upper::ze_rtas_float3_ext_t
end

const ze_rtas_aabb_ext_t = _ze_rtas_aabb_ext_t

struct _ze_rtas_triangle_indices_uint32_ext_t
    v0::UInt32
    v1::UInt32
    v2::UInt32
end

const ze_rtas_triangle_indices_uint32_ext_t = _ze_rtas_triangle_indices_uint32_ext_t

struct _ze_rtas_quad_indices_uint32_ext_t
    v0::UInt32
    v1::UInt32
    v2::UInt32
    v3::UInt32
end

const ze_rtas_quad_indices_uint32_ext_t = _ze_rtas_quad_indices_uint32_ext_t

const ze_rtas_builder_packed_geometry_type_ext_t = UInt8

struct _ze_rtas_builder_geometry_info_ext_t
    geometryType::ze_rtas_builder_packed_geometry_type_ext_t
end

const ze_rtas_builder_geometry_info_ext_t = _ze_rtas_builder_geometry_info_ext_t

const ze_rtas_builder_packed_geometry_ext_flags_t = UInt8

const ze_rtas_builder_packed_input_data_format_ext_t = UInt8

struct _ze_rtas_builder_triangles_geometry_info_ext_t
    geometryType::ze_rtas_builder_packed_geometry_type_ext_t
    geometryFlags::ze_rtas_builder_packed_geometry_ext_flags_t
    geometryMask::UInt8
    triangleFormat::ze_rtas_builder_packed_input_data_format_ext_t
    vertexFormat::ze_rtas_builder_packed_input_data_format_ext_t
    triangleCount::UInt32
    vertexCount::UInt32
    triangleStride::UInt32
    vertexStride::UInt32
    pTriangleBuffer::Ptr{Cvoid}
    pVertexBuffer::Ptr{Cvoid}
end

const ze_rtas_builder_triangles_geometry_info_ext_t = _ze_rtas_builder_triangles_geometry_info_ext_t

struct _ze_rtas_builder_quads_geometry_info_ext_t
    geometryType::ze_rtas_builder_packed_geometry_type_ext_t
    geometryFlags::ze_rtas_builder_packed_geometry_ext_flags_t
    geometryMask::UInt8
    quadFormat::ze_rtas_builder_packed_input_data_format_ext_t
    vertexFormat::ze_rtas_builder_packed_input_data_format_ext_t
    quadCount::UInt32
    vertexCount::UInt32
    quadStride::UInt32
    vertexStride::UInt32
    pQuadBuffer::Ptr{Cvoid}
    pVertexBuffer::Ptr{Cvoid}
end

const ze_rtas_builder_quads_geometry_info_ext_t = _ze_rtas_builder_quads_geometry_info_ext_t

struct _ze_rtas_geometry_aabbs_ext_cb_params_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    primID::UInt32
    primIDCount::UInt32
    pGeomUserPtr::Ptr{Cvoid}
    pBuildUserPtr::Ptr{Cvoid}
    pBoundsOut::Ptr{ze_rtas_aabb_ext_t}
end

const ze_rtas_geometry_aabbs_ext_cb_params_t = _ze_rtas_geometry_aabbs_ext_cb_params_t

# typedef void ( * ze_rtas_geometry_aabbs_cb_ext_t ) ( ze_rtas_geometry_aabbs_ext_cb_params_t * params ///< [in] callback function parameters structure )
const ze_rtas_geometry_aabbs_cb_ext_t = Ptr{Cvoid}

struct _ze_rtas_builder_procedural_geometry_info_ext_t
    geometryType::ze_rtas_builder_packed_geometry_type_ext_t
    geometryFlags::ze_rtas_builder_packed_geometry_ext_flags_t
    geometryMask::UInt8
    reserved::UInt8
    primCount::UInt32
    pfnGetBoundsCb::ze_rtas_geometry_aabbs_cb_ext_t
    pGeomUserPtr::Ptr{Cvoid}
end

const ze_rtas_builder_procedural_geometry_info_ext_t = _ze_rtas_builder_procedural_geometry_info_ext_t

const ze_rtas_builder_packed_instance_ext_flags_t = UInt8

struct _ze_rtas_builder_instance_geometry_info_ext_t
    geometryType::ze_rtas_builder_packed_geometry_type_ext_t
    instanceFlags::ze_rtas_builder_packed_instance_ext_flags_t
    geometryMask::UInt8
    transformFormat::ze_rtas_builder_packed_input_data_format_ext_t
    instanceUserID::UInt32
    pTransform::Ptr{Cvoid}
    pBounds::Ptr{ze_rtas_aabb_ext_t}
    pAccelerationStructure::Ptr{Cvoid}
end

const ze_rtas_builder_instance_geometry_info_ext_t = _ze_rtas_builder_instance_geometry_info_ext_t

@cenum _ze_rtas_builder_build_quality_hint_ext_t::UInt32 begin
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXT_LOW = 0
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXT_MEDIUM = 1
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXT_HIGH = 2
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXT_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_build_quality_hint_ext_t = _ze_rtas_builder_build_quality_hint_ext_t

const ze_rtas_builder_build_op_ext_flags_t = UInt32

struct _ze_rtas_builder_build_op_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    rtasFormat::ze_rtas_format_ext_t
    buildQuality::ze_rtas_builder_build_quality_hint_ext_t
    buildFlags::ze_rtas_builder_build_op_ext_flags_t
    ppGeometries::Ptr{Ptr{ze_rtas_builder_geometry_info_ext_t}}
    numGeometries::UInt32
end

const ze_rtas_builder_build_op_ext_desc_t = _ze_rtas_builder_build_op_ext_desc_t

struct _ze_device_vector_width_properties_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    vector_width_size::UInt32
    preferred_vector_width_char::UInt32
    preferred_vector_width_short::UInt32
    preferred_vector_width_int::UInt32
    preferred_vector_width_long::UInt32
    preferred_vector_width_float::UInt32
    preferred_vector_width_double::UInt32
    preferred_vector_width_half::UInt32
    native_vector_width_char::UInt32
    native_vector_width_short::UInt32
    native_vector_width_int::UInt32
    native_vector_width_long::UInt32
    native_vector_width_float::UInt32
    native_vector_width_double::UInt32
    native_vector_width_half::UInt32
end

const ze_device_vector_width_properties_ext_t = _ze_device_vector_width_properties_ext_t

struct _ze_cache_reservation_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    maxCacheReservationSize::Csize_t
end

const ze_cache_reservation_ext_desc_t = _ze_cache_reservation_ext_desc_t

struct _ze_image_memory_properties_exp_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    size::UInt64
    rowPitch::UInt64
    slicePitch::UInt64
end

const ze_image_memory_properties_exp_t = _ze_image_memory_properties_exp_t

struct _ze_image_view_planar_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    planeIndex::UInt32
end

const ze_image_view_planar_ext_desc_t = _ze_image_view_planar_ext_desc_t

struct _ze_image_view_planar_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    planeIndex::UInt32
end

const ze_image_view_planar_exp_desc_t = _ze_image_view_planar_exp_desc_t

const ze_scheduling_hint_exp_flags_t = UInt32

struct _ze_scheduling_hint_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    schedulingHintFlags::ze_scheduling_hint_exp_flags_t
end

const ze_scheduling_hint_exp_properties_t = _ze_scheduling_hint_exp_properties_t

struct _ze_scheduling_hint_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_scheduling_hint_exp_flags_t
end

const ze_scheduling_hint_exp_desc_t = _ze_scheduling_hint_exp_desc_t

struct _ze_context_power_saving_hint_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    hint::UInt32
end

const ze_context_power_saving_hint_exp_desc_t = _ze_context_power_saving_hint_exp_desc_t

struct _ze_eu_count_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    numTotalEUs::UInt32
end

const ze_eu_count_ext_t = _ze_eu_count_ext_t

struct _ze_pci_address_ext_t
    domain::UInt32
    bus::UInt32
    device::UInt32
    _function::UInt32
end

const ze_pci_address_ext_t = _ze_pci_address_ext_t

struct _ze_pci_speed_ext_t
    genVersion::Int32
    width::Int32
    maxBandwidth::Int64
end

const ze_pci_speed_ext_t = _ze_pci_speed_ext_t

struct _ze_pci_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    address::ze_pci_address_ext_t
    maxSpeed::ze_pci_speed_ext_t
end

const ze_pci_ext_properties_t = _ze_pci_ext_properties_t

struct _ze_srgb_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    sRGB::ze_bool_t
end

const ze_srgb_ext_desc_t = _ze_srgb_ext_desc_t

struct _ze_image_allocation_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    id::UInt64
end

const ze_image_allocation_ext_properties_t = _ze_image_allocation_ext_properties_t

const ze_linkage_inspection_ext_flags_t = UInt32

struct _ze_linkage_inspection_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_linkage_inspection_ext_flags_t
end

const ze_linkage_inspection_ext_desc_t = _ze_linkage_inspection_ext_desc_t

const ze_memory_compression_hints_ext_flags_t = UInt32

struct _ze_memory_compression_hints_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_memory_compression_hints_ext_flags_t
end

const ze_memory_compression_hints_ext_desc_t = _ze_memory_compression_hints_ext_desc_t

const ze_driver_memory_free_policy_ext_flags_t = UInt32

struct _ze_driver_memory_free_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    freePolicies::ze_driver_memory_free_policy_ext_flags_t
end

const ze_driver_memory_free_ext_properties_t = _ze_driver_memory_free_ext_properties_t

struct _ze_memory_free_ext_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    freePolicy::ze_driver_memory_free_policy_ext_flags_t
end

const ze_memory_free_ext_desc_t = _ze_memory_free_ext_desc_t

struct _ze_device_p2p_bandwidth_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    logicalBandwidth::UInt32
    physicalBandwidth::UInt32
    bandwidthUnit::ze_bandwidth_unit_t
    logicalLatency::UInt32
    physicalLatency::UInt32
    latencyUnit::ze_latency_unit_t
end

const ze_device_p2p_bandwidth_exp_properties_t = _ze_device_p2p_bandwidth_exp_properties_t

struct _ze_copy_bandwidth_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    copyBandwidth::UInt32
    copyBandwidthUnit::ze_bandwidth_unit_t
end

const ze_copy_bandwidth_exp_properties_t = _ze_copy_bandwidth_exp_properties_t

struct _ze_device_luid_ext_t
    id::NTuple{8,UInt8}
end

const ze_device_luid_ext_t = _ze_device_luid_ext_t

struct _ze_device_luid_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    luid::ze_device_luid_ext_t
    nodeMask::UInt32
end

const ze_device_luid_ext_properties_t = _ze_device_luid_ext_properties_t

struct _ze_fabric_vertex_pci_exp_address_t
    domain::UInt32
    bus::UInt32
    device::UInt32
    _function::UInt32
end

const ze_fabric_vertex_pci_exp_address_t = _ze_fabric_vertex_pci_exp_address_t

@cenum _ze_fabric_vertex_exp_type_t::UInt32 begin
    ZE_FABRIC_VERTEX_EXP_TYPE_UNKNOWN = 0
    ZE_FABRIC_VERTEX_EXP_TYPE_DEVICE = 1
    ZE_FABRIC_VERTEX_EXP_TYPE_SUBDEVICE = 2
    ZE_FABRIC_VERTEX_EXP_TYPE_SWITCH = 3
    ZE_FABRIC_VERTEX_EXP_TYPE_FORCE_UINT32 = 2147483647
end

const ze_fabric_vertex_exp_type_t = _ze_fabric_vertex_exp_type_t

struct _ze_fabric_vertex_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    uuid::ze_uuid_t
    type::ze_fabric_vertex_exp_type_t
    remote::ze_bool_t
    address::ze_fabric_vertex_pci_exp_address_t
end

const ze_fabric_vertex_exp_properties_t = _ze_fabric_vertex_exp_properties_t

@cenum _ze_fabric_edge_exp_duplexity_t::UInt32 begin
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_UNKNOWN = 0
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_HALF_DUPLEX = 1
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_FULL_DUPLEX = 2
    ZE_FABRIC_EDGE_EXP_DUPLEXITY_FORCE_UINT32 = 2147483647
end

const ze_fabric_edge_exp_duplexity_t = _ze_fabric_edge_exp_duplexity_t

struct _ze_fabric_edge_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    uuid::ze_uuid_t
    model::NTuple{256,Cchar}
    bandwidth::UInt32
    bandwidthUnit::ze_bandwidth_unit_t
    latency::UInt32
    latencyUnit::ze_latency_unit_t
    duplexity::ze_fabric_edge_exp_duplexity_t
end

const ze_fabric_edge_exp_properties_t = _ze_fabric_edge_exp_properties_t

@cenum _ze_device_memory_ext_type_t::UInt32 begin
    ZE_DEVICE_MEMORY_EXT_TYPE_HBM = 0
    ZE_DEVICE_MEMORY_EXT_TYPE_HBM2 = 1
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR = 2
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR2 = 3
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR3 = 4
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR4 = 5
    ZE_DEVICE_MEMORY_EXT_TYPE_DDR5 = 6
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR = 7
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR3 = 8
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR4 = 9
    ZE_DEVICE_MEMORY_EXT_TYPE_LPDDR5 = 10
    ZE_DEVICE_MEMORY_EXT_TYPE_SRAM = 11
    ZE_DEVICE_MEMORY_EXT_TYPE_L1 = 12
    ZE_DEVICE_MEMORY_EXT_TYPE_L3 = 13
    ZE_DEVICE_MEMORY_EXT_TYPE_GRF = 14
    ZE_DEVICE_MEMORY_EXT_TYPE_SLM = 15
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR4 = 16
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5 = 17
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR5X = 18
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6 = 19
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR6X = 20
    ZE_DEVICE_MEMORY_EXT_TYPE_GDDR7 = 21
    ZE_DEVICE_MEMORY_EXT_TYPE_FORCE_UINT32 = 2147483647
end

const ze_device_memory_ext_type_t = _ze_device_memory_ext_type_t

struct _ze_device_memory_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    type::ze_device_memory_ext_type_t
    physicalSize::UInt64
    readBandwidth::UInt32
    writeBandwidth::UInt32
    bandwidthUnit::ze_bandwidth_unit_t
end

const ze_device_memory_ext_properties_t = _ze_device_memory_ext_properties_t

struct _ze_device_ip_version_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    ipVersion::UInt32
end

const ze_device_ip_version_ext_t = _ze_device_ip_version_ext_t

struct _ze_kernel_max_group_size_properties_ext_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    maxGroupSize::UInt32
end

const ze_kernel_max_group_size_properties_ext_t = _ze_kernel_max_group_size_properties_ext_t

struct _ze_sub_allocation_t
    base::Ptr{Cvoid}
    size::Csize_t
end

const ze_sub_allocation_t = _ze_sub_allocation_t

struct _ze_memory_sub_allocations_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    pCount::Ptr{UInt32}
    pSubAllocations::Ptr{ze_sub_allocation_t}
end

const ze_memory_sub_allocations_exp_properties_t = _ze_memory_sub_allocations_exp_properties_t

const ze_event_query_kernel_timestamps_ext_flags_t = UInt32

struct _ze_event_query_kernel_timestamps_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_event_query_kernel_timestamps_ext_flags_t
end

const ze_event_query_kernel_timestamps_ext_properties_t = _ze_event_query_kernel_timestamps_ext_properties_t

struct _ze_synchronized_timestamp_data_ext_t
    kernelStart::UInt64
    kernelEnd::UInt64
end

const ze_synchronized_timestamp_data_ext_t = _ze_synchronized_timestamp_data_ext_t

struct _ze_synchronized_timestamp_result_ext_t
    _global::ze_synchronized_timestamp_data_ext_t
    context::ze_synchronized_timestamp_data_ext_t
end

const ze_synchronized_timestamp_result_ext_t = _ze_synchronized_timestamp_result_ext_t

struct _ze_event_query_kernel_timestamps_results_ext_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    pKernelTimestampsBuffer::Ptr{ze_kernel_timestamp_result_t}
    pSynchronizedTimestampsBuffer::Ptr{ze_synchronized_timestamp_result_ext_t}
end

const ze_event_query_kernel_timestamps_results_ext_properties_t = _ze_event_query_kernel_timestamps_results_ext_properties_t

@cenum _ze_rtas_builder_exp_version_t::UInt32 begin
    ZE_RTAS_BUILDER_EXP_VERSION_1_0 = 65536
    ZE_RTAS_BUILDER_EXP_VERSION_CURRENT = 65536
    ZE_RTAS_BUILDER_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_exp_version_t = _ze_rtas_builder_exp_version_t

struct _ze_rtas_builder_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    builderVersion::ze_rtas_builder_exp_version_t
end

const ze_rtas_builder_exp_desc_t = _ze_rtas_builder_exp_desc_t

const ze_rtas_builder_exp_flags_t = UInt32

struct _ze_rtas_builder_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_rtas_builder_exp_flags_t
    rtasBufferSizeBytesExpected::Csize_t
    rtasBufferSizeBytesMaxRequired::Csize_t
    scratchBufferSizeBytes::Csize_t
end

const ze_rtas_builder_exp_properties_t = _ze_rtas_builder_exp_properties_t

const ze_rtas_parallel_operation_exp_flags_t = UInt32

struct _ze_rtas_parallel_operation_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_rtas_parallel_operation_exp_flags_t
    maxConcurrency::UInt32
end

const ze_rtas_parallel_operation_exp_properties_t = _ze_rtas_parallel_operation_exp_properties_t

const ze_rtas_device_exp_flags_t = UInt32

@cenum _ze_rtas_format_exp_t::UInt32 begin
    ZE_RTAS_FORMAT_EXP_INVALID = 0
    ZE_RTAS_FORMAT_EXP_MAX = 2147483646
    ZE_RTAS_FORMAT_EXP_FORCE_UINT32 = 2147483647
end

const ze_rtas_format_exp_t = _ze_rtas_format_exp_t

struct _ze_rtas_device_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_rtas_device_exp_flags_t
    rtasFormat::ze_rtas_format_exp_t
    rtasBufferAlignment::UInt32
end

const ze_rtas_device_exp_properties_t = _ze_rtas_device_exp_properties_t

struct _ze_rtas_float3_exp_t
    x::Cfloat
    y::Cfloat
    z::Cfloat
end

const ze_rtas_float3_exp_t = _ze_rtas_float3_exp_t

struct _ze_rtas_transform_float3x4_column_major_exp_t
    vx_x::Cfloat
    vx_y::Cfloat
    vx_z::Cfloat
    vy_x::Cfloat
    vy_y::Cfloat
    vy_z::Cfloat
    vz_x::Cfloat
    vz_y::Cfloat
    vz_z::Cfloat
    p_x::Cfloat
    p_y::Cfloat
    p_z::Cfloat
end

const ze_rtas_transform_float3x4_column_major_exp_t = _ze_rtas_transform_float3x4_column_major_exp_t

struct _ze_rtas_transform_float3x4_aligned_column_major_exp_t
    vx_x::Cfloat
    vx_y::Cfloat
    vx_z::Cfloat
    pad0::Cfloat
    vy_x::Cfloat
    vy_y::Cfloat
    vy_z::Cfloat
    pad1::Cfloat
    vz_x::Cfloat
    vz_y::Cfloat
    vz_z::Cfloat
    pad2::Cfloat
    p_x::Cfloat
    p_y::Cfloat
    p_z::Cfloat
    pad3::Cfloat
end

const ze_rtas_transform_float3x4_aligned_column_major_exp_t = _ze_rtas_transform_float3x4_aligned_column_major_exp_t

struct _ze_rtas_transform_float3x4_row_major_exp_t
    vx_x::Cfloat
    vy_x::Cfloat
    vz_x::Cfloat
    p_x::Cfloat
    vx_y::Cfloat
    vy_y::Cfloat
    vz_y::Cfloat
    p_y::Cfloat
    vx_z::Cfloat
    vy_z::Cfloat
    vz_z::Cfloat
    p_z::Cfloat
end

const ze_rtas_transform_float3x4_row_major_exp_t = _ze_rtas_transform_float3x4_row_major_exp_t

struct _ze_rtas_aabb_exp_t
    lower::ze_rtas_float3_exp_t
    upper::ze_rtas_float3_exp_t
end

const ze_rtas_aabb_exp_t = _ze_rtas_aabb_exp_t

struct _ze_rtas_triangle_indices_uint32_exp_t
    v0::UInt32
    v1::UInt32
    v2::UInt32
end

const ze_rtas_triangle_indices_uint32_exp_t = _ze_rtas_triangle_indices_uint32_exp_t

struct _ze_rtas_quad_indices_uint32_exp_t
    v0::UInt32
    v1::UInt32
    v2::UInt32
    v3::UInt32
end

const ze_rtas_quad_indices_uint32_exp_t = _ze_rtas_quad_indices_uint32_exp_t

const ze_rtas_builder_packed_geometry_type_exp_t = UInt8

struct _ze_rtas_builder_geometry_info_exp_t
    geometryType::ze_rtas_builder_packed_geometry_type_exp_t
end

const ze_rtas_builder_geometry_info_exp_t = _ze_rtas_builder_geometry_info_exp_t

const ze_rtas_builder_packed_geometry_exp_flags_t = UInt8

const ze_rtas_builder_packed_input_data_format_exp_t = UInt8

struct _ze_rtas_builder_triangles_geometry_info_exp_t
    geometryType::ze_rtas_builder_packed_geometry_type_exp_t
    geometryFlags::ze_rtas_builder_packed_geometry_exp_flags_t
    geometryMask::UInt8
    triangleFormat::ze_rtas_builder_packed_input_data_format_exp_t
    vertexFormat::ze_rtas_builder_packed_input_data_format_exp_t
    triangleCount::UInt32
    vertexCount::UInt32
    triangleStride::UInt32
    vertexStride::UInt32
    pTriangleBuffer::Ptr{Cvoid}
    pVertexBuffer::Ptr{Cvoid}
end

const ze_rtas_builder_triangles_geometry_info_exp_t = _ze_rtas_builder_triangles_geometry_info_exp_t

struct _ze_rtas_builder_quads_geometry_info_exp_t
    geometryType::ze_rtas_builder_packed_geometry_type_exp_t
    geometryFlags::ze_rtas_builder_packed_geometry_exp_flags_t
    geometryMask::UInt8
    quadFormat::ze_rtas_builder_packed_input_data_format_exp_t
    vertexFormat::ze_rtas_builder_packed_input_data_format_exp_t
    quadCount::UInt32
    vertexCount::UInt32
    quadStride::UInt32
    vertexStride::UInt32
    pQuadBuffer::Ptr{Cvoid}
    pVertexBuffer::Ptr{Cvoid}
end

const ze_rtas_builder_quads_geometry_info_exp_t = _ze_rtas_builder_quads_geometry_info_exp_t

struct _ze_rtas_geometry_aabbs_exp_cb_params_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    primID::UInt32
    primIDCount::UInt32
    pGeomUserPtr::Ptr{Cvoid}
    pBuildUserPtr::Ptr{Cvoid}
    pBoundsOut::Ptr{ze_rtas_aabb_exp_t}
end

const ze_rtas_geometry_aabbs_exp_cb_params_t = _ze_rtas_geometry_aabbs_exp_cb_params_t

# typedef void ( * ze_rtas_geometry_aabbs_cb_exp_t ) ( ze_rtas_geometry_aabbs_exp_cb_params_t * params ///< [in] callback function parameters structure )
const ze_rtas_geometry_aabbs_cb_exp_t = Ptr{Cvoid}

struct _ze_rtas_builder_procedural_geometry_info_exp_t
    geometryType::ze_rtas_builder_packed_geometry_type_exp_t
    geometryFlags::ze_rtas_builder_packed_geometry_exp_flags_t
    geometryMask::UInt8
    reserved::UInt8
    primCount::UInt32
    pfnGetBoundsCb::ze_rtas_geometry_aabbs_cb_exp_t
    pGeomUserPtr::Ptr{Cvoid}
end

const ze_rtas_builder_procedural_geometry_info_exp_t = _ze_rtas_builder_procedural_geometry_info_exp_t

const ze_rtas_builder_packed_instance_exp_flags_t = UInt8

struct _ze_rtas_builder_instance_geometry_info_exp_t
    geometryType::ze_rtas_builder_packed_geometry_type_exp_t
    instanceFlags::ze_rtas_builder_packed_instance_exp_flags_t
    geometryMask::UInt8
    transformFormat::ze_rtas_builder_packed_input_data_format_exp_t
    instanceUserID::UInt32
    pTransform::Ptr{Cvoid}
    pBounds::Ptr{ze_rtas_aabb_exp_t}
    pAccelerationStructure::Ptr{Cvoid}
end

const ze_rtas_builder_instance_geometry_info_exp_t = _ze_rtas_builder_instance_geometry_info_exp_t

@cenum _ze_rtas_builder_build_quality_hint_exp_t::UInt32 begin
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_LOW = 0
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_MEDIUM = 1
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_HIGH = 2
    ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXP_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_build_quality_hint_exp_t = _ze_rtas_builder_build_quality_hint_exp_t

const ze_rtas_builder_build_op_exp_flags_t = UInt32

struct _ze_rtas_builder_build_op_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    rtasFormat::ze_rtas_format_exp_t
    buildQuality::ze_rtas_builder_build_quality_hint_exp_t
    buildFlags::ze_rtas_builder_build_op_exp_flags_t
    ppGeometries::Ptr{Ptr{ze_rtas_builder_geometry_info_exp_t}}
    numGeometries::UInt32
end

const ze_rtas_builder_build_op_exp_desc_t = _ze_rtas_builder_build_op_exp_desc_t

const ze_event_pool_counter_based_exp_flags_t = UInt32

struct _ze_event_pool_counter_based_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_event_pool_counter_based_exp_flags_t
end

const ze_event_pool_counter_based_exp_desc_t = _ze_event_pool_counter_based_exp_desc_t

const ze_image_bindless_exp_flags_t = UInt32

struct _ze_image_bindless_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_image_bindless_exp_flags_t
end

const ze_image_bindless_exp_desc_t = _ze_image_bindless_exp_desc_t

struct _ze_image_pitched_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    ptr::Ptr{Cvoid}
end

const ze_image_pitched_exp_desc_t = _ze_image_pitched_exp_desc_t

struct _ze_device_pitched_alloc_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    maxImageLinearWidth::Csize_t
    maxImageLinearHeight::Csize_t
end

const ze_device_pitched_alloc_exp_properties_t = _ze_device_pitched_alloc_exp_properties_t

const ze_mutable_command_exp_flags_t = UInt32

struct _ze_mutable_command_id_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_mutable_command_exp_flags_t
end

const ze_mutable_command_id_exp_desc_t = _ze_mutable_command_id_exp_desc_t

const ze_mutable_command_list_exp_flags_t = UInt32

struct _ze_mutable_command_list_exp_properties_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    mutableCommandListFlags::ze_mutable_command_list_exp_flags_t
    mutableCommandFlags::ze_mutable_command_exp_flags_t
end

const ze_mutable_command_list_exp_properties_t = _ze_mutable_command_list_exp_properties_t

struct _ze_mutable_command_list_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::ze_mutable_command_list_exp_flags_t
end

const ze_mutable_command_list_exp_desc_t = _ze_mutable_command_list_exp_desc_t

struct _ze_mutable_commands_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    flags::UInt32
end

const ze_mutable_commands_exp_desc_t = _ze_mutable_commands_exp_desc_t

struct _ze_mutable_kernel_argument_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    commandId::UInt64
    argIndex::UInt32
    argSize::Csize_t
    pArgValue::Ptr{Cvoid}
end

const ze_mutable_kernel_argument_exp_desc_t = _ze_mutable_kernel_argument_exp_desc_t

struct _ze_mutable_group_count_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    commandId::UInt64
    pGroupCount::Ptr{ze_group_count_t}
end

const ze_mutable_group_count_exp_desc_t = _ze_mutable_group_count_exp_desc_t

struct _ze_mutable_group_size_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    commandId::UInt64
    groupSizeX::UInt32
    groupSizeY::UInt32
    groupSizeZ::UInt32
end

const ze_mutable_group_size_exp_desc_t = _ze_mutable_group_size_exp_desc_t

struct _ze_mutable_global_offset_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    commandId::UInt64
    offsetX::UInt32
    offsetY::UInt32
    offsetZ::UInt32
end

const ze_mutable_global_offset_exp_desc_t = _ze_mutable_global_offset_exp_desc_t

struct _ze_mutable_graph_argument_exp_desc_t
    stype::ze_structure_type_t
    pNext::Ptr{Cvoid}
    commandId::UInt64
    argIndex::UInt32
    pArgValue::Ptr{Cvoid}
end

const ze_mutable_graph_argument_exp_desc_t = _ze_mutable_graph_argument_exp_desc_t

const ze_init_flags_t = UInt32

@cenum _ze_init_flag_t::UInt32 begin
    ZE_INIT_FLAG_GPU_ONLY = 1
    ZE_INIT_FLAG_VPU_ONLY = 2
    ZE_INIT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_init_flag_t = _ze_init_flag_t

@checked function zeInit(flags)
    @ccall libze_loader.zeInit(flags::ze_init_flags_t)::ze_result_t
end

@checked function zeDriverGet(pCount, phDrivers)
    @ccall libze_loader.zeDriverGet(pCount::Ptr{UInt32},
                                    phDrivers::Ptr{ze_driver_handle_t})::ze_result_t
end

@cenum _ze_init_driver_type_flag_t::UInt32 begin
    ZE_INIT_DRIVER_TYPE_FLAG_GPU = 1
    ZE_INIT_DRIVER_TYPE_FLAG_NPU = 2
    ZE_INIT_DRIVER_TYPE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_init_driver_type_flag_t = _ze_init_driver_type_flag_t

@checked function zeInitDrivers(pCount, phDrivers, desc)
    @ccall libze_loader.zeInitDrivers(pCount::Ptr{UInt32},
                                      phDrivers::Ptr{ze_driver_handle_t},
                                      desc::Ptr{ze_init_driver_type_desc_t})::ze_result_t
end

@cenum _ze_api_version_t::UInt32 begin
    ZE_API_VERSION_1_0 = 65536
    ZE_API_VERSION_1_1 = 65537
    ZE_API_VERSION_1_2 = 65538
    ZE_API_VERSION_1_3 = 65539
    ZE_API_VERSION_1_4 = 65540
    ZE_API_VERSION_1_5 = 65541
    ZE_API_VERSION_1_6 = 65542
    ZE_API_VERSION_1_7 = 65543
    ZE_API_VERSION_1_8 = 65544
    ZE_API_VERSION_1_9 = 65545
    ZE_API_VERSION_1_10 = 65546
    ZE_API_VERSION_1_11 = 65547
    ZE_API_VERSION_1_12 = 65548
    ZE_API_VERSION_1_13 = 65549
    ZE_API_VERSION_CURRENT = 65549
    ZE_API_VERSION_FORCE_UINT32 = 2147483647
end

const ze_api_version_t = _ze_api_version_t

@checked function zeDriverGetApiVersion(hDriver, version)
    @ccall libze_loader.zeDriverGetApiVersion(hDriver::ze_driver_handle_t,
                                              version::Ptr{ze_api_version_t})::ze_result_t
end

@checked function zeDriverGetProperties(hDriver, pDriverProperties)
    @ccall libze_loader.zeDriverGetProperties(hDriver::ze_driver_handle_t,
                                              pDriverProperties::Ptr{ze_driver_properties_t})::ze_result_t
end

@cenum _ze_ipc_property_flag_t::UInt32 begin
    ZE_IPC_PROPERTY_FLAG_MEMORY = 1
    ZE_IPC_PROPERTY_FLAG_EVENT_POOL = 2
    ZE_IPC_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_ipc_property_flag_t = _ze_ipc_property_flag_t

@checked function zeDriverGetIpcProperties(hDriver, pIpcProperties)
    @ccall libze_loader.zeDriverGetIpcProperties(hDriver::ze_driver_handle_t,
                                                 pIpcProperties::Ptr{ze_driver_ipc_properties_t})::ze_result_t
end

@checked function zeDriverGetExtensionProperties(hDriver, pCount, pExtensionProperties)
    @ccall libze_loader.zeDriverGetExtensionProperties(hDriver::ze_driver_handle_t,
                                                       pCount::Ptr{UInt32},
                                                       pExtensionProperties::Ptr{ze_driver_extension_properties_t})::ze_result_t
end

@checked function zeDriverGetExtensionFunctionAddress(hDriver, name, ppFunctionAddress)
    @ccall libze_loader.zeDriverGetExtensionFunctionAddress(hDriver::ze_driver_handle_t,
                                                            name::Ptr{Cchar},
                                                            ppFunctionAddress::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeDriverGetLastErrorDescription(hDriver, ppString)
    @ccall libze_loader.zeDriverGetLastErrorDescription(hDriver::ze_driver_handle_t,
                                                        ppString::Ptr{Ptr{Cchar}})::ze_result_t
end

@checked function zeDeviceGet(hDriver, pCount, phDevices)
    @ccall libze_loader.zeDeviceGet(hDriver::ze_driver_handle_t, pCount::Ptr{UInt32},
                                    phDevices::Ptr{ze_device_handle_t})::ze_result_t
end

@checked function zeDeviceGetRootDevice(hDevice, phRootDevice)
    @ccall libze_loader.zeDeviceGetRootDevice(hDevice::ze_device_handle_t,
                                              phRootDevice::Ptr{ze_device_handle_t})::ze_result_t
end

@checked function zeDeviceGetSubDevices(hDevice, pCount, phSubdevices)
    @ccall libze_loader.zeDeviceGetSubDevices(hDevice::ze_device_handle_t,
                                              pCount::Ptr{UInt32},
                                              phSubdevices::Ptr{ze_device_handle_t})::ze_result_t
end

@cenum _ze_device_property_flag_t::UInt32 begin
    ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1
    ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE = 2
    ZE_DEVICE_PROPERTY_FLAG_ECC = 4
    ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = 8
    ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_property_flag_t = _ze_device_property_flag_t

@checked function zeDeviceGetProperties(hDevice, pDeviceProperties)
    @ccall libze_loader.zeDeviceGetProperties(hDevice::ze_device_handle_t,
                                              pDeviceProperties::Ptr{ze_device_properties_t})::ze_result_t
end

@checked function zeDeviceGetComputeProperties(hDevice, pComputeProperties)
    @ccall libze_loader.zeDeviceGetComputeProperties(hDevice::ze_device_handle_t,
                                                     pComputeProperties::Ptr{ze_device_compute_properties_t})::ze_result_t
end

@cenum _ze_device_module_flag_t::UInt32 begin
    ZE_DEVICE_MODULE_FLAG_FP16 = 1
    ZE_DEVICE_MODULE_FLAG_FP64 = 2
    ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS = 4
    ZE_DEVICE_MODULE_FLAG_DP4A = 8
    ZE_DEVICE_MODULE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_module_flag_t = _ze_device_module_flag_t

@cenum _ze_device_fp_flag_t::UInt32 begin
    ZE_DEVICE_FP_FLAG_DENORM = 1
    ZE_DEVICE_FP_FLAG_INF_NAN = 2
    ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST = 4
    ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO = 8
    ZE_DEVICE_FP_FLAG_ROUND_TO_INF = 16
    ZE_DEVICE_FP_FLAG_FMA = 32
    ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT = 64
    ZE_DEVICE_FP_FLAG_SOFT_FLOAT = 128
    ZE_DEVICE_FP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_fp_flag_t = _ze_device_fp_flag_t

@checked function zeDeviceGetModuleProperties(hDevice, pModuleProperties)
    @ccall libze_loader.zeDeviceGetModuleProperties(hDevice::ze_device_handle_t,
                                                    pModuleProperties::Ptr{ze_device_module_properties_t})::ze_result_t
end

@cenum _ze_command_queue_group_property_flag_t::UInt32 begin
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 1
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = 2
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS = 4
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS = 8
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_command_queue_group_property_flag_t = _ze_command_queue_group_property_flag_t

@checked function zeDeviceGetCommandQueueGroupProperties(hDevice, pCount,
                                                         pCommandQueueGroupProperties)
    @ccall libze_loader.zeDeviceGetCommandQueueGroupProperties(hDevice::ze_device_handle_t,
                                                               pCount::Ptr{UInt32},
                                                               pCommandQueueGroupProperties::Ptr{ze_command_queue_group_properties_t})::ze_result_t
end

@cenum _ze_device_memory_property_flag_t::UInt32 begin
    ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD = 1
    ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_memory_property_flag_t = _ze_device_memory_property_flag_t

@checked function zeDeviceGetMemoryProperties(hDevice, pCount, pMemProperties)
    @ccall libze_loader.zeDeviceGetMemoryProperties(hDevice::ze_device_handle_t,
                                                    pCount::Ptr{UInt32},
                                                    pMemProperties::Ptr{ze_device_memory_properties_t})::ze_result_t
end

@cenum _ze_memory_access_cap_flag_t::UInt32 begin
    ZE_MEMORY_ACCESS_CAP_FLAG_RW = 1
    ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC = 2
    ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT = 4
    ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC = 8
    ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_memory_access_cap_flag_t = _ze_memory_access_cap_flag_t

@checked function zeDeviceGetMemoryAccessProperties(hDevice, pMemAccessProperties)
    @ccall libze_loader.zeDeviceGetMemoryAccessProperties(hDevice::ze_device_handle_t,
                                                          pMemAccessProperties::Ptr{ze_device_memory_access_properties_t})::ze_result_t
end

@cenum _ze_device_cache_property_flag_t::UInt32 begin
    ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL = 1
    ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_cache_property_flag_t = _ze_device_cache_property_flag_t

@checked function zeDeviceGetCacheProperties(hDevice, pCount, pCacheProperties)
    @ccall libze_loader.zeDeviceGetCacheProperties(hDevice::ze_device_handle_t,
                                                   pCount::Ptr{UInt32},
                                                   pCacheProperties::Ptr{ze_device_cache_properties_t})::ze_result_t
end

@checked function zeDeviceGetImageProperties(hDevice, pImageProperties)
    @ccall libze_loader.zeDeviceGetImageProperties(hDevice::ze_device_handle_t,
                                                   pImageProperties::Ptr{ze_device_image_properties_t})::ze_result_t
end

@checked function zeDeviceGetExternalMemoryProperties(hDevice, pExternalMemoryProperties)
    @ccall libze_loader.zeDeviceGetExternalMemoryProperties(hDevice::ze_device_handle_t,
                                                            pExternalMemoryProperties::Ptr{ze_device_external_memory_properties_t})::ze_result_t
end

@cenum _ze_device_p2p_property_flag_t::UInt32 begin
    ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS = 1
    ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS = 2
    ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_p2p_property_flag_t = _ze_device_p2p_property_flag_t

@checked function zeDeviceGetP2PProperties(hDevice, hPeerDevice, pP2PProperties)
    @ccall libze_loader.zeDeviceGetP2PProperties(hDevice::ze_device_handle_t,
                                                 hPeerDevice::ze_device_handle_t,
                                                 pP2PProperties::Ptr{ze_device_p2p_properties_t})::ze_result_t
end

@checked function zeDeviceCanAccessPeer(hDevice, hPeerDevice, value)
    @ccall libze_loader.zeDeviceCanAccessPeer(hDevice::ze_device_handle_t,
                                              hPeerDevice::ze_device_handle_t,
                                              value::Ptr{ze_bool_t})::ze_result_t
end

@checked function zeDeviceGetStatus(hDevice)
    @ccall libze_loader.zeDeviceGetStatus(hDevice::ze_device_handle_t)::ze_result_t
end

@checked function zeDeviceGetGlobalTimestamps(hDevice, hostTimestamp, deviceTimestamp)
    @ccall libze_loader.zeDeviceGetGlobalTimestamps(hDevice::ze_device_handle_t,
                                                    hostTimestamp::Ptr{UInt64},
                                                    deviceTimestamp::Ptr{UInt64})::ze_result_t
end

@cenum _ze_context_flag_t::UInt32 begin
    ZE_CONTEXT_FLAG_TBD = 1
    ZE_CONTEXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_context_flag_t = _ze_context_flag_t

@checked function zeContextCreate(hDriver, desc, phContext)
    @ccall libze_loader.zeContextCreate(hDriver::ze_driver_handle_t,
                                        desc::Ptr{ze_context_desc_t},
                                        phContext::Ptr{ze_context_handle_t})::ze_result_t
end

@checked function zeContextCreateEx(hDriver, desc, numDevices, phDevices, phContext)
    @ccall libze_loader.zeContextCreateEx(hDriver::ze_driver_handle_t,
                                          desc::Ptr{ze_context_desc_t}, numDevices::UInt32,
                                          phDevices::Ptr{ze_device_handle_t},
                                          phContext::Ptr{ze_context_handle_t})::ze_result_t
end

@checked function zeContextDestroy(hContext)
    @ccall libze_loader.zeContextDestroy(hContext::ze_context_handle_t)::ze_result_t
end

@checked function zeContextGetStatus(hContext)
    @ccall libze_loader.zeContextGetStatus(hContext::ze_context_handle_t)::ze_result_t
end

@cenum _ze_command_queue_flag_t::UInt32 begin
    ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY = 1
    ZE_COMMAND_QUEUE_FLAG_IN_ORDER = 2
    ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_command_queue_flag_t = _ze_command_queue_flag_t

@checked function zeCommandQueueCreate(hContext, hDevice, desc, phCommandQueue)
    @ccall libze_loader.zeCommandQueueCreate(hContext::ze_context_handle_t,
                                             hDevice::ze_device_handle_t,
                                             desc::Ptr{ze_command_queue_desc_t},
                                             phCommandQueue::Ptr{ze_command_queue_handle_t})::ze_result_t
end

@checked function zeCommandQueueDestroy(hCommandQueue)
    @ccall libze_loader.zeCommandQueueDestroy(hCommandQueue::ze_command_queue_handle_t)::ze_result_t
end

@checked function zeCommandQueueExecuteCommandLists(hCommandQueue, numCommandLists,
                                                    phCommandLists, hFence)
    @ccall libze_loader.zeCommandQueueExecuteCommandLists(hCommandQueue::ze_command_queue_handle_t,
                                                          numCommandLists::UInt32,
                                                          phCommandLists::Ptr{ze_command_list_handle_t},
                                                          hFence::ze_fence_handle_t)::ze_result_t
end

@checked function zeCommandQueueSynchronize(hCommandQueue, timeout)
    @ccall libze_loader.zeCommandQueueSynchronize(hCommandQueue::ze_command_queue_handle_t,
                                                  timeout::UInt64)::ze_result_t
end

@checked function zeCommandQueueGetOrdinal(hCommandQueue, pOrdinal)
    @ccall libze_loader.zeCommandQueueGetOrdinal(hCommandQueue::ze_command_queue_handle_t,
                                                 pOrdinal::Ptr{UInt32})::ze_result_t
end

@checked function zeCommandQueueGetIndex(hCommandQueue, pIndex)
    @ccall libze_loader.zeCommandQueueGetIndex(hCommandQueue::ze_command_queue_handle_t,
                                               pIndex::Ptr{UInt32})::ze_result_t
end

@cenum _ze_command_list_flag_t::UInt32 begin
    ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = 1
    ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = 2
    ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = 4
    ZE_COMMAND_LIST_FLAG_IN_ORDER = 8
    ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE = 16
    ZE_COMMAND_LIST_FLAG_FORCE_UINT32 = 2147483647
end

const ze_command_list_flag_t = _ze_command_list_flag_t

@checked function zeCommandListCreate(hContext, hDevice, desc, phCommandList)
    @ccall libze_loader.zeCommandListCreate(hContext::ze_context_handle_t,
                                            hDevice::ze_device_handle_t,
                                            desc::Ptr{ze_command_list_desc_t},
                                            phCommandList::Ptr{ze_command_list_handle_t})::ze_result_t
end

@checked function zeCommandListCreateImmediate(hContext, hDevice, altdesc, phCommandList)
    @ccall libze_loader.zeCommandListCreateImmediate(hContext::ze_context_handle_t,
                                                     hDevice::ze_device_handle_t,
                                                     altdesc::Ptr{ze_command_queue_desc_t},
                                                     phCommandList::Ptr{ze_command_list_handle_t})::ze_result_t
end

@checked function zeCommandListDestroy(hCommandList)
    @ccall libze_loader.zeCommandListDestroy(hCommandList::ze_command_list_handle_t)::ze_result_t
end

@checked function zeCommandListClose(hCommandList)
    @ccall libze_loader.zeCommandListClose(hCommandList::ze_command_list_handle_t)::ze_result_t
end

@checked function zeCommandListReset(hCommandList)
    @ccall libze_loader.zeCommandListReset(hCommandList::ze_command_list_handle_t)::ze_result_t
end

@checked function zeCommandListAppendWriteGlobalTimestamp(hCommandList, dstptr,
                                                          hSignalEvent, numWaitEvents,
                                                          phWaitEvents)
    @ccall libze_loader.zeCommandListAppendWriteGlobalTimestamp(hCommandList::ze_command_list_handle_t,
                                                                dstptr::Ptr{UInt64},
                                                                hSignalEvent::ze_event_handle_t,
                                                                numWaitEvents::UInt32,
                                                                phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListHostSynchronize(hCommandList, timeout)
    @ccall libze_loader.zeCommandListHostSynchronize(hCommandList::ze_command_list_handle_t,
                                                     timeout::UInt64)::ze_result_t
end

@checked function zeCommandListGetDeviceHandle(hCommandList, phDevice)
    @ccall libze_loader.zeCommandListGetDeviceHandle(hCommandList::ze_command_list_handle_t,
                                                     phDevice::Ptr{ze_device_handle_t})::ze_result_t
end

@checked function zeCommandListGetContextHandle(hCommandList, phContext)
    @ccall libze_loader.zeCommandListGetContextHandle(hCommandList::ze_command_list_handle_t,
                                                      phContext::Ptr{ze_context_handle_t})::ze_result_t
end

@checked function zeCommandListGetOrdinal(hCommandList, pOrdinal)
    @ccall libze_loader.zeCommandListGetOrdinal(hCommandList::ze_command_list_handle_t,
                                                pOrdinal::Ptr{UInt32})::ze_result_t
end

@checked function zeCommandListImmediateGetIndex(hCommandListImmediate, pIndex)
    @ccall libze_loader.zeCommandListImmediateGetIndex(hCommandListImmediate::ze_command_list_handle_t,
                                                       pIndex::Ptr{UInt32})::ze_result_t
end

@checked function zeCommandListIsImmediate(hCommandList, pIsImmediate)
    @ccall libze_loader.zeCommandListIsImmediate(hCommandList::ze_command_list_handle_t,
                                                 pIsImmediate::Ptr{ze_bool_t})::ze_result_t
end

@checked function zeCommandListAppendBarrier(hCommandList, hSignalEvent, numWaitEvents,
                                             phWaitEvents)
    @ccall libze_loader.zeCommandListAppendBarrier(hCommandList::ze_command_list_handle_t,
                                                   hSignalEvent::ze_event_handle_t,
                                                   numWaitEvents::UInt32,
                                                   phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendMemoryRangesBarrier(hCommandList, numRanges,
                                                         pRangeSizes, pRanges, hSignalEvent,
                                                         numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendMemoryRangesBarrier(hCommandList::ze_command_list_handle_t,
                                                               numRanges::UInt32,
                                                               pRangeSizes::Ptr{Csize_t},
                                                               pRanges::Ptr{Ptr{Cvoid}},
                                                               hSignalEvent::ze_event_handle_t,
                                                               numWaitEvents::UInt32,
                                                               phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeContextSystemBarrier(hContext, hDevice)
    @ccall libze_loader.zeContextSystemBarrier(hContext::ze_context_handle_t,
                                               hDevice::ze_device_handle_t)::ze_result_t
end

@checked function zeCommandListAppendMemoryCopy(hCommandList, dstptr, srcptr, size,
                                                hSignalEvent, numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendMemoryCopy(hCommandList::ze_command_list_handle_t,
                                                      dstptr::PtrOrZePtr{Cvoid},
                                                      srcptr::PtrOrZePtr{Cvoid},
                                                      size::Csize_t,
                                                      hSignalEvent::ze_event_handle_t,
                                                      numWaitEvents::UInt32,
                                                      phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendMemoryFill(hCommandList, ptr, pattern, pattern_size,
                                                size, hSignalEvent, numWaitEvents,
                                                phWaitEvents)
    @ccall libze_loader.zeCommandListAppendMemoryFill(hCommandList::ze_command_list_handle_t,
                                                      ptr::PtrOrZePtr{Cvoid},
                                                      pattern::PtrOrZePtr{Cvoid},
                                                      pattern_size::Csize_t, size::Csize_t,
                                                      hSignalEvent::ze_event_handle_t,
                                                      numWaitEvents::UInt32,
                                                      phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendMemoryCopyRegion(hCommandList, dstptr, dstRegion,
                                                      dstPitch, dstSlicePitch, srcptr,
                                                      srcRegion, srcPitch, srcSlicePitch,
                                                      hSignalEvent, numWaitEvents,
                                                      phWaitEvents)
    @ccall libze_loader.zeCommandListAppendMemoryCopyRegion(hCommandList::ze_command_list_handle_t,
                                                            dstptr::PtrOrZePtr{Cvoid},
                                                            dstRegion::Ptr{ze_copy_region_t},
                                                            dstPitch::UInt32,
                                                            dstSlicePitch::UInt32,
                                                            srcptr::PtrOrZePtr{Cvoid},
                                                            srcRegion::Ptr{ze_copy_region_t},
                                                            srcPitch::UInt32,
                                                            srcSlicePitch::UInt32,
                                                            hSignalEvent::ze_event_handle_t,
                                                            numWaitEvents::UInt32,
                                                            phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendMemoryCopyFromContext(hCommandList, dstptr,
                                                           hContextSrc, srcptr, size,
                                                           hSignalEvent, numWaitEvents,
                                                           phWaitEvents)
    @ccall libze_loader.zeCommandListAppendMemoryCopyFromContext(hCommandList::ze_command_list_handle_t,
                                                                 dstptr::PtrOrZePtr{Cvoid},
                                                                 hContextSrc::ze_context_handle_t,
                                                                 srcptr::PtrOrZePtr{Cvoid},
                                                                 size::Csize_t,
                                                                 hSignalEvent::ze_event_handle_t,
                                                                 numWaitEvents::UInt32,
                                                                 phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendImageCopy(hCommandList, hDstImage, hSrcImage,
                                               hSignalEvent, numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendImageCopy(hCommandList::ze_command_list_handle_t,
                                                     hDstImage::ze_image_handle_t,
                                                     hSrcImage::ze_image_handle_t,
                                                     hSignalEvent::ze_event_handle_t,
                                                     numWaitEvents::UInt32,
                                                     phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendImageCopyRegion(hCommandList, hDstImage, hSrcImage,
                                                     pDstRegion, pSrcRegion, hSignalEvent,
                                                     numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendImageCopyRegion(hCommandList::ze_command_list_handle_t,
                                                           hDstImage::ze_image_handle_t,
                                                           hSrcImage::ze_image_handle_t,
                                                           pDstRegion::Ptr{ze_image_region_t},
                                                           pSrcRegion::Ptr{ze_image_region_t},
                                                           hSignalEvent::ze_event_handle_t,
                                                           numWaitEvents::UInt32,
                                                           phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendImageCopyToMemory(hCommandList, dstptr, hSrcImage,
                                                       pSrcRegion, hSignalEvent,
                                                       numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendImageCopyToMemory(hCommandList::ze_command_list_handle_t,
                                                             dstptr::Ptr{Cvoid},
                                                             hSrcImage::ze_image_handle_t,
                                                             pSrcRegion::Ptr{ze_image_region_t},
                                                             hSignalEvent::ze_event_handle_t,
                                                             numWaitEvents::UInt32,
                                                             phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendImageCopyFromMemory(hCommandList, hDstImage, srcptr,
                                                         pDstRegion, hSignalEvent,
                                                         numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendImageCopyFromMemory(hCommandList::ze_command_list_handle_t,
                                                               hDstImage::ze_image_handle_t,
                                                               srcptr::Ptr{Cvoid},
                                                               pDstRegion::Ptr{ze_image_region_t},
                                                               hSignalEvent::ze_event_handle_t,
                                                               numWaitEvents::UInt32,
                                                               phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendMemoryPrefetch(hCommandList, ptr, size)
    @ccall libze_loader.zeCommandListAppendMemoryPrefetch(hCommandList::ze_command_list_handle_t,
                                                          ptr::PtrOrZePtr{Cvoid},
                                                          size::Csize_t)::ze_result_t
end

@cenum _ze_memory_advice_t::UInt32 begin
    ZE_MEMORY_ADVICE_SET_READ_MOSTLY = 0
    ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY = 1
    ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION = 2
    ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION = 3
    ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY = 4
    ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 5
    ZE_MEMORY_ADVICE_BIAS_CACHED = 6
    ZE_MEMORY_ADVICE_BIAS_UNCACHED = 7
    ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION = 8
    ZE_MEMORY_ADVICE_CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION = 9
    ZE_MEMORY_ADVICE_FORCE_UINT32 = 2147483647
end

const ze_memory_advice_t = _ze_memory_advice_t

@checked function zeCommandListAppendMemAdvise(hCommandList, hDevice, ptr, size, advice)
    @ccall libze_loader.zeCommandListAppendMemAdvise(hCommandList::ze_command_list_handle_t,
                                                     hDevice::ze_device_handle_t,
                                                     ptr::PtrOrZePtr{Cvoid}, size::Csize_t,
                                                     advice::ze_memory_advice_t)::ze_result_t
end

@cenum _ze_event_pool_flag_t::UInt32 begin
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE = 1
    ZE_EVENT_POOL_FLAG_IPC = 2
    ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP = 4
    ZE_EVENT_POOL_FLAG_KERNEL_MAPPED_TIMESTAMP = 8
    ZE_EVENT_POOL_FLAG_FORCE_UINT32 = 2147483647
end

const ze_event_pool_flag_t = _ze_event_pool_flag_t

@checked function zeEventPoolCreate(hContext, desc, numDevices, phDevices, phEventPool)
    @ccall libze_loader.zeEventPoolCreate(hContext::ze_context_handle_t,
                                          desc::Ptr{ze_event_pool_desc_t},
                                          numDevices::UInt32,
                                          phDevices::Ptr{ze_device_handle_t},
                                          phEventPool::Ptr{ze_event_pool_handle_t})::ze_result_t
end

@checked function zeEventPoolDestroy(hEventPool)
    @ccall libze_loader.zeEventPoolDestroy(hEventPool::ze_event_pool_handle_t)::ze_result_t
end

@cenum _ze_event_scope_flag_t::UInt32 begin
    ZE_EVENT_SCOPE_FLAG_SUBDEVICE = 1
    ZE_EVENT_SCOPE_FLAG_DEVICE = 2
    ZE_EVENT_SCOPE_FLAG_HOST = 4
    ZE_EVENT_SCOPE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_event_scope_flag_t = _ze_event_scope_flag_t

@checked function zeEventCreate(hEventPool, desc, phEvent)
    @ccall libze_loader.zeEventCreate(hEventPool::ze_event_pool_handle_t,
                                      desc::Ptr{ze_event_desc_t},
                                      phEvent::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeEventDestroy(hEvent)
    @ccall libze_loader.zeEventDestroy(hEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeEventPoolGetIpcHandle(hEventPool, phIpc)
    @ccall libze_loader.zeEventPoolGetIpcHandle(hEventPool::ze_event_pool_handle_t,
                                                phIpc::Ptr{ze_ipc_event_pool_handle_t})::ze_result_t
end

@checked function zeEventPoolPutIpcHandle(hContext, hIpc)
    @ccall libze_loader.zeEventPoolPutIpcHandle(hContext::ze_context_handle_t,
                                                hIpc::ze_ipc_event_pool_handle_t)::ze_result_t
end

@checked function zeEventPoolOpenIpcHandle(hContext, hIpc, phEventPool)
    @ccall libze_loader.zeEventPoolOpenIpcHandle(hContext::ze_context_handle_t,
                                                 hIpc::ze_ipc_event_pool_handle_t,
                                                 phEventPool::Ptr{ze_event_pool_handle_t})::ze_result_t
end

@checked function zeEventPoolCloseIpcHandle(hEventPool)
    @ccall libze_loader.zeEventPoolCloseIpcHandle(hEventPool::ze_event_pool_handle_t)::ze_result_t
end

@checked function zeCommandListAppendSignalEvent(hCommandList, hEvent)
    @ccall libze_loader.zeCommandListAppendSignalEvent(hCommandList::ze_command_list_handle_t,
                                                       hEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeCommandListAppendWaitOnEvents(hCommandList, numEvents, phEvents)
    @ccall libze_loader.zeCommandListAppendWaitOnEvents(hCommandList::ze_command_list_handle_t,
                                                        numEvents::UInt32,
                                                        phEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeEventHostSignal(hEvent)
    @ccall libze_loader.zeEventHostSignal(hEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeEventHostSynchronize(hEvent, timeout)
    @ccall libze_loader.zeEventHostSynchronize(hEvent::ze_event_handle_t,
                                               timeout::UInt64)::ze_result_t
end

@checked function zeEventQueryStatus(hEvent)
    @ccall libze_loader.zeEventQueryStatus(hEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeCommandListAppendEventReset(hCommandList, hEvent)
    @ccall libze_loader.zeCommandListAppendEventReset(hCommandList::ze_command_list_handle_t,
                                                      hEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeEventHostReset(hEvent)
    @ccall libze_loader.zeEventHostReset(hEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeEventQueryKernelTimestamp(hEvent, dstptr)
    @ccall libze_loader.zeEventQueryKernelTimestamp(hEvent::ze_event_handle_t,
                                                    dstptr::Ptr{ze_kernel_timestamp_result_t})::ze_result_t
end

@checked function zeCommandListAppendQueryKernelTimestamps(hCommandList, numEvents,
                                                           phEvents, dstptr, pOffsets,
                                                           hSignalEvent, numWaitEvents,
                                                           phWaitEvents)
    @ccall libze_loader.zeCommandListAppendQueryKernelTimestamps(hCommandList::ze_command_list_handle_t,
                                                                 numEvents::UInt32,
                                                                 phEvents::Ptr{ze_event_handle_t},
                                                                 dstptr::Ptr{Cvoid},
                                                                 pOffsets::Ptr{Csize_t},
                                                                 hSignalEvent::ze_event_handle_t,
                                                                 numWaitEvents::UInt32,
                                                                 phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeEventGetEventPool(hEvent, phEventPool)
    @ccall libze_loader.zeEventGetEventPool(hEvent::ze_event_handle_t,
                                            phEventPool::Ptr{ze_event_pool_handle_t})::ze_result_t
end

@checked function zeEventGetSignalScope(hEvent, pSignalScope)
    @ccall libze_loader.zeEventGetSignalScope(hEvent::ze_event_handle_t,
                                              pSignalScope::Ptr{ze_event_scope_flags_t})::ze_result_t
end

@checked function zeEventGetWaitScope(hEvent, pWaitScope)
    @ccall libze_loader.zeEventGetWaitScope(hEvent::ze_event_handle_t,
                                            pWaitScope::Ptr{ze_event_scope_flags_t})::ze_result_t
end

@checked function zeEventPoolGetContextHandle(hEventPool, phContext)
    @ccall libze_loader.zeEventPoolGetContextHandle(hEventPool::ze_event_pool_handle_t,
                                                    phContext::Ptr{ze_context_handle_t})::ze_result_t
end

@checked function zeEventPoolGetFlags(hEventPool, pFlags)
    @ccall libze_loader.zeEventPoolGetFlags(hEventPool::ze_event_pool_handle_t,
                                            pFlags::Ptr{ze_event_pool_flags_t})::ze_result_t
end

@cenum _ze_fence_flag_t::UInt32 begin
    ZE_FENCE_FLAG_SIGNALED = 1
    ZE_FENCE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_fence_flag_t = _ze_fence_flag_t

@checked function zeFenceCreate(hCommandQueue, desc, phFence)
    @ccall libze_loader.zeFenceCreate(hCommandQueue::ze_command_queue_handle_t,
                                      desc::Ptr{ze_fence_desc_t},
                                      phFence::Ptr{ze_fence_handle_t})::ze_result_t
end

@checked function zeFenceDestroy(hFence)
    @ccall libze_loader.zeFenceDestroy(hFence::ze_fence_handle_t)::ze_result_t
end

@checked function zeFenceHostSynchronize(hFence, timeout)
    @ccall libze_loader.zeFenceHostSynchronize(hFence::ze_fence_handle_t,
                                               timeout::UInt64)::ze_result_t
end

@checked function zeFenceQueryStatus(hFence)
    @ccall libze_loader.zeFenceQueryStatus(hFence::ze_fence_handle_t)::ze_result_t
end

@checked function zeFenceReset(hFence)
    @ccall libze_loader.zeFenceReset(hFence::ze_fence_handle_t)::ze_result_t
end

@cenum _ze_image_flag_t::UInt32 begin
    ZE_IMAGE_FLAG_KERNEL_WRITE = 1
    ZE_IMAGE_FLAG_BIAS_UNCACHED = 2
    ZE_IMAGE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_image_flag_t = _ze_image_flag_t

@cenum _ze_image_sampler_filter_flag_t::UInt32 begin
    ZE_IMAGE_SAMPLER_FILTER_FLAG_POINT = 1
    ZE_IMAGE_SAMPLER_FILTER_FLAG_LINEAR = 2
    ZE_IMAGE_SAMPLER_FILTER_FLAG_FORCE_UINT32 = 2147483647
end

const ze_image_sampler_filter_flag_t = _ze_image_sampler_filter_flag_t

@checked function zeImageGetProperties(hDevice, desc, pImageProperties)
    @ccall libze_loader.zeImageGetProperties(hDevice::ze_device_handle_t,
                                             desc::Ptr{ze_image_desc_t},
                                             pImageProperties::Ptr{ze_image_properties_t})::ze_result_t
end

@checked function zeImageCreate(hContext, hDevice, desc, phImage)
    @ccall libze_loader.zeImageCreate(hContext::ze_context_handle_t,
                                      hDevice::ze_device_handle_t,
                                      desc::Ptr{ze_image_desc_t},
                                      phImage::Ptr{ze_image_handle_t})::ze_result_t
end

@checked function zeImageDestroy(hImage)
    @ccall libze_loader.zeImageDestroy(hImage::ze_image_handle_t)::ze_result_t
end

@cenum _ze_device_mem_alloc_flag_t::UInt32 begin
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED = 1
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = 4
    ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_mem_alloc_flag_t = _ze_device_mem_alloc_flag_t

@cenum _ze_host_mem_alloc_flag_t::UInt32 begin
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = 1
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED = 4
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT = 8
    ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32 = 2147483647
end

const ze_host_mem_alloc_flag_t = _ze_host_mem_alloc_flag_t

@checked function zeMemAllocShared(hContext, device_desc, host_desc, size, alignment,
                                   hDevice, pptr)
    @ccall libze_loader.zeMemAllocShared(hContext::ze_context_handle_t,
                                         device_desc::Ptr{ze_device_mem_alloc_desc_t},
                                         host_desc::Ptr{ze_host_mem_alloc_desc_t},
                                         size::Csize_t, alignment::Csize_t,
                                         hDevice::ze_device_handle_t,
                                         pptr::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeMemAllocDevice(hContext, device_desc, size, alignment, hDevice, pptr)
    @ccall libze_loader.zeMemAllocDevice(hContext::ze_context_handle_t,
                                         device_desc::Ptr{ze_device_mem_alloc_desc_t},
                                         size::Csize_t, alignment::Csize_t,
                                         hDevice::ze_device_handle_t,
                                         pptr::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeMemAllocHost(hContext, host_desc, size, alignment, pptr)
    @ccall libze_loader.zeMemAllocHost(hContext::ze_context_handle_t,
                                       host_desc::Ptr{ze_host_mem_alloc_desc_t},
                                       size::Csize_t, alignment::Csize_t,
                                       pptr::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeMemFree(hContext, ptr)
    @ccall libze_loader.zeMemFree(hContext::ze_context_handle_t,
                                  ptr::PtrOrZePtr{Cvoid})::ze_result_t
end

@checked function zeMemGetAllocProperties(hContext, ptr, pMemAllocProperties, phDevice)
    @ccall libze_loader.zeMemGetAllocProperties(hContext::ze_context_handle_t,
                                                ptr::PtrOrZePtr{Cvoid},
                                                pMemAllocProperties::Ptr{ze_memory_allocation_properties_t},
                                                phDevice::Ptr{ze_device_handle_t})::ze_result_t
end

@checked function zeMemGetAddressRange(hContext, ptr, pBase, pSize)
    @ccall libze_loader.zeMemGetAddressRange(hContext::ze_context_handle_t,
                                             ptr::PtrOrZePtr{Cvoid}, pBase::Ptr{Ptr{Cvoid}},
                                             pSize::Ptr{Csize_t})::ze_result_t
end

@checked function zeMemGetIpcHandle(hContext, ptr, pIpcHandle)
    @ccall libze_loader.zeMemGetIpcHandle(hContext::ze_context_handle_t,
                                          ptr::PtrOrZePtr{Cvoid},
                                          pIpcHandle::Ptr{ze_ipc_mem_handle_t})::ze_result_t
end

@checked function zeMemGetIpcHandleFromFileDescriptorExp(hContext, handle, pIpcHandle)
    @ccall libze_loader.zeMemGetIpcHandleFromFileDescriptorExp(hContext::ze_context_handle_t,
                                                               handle::UInt64,
                                                               pIpcHandle::Ptr{ze_ipc_mem_handle_t})::ze_result_t
end

@checked function zeMemGetFileDescriptorFromIpcHandleExp(hContext, ipcHandle, pHandle)
    @ccall libze_loader.zeMemGetFileDescriptorFromIpcHandleExp(hContext::ze_context_handle_t,
                                                               ipcHandle::ze_ipc_mem_handle_t,
                                                               pHandle::Ptr{UInt64})::ze_result_t
end

@checked function zeMemPutIpcHandle(hContext, handle)
    @ccall libze_loader.zeMemPutIpcHandle(hContext::ze_context_handle_t,
                                          handle::ze_ipc_mem_handle_t)::ze_result_t
end

const ze_ipc_memory_flags_t = UInt32

@cenum _ze_ipc_memory_flag_t::UInt32 begin
    ZE_IPC_MEMORY_FLAG_BIAS_CACHED = 1
    ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED = 2
    ZE_IPC_MEMORY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_ipc_memory_flag_t = _ze_ipc_memory_flag_t

@checked function zeMemOpenIpcHandle(hContext, hDevice, handle, flags, pptr)
    @ccall libze_loader.zeMemOpenIpcHandle(hContext::ze_context_handle_t,
                                           hDevice::ze_device_handle_t,
                                           handle::ze_ipc_mem_handle_t,
                                           flags::ze_ipc_memory_flags_t,
                                           pptr::Ptr{PtrOrZePtr{Cvoid}})::ze_result_t
end

@checked function zeMemCloseIpcHandle(hContext, ptr)
    @ccall libze_loader.zeMemCloseIpcHandle(hContext::ze_context_handle_t,
                                            ptr::PtrOrZePtr{Cvoid})::ze_result_t
end

const ze_memory_atomic_attr_exp_flags_t = UInt32

@cenum _ze_memory_atomic_attr_exp_flag_t::UInt32 begin
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_ATOMICS = 1
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_HOST_ATOMICS = 2
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_HOST_ATOMICS = 4
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_DEVICE_ATOMICS = 8
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_DEVICE_ATOMICS = 16
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_NO_SYSTEM_ATOMICS = 32
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_SYSTEM_ATOMICS = 64
    ZE_MEMORY_ATOMIC_ATTR_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_memory_atomic_attr_exp_flag_t = _ze_memory_atomic_attr_exp_flag_t

@checked function zeMemSetAtomicAccessAttributeExp(hContext, hDevice, ptr, size, attr)
    @ccall libze_loader.zeMemSetAtomicAccessAttributeExp(hContext::ze_context_handle_t,
                                                         hDevice::ze_device_handle_t,
                                                         ptr::Ptr{Cvoid}, size::Csize_t,
                                                         attr::ze_memory_atomic_attr_exp_flags_t)::ze_result_t
end

@checked function zeMemGetAtomicAccessAttributeExp(hContext, hDevice, ptr, size, pAttr)
    @ccall libze_loader.zeMemGetAtomicAccessAttributeExp(hContext::ze_context_handle_t,
                                                         hDevice::ze_device_handle_t,
                                                         ptr::Ptr{Cvoid}, size::Csize_t,
                                                         pAttr::Ptr{ze_memory_atomic_attr_exp_flags_t})::ze_result_t
end

@checked function zeModuleCreate(hContext, hDevice, desc, phModule, phBuildLog)
    @ccall libze_loader.zeModuleCreate(hContext::ze_context_handle_t,
                                       hDevice::ze_device_handle_t,
                                       desc::Ptr{ze_module_desc_t},
                                       phModule::Ptr{ze_module_handle_t},
                                       phBuildLog::Ptr{ze_module_build_log_handle_t})::ze_result_t
end

@checked function zeModuleDestroy(hModule)
    @ccall libze_loader.zeModuleDestroy(hModule::ze_module_handle_t)::ze_result_t
end

@checked function zeModuleDynamicLink(numModules, phModules, phLinkLog)
    @ccall libze_loader.zeModuleDynamicLink(numModules::UInt32,
                                            phModules::Ptr{ze_module_handle_t},
                                            phLinkLog::Ptr{ze_module_build_log_handle_t})::ze_result_t
end

@checked function zeModuleBuildLogDestroy(hModuleBuildLog)
    @ccall libze_loader.zeModuleBuildLogDestroy(hModuleBuildLog::ze_module_build_log_handle_t)::ze_result_t
end

@checked function zeModuleBuildLogGetString(hModuleBuildLog, pSize, pBuildLog)
    @ccall libze_loader.zeModuleBuildLogGetString(hModuleBuildLog::ze_module_build_log_handle_t,
                                                  pSize::Ptr{Csize_t},
                                                  pBuildLog::Ptr{Cchar})::ze_result_t
end

@checked function zeModuleGetNativeBinary(hModule, pSize, pModuleNativeBinary)
    @ccall libze_loader.zeModuleGetNativeBinary(hModule::ze_module_handle_t,
                                                pSize::Ptr{Csize_t},
                                                pModuleNativeBinary::Ptr{UInt8})::ze_result_t
end

@checked function zeModuleGetGlobalPointer(hModule, pGlobalName, pSize, pptr)
    @ccall libze_loader.zeModuleGetGlobalPointer(hModule::ze_module_handle_t,
                                                 pGlobalName::Ptr{Cchar},
                                                 pSize::Ptr{Csize_t},
                                                 pptr::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeModuleGetKernelNames(hModule, pCount, pNames)
    @ccall libze_loader.zeModuleGetKernelNames(hModule::ze_module_handle_t,
                                               pCount::Ptr{UInt32},
                                               pNames::Ptr{Ptr{Cchar}})::ze_result_t
end

@cenum _ze_module_property_flag_t::UInt32 begin
    ZE_MODULE_PROPERTY_FLAG_IMPORTS = 1
    ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_module_property_flag_t = _ze_module_property_flag_t

@checked function zeModuleGetProperties(hModule, pModuleProperties)
    @ccall libze_loader.zeModuleGetProperties(hModule::ze_module_handle_t,
                                              pModuleProperties::Ptr{ze_module_properties_t})::ze_result_t
end

@cenum _ze_kernel_flag_t::UInt32 begin
    ZE_KERNEL_FLAG_FORCE_RESIDENCY = 1
    ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY = 2
    ZE_KERNEL_FLAG_FORCE_UINT32 = 2147483647
end

const ze_kernel_flag_t = _ze_kernel_flag_t

@checked function zeKernelCreate(hModule, desc, phKernel)
    @ccall libze_loader.zeKernelCreate(hModule::ze_module_handle_t,
                                       desc::Ptr{ze_kernel_desc_t},
                                       phKernel::Ptr{ze_kernel_handle_t})::ze_result_t
end

@checked function zeKernelDestroy(hKernel)
    @ccall libze_loader.zeKernelDestroy(hKernel::ze_kernel_handle_t)::ze_result_t
end

@checked function zeModuleGetFunctionPointer(hModule, pFunctionName, pfnFunction)
    @ccall libze_loader.zeModuleGetFunctionPointer(hModule::ze_module_handle_t,
                                                   pFunctionName::Ptr{Cchar},
                                                   pfnFunction::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeKernelSetGroupSize(hKernel, groupSizeX, groupSizeY, groupSizeZ)
    @ccall libze_loader.zeKernelSetGroupSize(hKernel::ze_kernel_handle_t,
                                             groupSizeX::UInt32, groupSizeY::UInt32,
                                             groupSizeZ::UInt32)::ze_result_t
end

@checked function zeKernelSuggestGroupSize(hKernel, globalSizeX, globalSizeY, globalSizeZ,
                                           groupSizeX, groupSizeY, groupSizeZ)
    @ccall libze_loader.zeKernelSuggestGroupSize(hKernel::ze_kernel_handle_t,
                                                 globalSizeX::UInt32, globalSizeY::UInt32,
                                                 globalSizeZ::UInt32,
                                                 groupSizeX::Ptr{UInt32},
                                                 groupSizeY::Ptr{UInt32},
                                                 groupSizeZ::Ptr{UInt32})::ze_result_t
end

@checked function zeKernelSuggestMaxCooperativeGroupCount(hKernel, totalGroupCount)
    @ccall libze_loader.zeKernelSuggestMaxCooperativeGroupCount(hKernel::ze_kernel_handle_t,
                                                                totalGroupCount::Ptr{UInt32})::ze_result_t
end

@checked function zeKernelSetArgumentValue(hKernel, argIndex, argSize, pArgValue)
    @ccall libze_loader.zeKernelSetArgumentValue(hKernel::ze_kernel_handle_t,
                                                 argIndex::UInt32, argSize::Csize_t,
                                                 pArgValue::Ptr{Cvoid})::ze_result_t
end

const ze_kernel_indirect_access_flags_t = UInt32

@cenum _ze_kernel_indirect_access_flag_t::UInt32 begin
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = 1
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = 2
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = 4
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32 = 2147483647
end

const ze_kernel_indirect_access_flag_t = _ze_kernel_indirect_access_flag_t

@checked function zeKernelSetIndirectAccess(hKernel, flags)
    @ccall libze_loader.zeKernelSetIndirectAccess(hKernel::ze_kernel_handle_t,
                                                  flags::ze_kernel_indirect_access_flags_t)::ze_result_t
end

@checked function zeKernelGetIndirectAccess(hKernel, pFlags)
    @ccall libze_loader.zeKernelGetIndirectAccess(hKernel::ze_kernel_handle_t,
                                                  pFlags::Ptr{ze_kernel_indirect_access_flags_t})::ze_result_t
end

@checked function zeKernelGetSourceAttributes(hKernel, pSize, pString)
    @ccall libze_loader.zeKernelGetSourceAttributes(hKernel::ze_kernel_handle_t,
                                                    pSize::Ptr{UInt32},
                                                    pString::Ptr{Ptr{Cchar}})::ze_result_t
end

const ze_cache_config_flags_t = UInt32

@cenum _ze_cache_config_flag_t::UInt32 begin
    ZE_CACHE_CONFIG_FLAG_LARGE_SLM = 1
    ZE_CACHE_CONFIG_FLAG_LARGE_DATA = 2
    ZE_CACHE_CONFIG_FLAG_FORCE_UINT32 = 2147483647
end

const ze_cache_config_flag_t = _ze_cache_config_flag_t

@checked function zeKernelSetCacheConfig(hKernel, flags)
    @ccall libze_loader.zeKernelSetCacheConfig(hKernel::ze_kernel_handle_t,
                                               flags::ze_cache_config_flags_t)::ze_result_t
end

@checked function zeKernelGetProperties(hKernel, pKernelProperties)
    @ccall libze_loader.zeKernelGetProperties(hKernel::ze_kernel_handle_t,
                                              pKernelProperties::Ptr{ze_kernel_properties_t})::ze_result_t
end

@checked function zeKernelGetName(hKernel, pSize, pName)
    @ccall libze_loader.zeKernelGetName(hKernel::ze_kernel_handle_t, pSize::Ptr{Csize_t},
                                        pName::Ptr{Cchar})::ze_result_t
end

@checked function zeCommandListAppendLaunchKernel(hCommandList, hKernel, pLaunchFuncArgs,
                                                  hSignalEvent, numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendLaunchKernel(hCommandList::ze_command_list_handle_t,
                                                        hKernel::ze_kernel_handle_t,
                                                        pLaunchFuncArgs::Ptr{ze_group_count_t},
                                                        hSignalEvent::ze_event_handle_t,
                                                        numWaitEvents::UInt32,
                                                        phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendLaunchCooperativeKernel(hCommandList, hKernel,
                                                             pLaunchFuncArgs, hSignalEvent,
                                                             numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendLaunchCooperativeKernel(hCommandList::ze_command_list_handle_t,
                                                                   hKernel::ze_kernel_handle_t,
                                                                   pLaunchFuncArgs::Ptr{ze_group_count_t},
                                                                   hSignalEvent::ze_event_handle_t,
                                                                   numWaitEvents::UInt32,
                                                                   phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendLaunchKernelIndirect(hCommandList, hKernel,
                                                          pLaunchArgumentsBuffer,
                                                          hSignalEvent, numWaitEvents,
                                                          phWaitEvents)
    @ccall libze_loader.zeCommandListAppendLaunchKernelIndirect(hCommandList::ze_command_list_handle_t,
                                                                hKernel::ze_kernel_handle_t,
                                                                pLaunchArgumentsBuffer::Ptr{ze_group_count_t},
                                                                hSignalEvent::ze_event_handle_t,
                                                                numWaitEvents::UInt32,
                                                                phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendLaunchMultipleKernelsIndirect(hCommandList, numKernels,
                                                                   phKernels, pCountBuffer,
                                                                   pLaunchArgumentsBuffer,
                                                                   hSignalEvent,
                                                                   numWaitEvents,
                                                                   phWaitEvents)
    @ccall libze_loader.zeCommandListAppendLaunchMultipleKernelsIndirect(hCommandList::ze_command_list_handle_t,
                                                                         numKernels::UInt32,
                                                                         phKernels::Ptr{ze_kernel_handle_t},
                                                                         pCountBuffer::Ptr{UInt32},
                                                                         pLaunchArgumentsBuffer::Ptr{ze_group_count_t},
                                                                         hSignalEvent::ze_event_handle_t,
                                                                         numWaitEvents::UInt32,
                                                                         phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@cenum _ze_module_program_exp_version_t::UInt32 begin
    ZE_MODULE_PROGRAM_EXP_VERSION_1_0 = 65536
    ZE_MODULE_PROGRAM_EXP_VERSION_CURRENT = 65536
    ZE_MODULE_PROGRAM_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_module_program_exp_version_t = _ze_module_program_exp_version_t

@cenum _ze_raytracing_ext_version_t::UInt32 begin
    ZE_RAYTRACING_EXT_VERSION_1_0 = 65536
    ZE_RAYTRACING_EXT_VERSION_CURRENT = 65536
    ZE_RAYTRACING_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_raytracing_ext_version_t = _ze_raytracing_ext_version_t

@cenum _ze_device_raytracing_ext_flag_t::UInt32 begin
    ZE_DEVICE_RAYTRACING_EXT_FLAG_RAYQUERY = 1
    ZE_DEVICE_RAYTRACING_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_raytracing_ext_flag_t = _ze_device_raytracing_ext_flag_t

@cenum _ze_raytracing_mem_alloc_ext_flag_t::UInt32 begin
    ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_TBD = 1
    ZE_RAYTRACING_MEM_ALLOC_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_raytracing_mem_alloc_ext_flag_t = _ze_raytracing_mem_alloc_ext_flag_t

@checked function zeContextMakeMemoryResident(hContext, hDevice, ptr, size)
    @ccall libze_loader.zeContextMakeMemoryResident(hContext::ze_context_handle_t,
                                                    hDevice::ze_device_handle_t,
                                                    ptr::PtrOrZePtr{Cvoid},
                                                    size::Csize_t)::ze_result_t
end

@checked function zeContextEvictMemory(hContext, hDevice, ptr, size)
    @ccall libze_loader.zeContextEvictMemory(hContext::ze_context_handle_t,
                                             hDevice::ze_device_handle_t,
                                             ptr::PtrOrZePtr{Cvoid},
                                             size::Csize_t)::ze_result_t
end

@checked function zeContextMakeImageResident(hContext, hDevice, hImage)
    @ccall libze_loader.zeContextMakeImageResident(hContext::ze_context_handle_t,
                                                   hDevice::ze_device_handle_t,
                                                   hImage::ze_image_handle_t)::ze_result_t
end

@checked function zeContextEvictImage(hContext, hDevice, hImage)
    @ccall libze_loader.zeContextEvictImage(hContext::ze_context_handle_t,
                                            hDevice::ze_device_handle_t,
                                            hImage::ze_image_handle_t)::ze_result_t
end

@checked function zeSamplerCreate(hContext, hDevice, desc, phSampler)
    @ccall libze_loader.zeSamplerCreate(hContext::ze_context_handle_t,
                                        hDevice::ze_device_handle_t,
                                        desc::Ptr{ze_sampler_desc_t},
                                        phSampler::Ptr{ze_sampler_handle_t})::ze_result_t
end

@checked function zeSamplerDestroy(hSampler)
    @ccall libze_loader.zeSamplerDestroy(hSampler::ze_sampler_handle_t)::ze_result_t
end

@cenum _ze_memory_access_attribute_t::UInt32 begin
    ZE_MEMORY_ACCESS_ATTRIBUTE_NONE = 0
    ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1
    ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY = 2
    ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32 = 2147483647
end

const ze_memory_access_attribute_t = _ze_memory_access_attribute_t

@checked function zeVirtualMemReserve(hContext, pStart, size, pptr)
    @ccall libze_loader.zeVirtualMemReserve(hContext::ze_context_handle_t,
                                            pStart::Ptr{Cvoid}, size::Csize_t,
                                            pptr::Ptr{Ptr{Cvoid}})::ze_result_t
end

@checked function zeVirtualMemFree(hContext, ptr, size)
    @ccall libze_loader.zeVirtualMemFree(hContext::ze_context_handle_t,
                                         ptr::PtrOrZePtr{Cvoid}, size::Csize_t)::ze_result_t
end

@checked function zeVirtualMemQueryPageSize(hContext, hDevice, size, pagesize)
    @ccall libze_loader.zeVirtualMemQueryPageSize(hContext::ze_context_handle_t,
                                                  hDevice::ze_device_handle_t,
                                                  size::Csize_t,
                                                  pagesize::Ptr{Csize_t})::ze_result_t
end

@cenum _ze_physical_mem_flag_t::UInt32 begin
    ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE = 1
    ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_HOST = 2
    ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32 = 2147483647
end

const ze_physical_mem_flag_t = _ze_physical_mem_flag_t

@checked function zePhysicalMemCreate(hContext, hDevice, desc, phPhysicalMemory)
    @ccall libze_loader.zePhysicalMemCreate(hContext::ze_context_handle_t,
                                            hDevice::ze_device_handle_t,
                                            desc::Ptr{ze_physical_mem_desc_t},
                                            phPhysicalMemory::Ptr{ze_physical_mem_handle_t})::ze_result_t
end

@checked function zePhysicalMemDestroy(hContext, hPhysicalMemory)
    @ccall libze_loader.zePhysicalMemDestroy(hContext::ze_context_handle_t,
                                             hPhysicalMemory::ze_physical_mem_handle_t)::ze_result_t
end

@checked function zeVirtualMemMap(hContext, ptr, size, hPhysicalMemory, offset, access)
    @ccall libze_loader.zeVirtualMemMap(hContext::ze_context_handle_t, ptr::Ptr{Cvoid},
                                        size::Csize_t,
                                        hPhysicalMemory::ze_physical_mem_handle_t,
                                        offset::Csize_t,
                                        access::ze_memory_access_attribute_t)::ze_result_t
end

@checked function zeVirtualMemUnmap(hContext, ptr, size)
    @ccall libze_loader.zeVirtualMemUnmap(hContext::ze_context_handle_t, ptr::Ptr{Cvoid},
                                          size::Csize_t)::ze_result_t
end

@checked function zeVirtualMemSetAccessAttribute(hContext, ptr, size, access)
    @ccall libze_loader.zeVirtualMemSetAccessAttribute(hContext::ze_context_handle_t,
                                                       ptr::Ptr{Cvoid}, size::Csize_t,
                                                       access::ze_memory_access_attribute_t)::ze_result_t
end

@checked function zeVirtualMemGetAccessAttribute(hContext, ptr, size, access, outSize)
    @ccall libze_loader.zeVirtualMemGetAccessAttribute(hContext::ze_context_handle_t,
                                                       ptr::Ptr{Cvoid}, size::Csize_t,
                                                       access::Ptr{ze_memory_access_attribute_t},
                                                       outSize::Ptr{Csize_t})::ze_result_t
end

@cenum _ze_float_atomics_ext_version_t::UInt32 begin
    ZE_FLOAT_ATOMICS_EXT_VERSION_1_0 = 65536
    ZE_FLOAT_ATOMICS_EXT_VERSION_CURRENT = 65536
    ZE_FLOAT_ATOMICS_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_float_atomics_ext_version_t = _ze_float_atomics_ext_version_t

@cenum _ze_device_fp_atomic_ext_flag_t::UInt32 begin
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE = 1
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD = 2
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX = 4
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE = 65536
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD = 131072
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX = 262144
    ZE_DEVICE_FP_ATOMIC_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_fp_atomic_ext_flag_t = _ze_device_fp_atomic_ext_flag_t

@cenum _ze_global_offset_exp_version_t::UInt32 begin
    ZE_GLOBAL_OFFSET_EXP_VERSION_1_0 = 65536
    ZE_GLOBAL_OFFSET_EXP_VERSION_CURRENT = 65536
    ZE_GLOBAL_OFFSET_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_global_offset_exp_version_t = _ze_global_offset_exp_version_t

@checked function zeKernelSetGlobalOffsetExp(hKernel, offsetX, offsetY, offsetZ)
    @ccall libze_loader.zeKernelSetGlobalOffsetExp(hKernel::ze_kernel_handle_t,
                                                   offsetX::UInt32, offsetY::UInt32,
                                                   offsetZ::UInt32)::ze_result_t
end

@cenum _ze_relaxed_allocation_limits_exp_version_t::UInt32 begin
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_1_0 = 65536
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_CURRENT = 65536
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_relaxed_allocation_limits_exp_version_t = _ze_relaxed_allocation_limits_exp_version_t

@cenum _ze_relaxed_allocation_limits_exp_flag_t::UInt32 begin
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE = 1
    ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_relaxed_allocation_limits_exp_flag_t = _ze_relaxed_allocation_limits_exp_flag_t

@cenum _ze_kernel_get_binary_exp_version_t::UInt32 begin
    ZE_KERNEL_GET_BINARY_EXP_VERSION_1_0 = 65536
    ZE_KERNEL_GET_BINARY_EXP_VERSION_CURRENT = 65536
    ZE_KERNEL_GET_BINARY_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_kernel_get_binary_exp_version_t = _ze_kernel_get_binary_exp_version_t

@checked function zeKernelGetBinaryExp(hKernel, pSize, pKernelBinary)
    @ccall libze_loader.zeKernelGetBinaryExp(hKernel::ze_kernel_handle_t,
                                             pSize::Ptr{Csize_t},
                                             pKernelBinary::Ptr{UInt8})::ze_result_t
end

@cenum _ze_driver_ddi_handles_ext_version_t::UInt32 begin
    ZE_DRIVER_DDI_HANDLES_EXT_VERSION_1_0 = 65536
    ZE_DRIVER_DDI_HANDLES_EXT_VERSION_CURRENT = 65536
    ZE_DRIVER_DDI_HANDLES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_driver_ddi_handles_ext_version_t = _ze_driver_ddi_handles_ext_version_t

@cenum _ze_driver_ddi_handle_ext_flag_t::UInt32 begin
    ZE_DRIVER_DDI_HANDLE_EXT_FLAG_DDI_HANDLE_EXT_SUPPORTED = 1
    ZE_DRIVER_DDI_HANDLE_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_driver_ddi_handle_ext_flag_t = _ze_driver_ddi_handle_ext_flag_t

@cenum _ze_external_semaphore_ext_version_t::UInt32 begin
    ZE_EXTERNAL_SEMAPHORE_EXT_VERSION_1_0 = 65536
    ZE_EXTERNAL_SEMAPHORE_EXT_VERSION_CURRENT = 65536
    ZE_EXTERNAL_SEMAPHORE_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_external_semaphore_ext_version_t = _ze_external_semaphore_ext_version_t

mutable struct _ze_external_semaphore_ext_handle_t end

const ze_external_semaphore_ext_handle_t = Ptr{_ze_external_semaphore_ext_handle_t}

@cenum _ze_external_semaphore_ext_flag_t::UInt32 begin
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_OPAQUE_FD = 1
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_OPAQUE_WIN32 = 2
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_OPAQUE_WIN32_KMT = 4
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_D3D12_FENCE = 8
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_D3D11_FENCE = 16
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_KEYED_MUTEX = 32
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_KEYED_MUTEX_KMT = 64
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_VK_TIMELINE_SEMAPHORE_FD = 128
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_VK_TIMELINE_SEMAPHORE_WIN32 = 256
    ZE_EXTERNAL_SEMAPHORE_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_external_semaphore_ext_flag_t = _ze_external_semaphore_ext_flag_t

@checked function zeDeviceImportExternalSemaphoreExt(hDevice, desc, phSemaphore)
    @ccall libze_loader.zeDeviceImportExternalSemaphoreExt(hDevice::ze_device_handle_t,
                                                           desc::Ptr{ze_external_semaphore_ext_desc_t},
                                                           phSemaphore::Ptr{ze_external_semaphore_ext_handle_t})::ze_result_t
end

@checked function zeDeviceReleaseExternalSemaphoreExt(hSemaphore)
    @ccall libze_loader.zeDeviceReleaseExternalSemaphoreExt(hSemaphore::ze_external_semaphore_ext_handle_t)::ze_result_t
end

@checked function zeCommandListAppendSignalExternalSemaphoreExt(hCommandList, numSemaphores,
                                                                phSemaphores, signalParams,
                                                                hSignalEvent, numWaitEvents,
                                                                phWaitEvents)
    @ccall libze_loader.zeCommandListAppendSignalExternalSemaphoreExt(hCommandList::ze_command_list_handle_t,
                                                                      numSemaphores::UInt32,
                                                                      phSemaphores::Ptr{ze_external_semaphore_ext_handle_t},
                                                                      signalParams::Ptr{ze_external_semaphore_signal_params_ext_t},
                                                                      hSignalEvent::ze_event_handle_t,
                                                                      numWaitEvents::UInt32,
                                                                      phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendWaitExternalSemaphoreExt(hCommandList, numSemaphores,
                                                              phSemaphores, waitParams,
                                                              hSignalEvent, numWaitEvents,
                                                              phWaitEvents)
    @ccall libze_loader.zeCommandListAppendWaitExternalSemaphoreExt(hCommandList::ze_command_list_handle_t,
                                                                    numSemaphores::UInt32,
                                                                    phSemaphores::Ptr{ze_external_semaphore_ext_handle_t},
                                                                    waitParams::Ptr{ze_external_semaphore_wait_params_ext_t},
                                                                    hSignalEvent::ze_event_handle_t,
                                                                    numWaitEvents::UInt32,
                                                                    phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@cenum _ze_device_cache_line_size_ext_version_t::UInt32 begin
    ZE_DEVICE_CACHE_LINE_SIZE_EXT_VERSION_1_0 = 65536
    ZE_DEVICE_CACHE_LINE_SIZE_EXT_VERSION_CURRENT = 65536
    ZE_DEVICE_CACHE_LINE_SIZE_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_device_cache_line_size_ext_version_t = _ze_device_cache_line_size_ext_version_t

@cenum _ze_rtas_device_ext_flag_t::UInt32 begin
    ZE_RTAS_DEVICE_EXT_FLAG_RESERVED = 1
    ZE_RTAS_DEVICE_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_device_ext_flag_t = _ze_rtas_device_ext_flag_t

@cenum _ze_rtas_builder_ext_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_EXT_FLAG_RESERVED = 1
    ZE_RTAS_BUILDER_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_ext_flag_t = _ze_rtas_builder_ext_flag_t

@cenum _ze_rtas_parallel_operation_ext_flag_t::UInt32 begin
    ZE_RTAS_PARALLEL_OPERATION_EXT_FLAG_RESERVED = 1
    ZE_RTAS_PARALLEL_OPERATION_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_parallel_operation_ext_flag_t = _ze_rtas_parallel_operation_ext_flag_t

const ze_rtas_builder_geometry_ext_flags_t = UInt32

@cenum _ze_rtas_builder_geometry_ext_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_GEOMETRY_EXT_FLAG_NON_OPAQUE = 1
    ZE_RTAS_BUILDER_GEOMETRY_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_geometry_ext_flag_t = _ze_rtas_builder_geometry_ext_flag_t

const ze_rtas_builder_instance_ext_flags_t = UInt32

@cenum _ze_rtas_builder_instance_ext_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_INSTANCE_EXT_FLAG_TRIANGLE_CULL_DISABLE = 1
    ZE_RTAS_BUILDER_INSTANCE_EXT_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE = 2
    ZE_RTAS_BUILDER_INSTANCE_EXT_FLAG_TRIANGLE_FORCE_OPAQUE = 4
    ZE_RTAS_BUILDER_INSTANCE_EXT_FLAG_TRIANGLE_FORCE_NON_OPAQUE = 8
    ZE_RTAS_BUILDER_INSTANCE_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_instance_ext_flag_t = _ze_rtas_builder_instance_ext_flag_t

@cenum _ze_rtas_builder_build_op_ext_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_BUILD_OP_EXT_FLAG_COMPACT = 1
    ZE_RTAS_BUILDER_BUILD_OP_EXT_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION = 2
    ZE_RTAS_BUILDER_BUILD_OP_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_build_op_ext_flag_t = _ze_rtas_builder_build_op_ext_flag_t

@cenum _ze_rtas_builder_geometry_type_ext_t::UInt32 begin
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_TRIANGLES = 0
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_QUADS = 1
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_PROCEDURAL = 2
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_INSTANCE = 3
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_geometry_type_ext_t = _ze_rtas_builder_geometry_type_ext_t

@cenum _ze_rtas_builder_input_data_format_ext_t::UInt32 begin
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3 = 0
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3X4_COLUMN_MAJOR = 1
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3X4_ALIGNED_COLUMN_MAJOR = 2
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3X4_ROW_MAJOR = 3
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_AABB = 4
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_TRIANGLE_INDICES_UINT32 = 5
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_QUAD_INDICES_UINT32 = 6
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_input_data_format_ext_t = _ze_rtas_builder_input_data_format_ext_t

mutable struct _ze_rtas_builder_ext_handle_t end

const ze_rtas_builder_ext_handle_t = Ptr{_ze_rtas_builder_ext_handle_t}

mutable struct _ze_rtas_parallel_operation_ext_handle_t end

const ze_rtas_parallel_operation_ext_handle_t = Ptr{_ze_rtas_parallel_operation_ext_handle_t}

@checked function zeRTASBuilderCreateExt(hDriver, pDescriptor, phBuilder)
    @ccall libze_loader.zeRTASBuilderCreateExt(hDriver::ze_driver_handle_t,
                                               pDescriptor::Ptr{ze_rtas_builder_ext_desc_t},
                                               phBuilder::Ptr{ze_rtas_builder_ext_handle_t})::ze_result_t
end

@checked function zeRTASBuilderGetBuildPropertiesExt(hBuilder, pBuildOpDescriptor,
                                                     pProperties)
    @ccall libze_loader.zeRTASBuilderGetBuildPropertiesExt(hBuilder::ze_rtas_builder_ext_handle_t,
                                                           pBuildOpDescriptor::Ptr{ze_rtas_builder_build_op_ext_desc_t},
                                                           pProperties::Ptr{ze_rtas_builder_ext_properties_t})::ze_result_t
end

@checked function zeDriverRTASFormatCompatibilityCheckExt(hDriver, rtasFormatA, rtasFormatB)
    @ccall libze_loader.zeDriverRTASFormatCompatibilityCheckExt(hDriver::ze_driver_handle_t,
                                                                rtasFormatA::ze_rtas_format_ext_t,
                                                                rtasFormatB::ze_rtas_format_ext_t)::ze_result_t
end

@checked function zeRTASBuilderBuildExt(hBuilder, pBuildOpDescriptor, pScratchBuffer,
                                        scratchBufferSizeBytes, pRtasBuffer,
                                        rtasBufferSizeBytes, hParallelOperation,
                                        pBuildUserPtr, pBounds, pRtasBufferSizeBytes)
    @ccall libze_loader.zeRTASBuilderBuildExt(hBuilder::ze_rtas_builder_ext_handle_t,
                                              pBuildOpDescriptor::Ptr{ze_rtas_builder_build_op_ext_desc_t},
                                              pScratchBuffer::Ptr{Cvoid},
                                              scratchBufferSizeBytes::Csize_t,
                                              pRtasBuffer::Ptr{Cvoid},
                                              rtasBufferSizeBytes::Csize_t,
                                              hParallelOperation::ze_rtas_parallel_operation_ext_handle_t,
                                              pBuildUserPtr::Ptr{Cvoid},
                                              pBounds::Ptr{ze_rtas_aabb_ext_t},
                                              pRtasBufferSizeBytes::Ptr{Csize_t})::ze_result_t
end

@checked function zeRTASBuilderCommandListAppendCopyExt(hCommandList, dstptr, srcptr, size,
                                                        hSignalEvent, numWaitEvents,
                                                        phWaitEvents)
    @ccall libze_loader.zeRTASBuilderCommandListAppendCopyExt(hCommandList::ze_command_list_handle_t,
                                                              dstptr::Ptr{Cvoid},
                                                              srcptr::Ptr{Cvoid},
                                                              size::Csize_t,
                                                              hSignalEvent::ze_event_handle_t,
                                                              numWaitEvents::UInt32,
                                                              phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeRTASBuilderDestroyExt(hBuilder)
    @ccall libze_loader.zeRTASBuilderDestroyExt(hBuilder::ze_rtas_builder_ext_handle_t)::ze_result_t
end

@checked function zeRTASParallelOperationCreateExt(hDriver, phParallelOperation)
    @ccall libze_loader.zeRTASParallelOperationCreateExt(hDriver::ze_driver_handle_t,
                                                         phParallelOperation::Ptr{ze_rtas_parallel_operation_ext_handle_t})::ze_result_t
end

@checked function zeRTASParallelOperationGetPropertiesExt(hParallelOperation, pProperties)
    @ccall libze_loader.zeRTASParallelOperationGetPropertiesExt(hParallelOperation::ze_rtas_parallel_operation_ext_handle_t,
                                                                pProperties::Ptr{ze_rtas_parallel_operation_ext_properties_t})::ze_result_t
end

@checked function zeRTASParallelOperationJoinExt(hParallelOperation)
    @ccall libze_loader.zeRTASParallelOperationJoinExt(hParallelOperation::ze_rtas_parallel_operation_ext_handle_t)::ze_result_t
end

@checked function zeRTASParallelOperationDestroyExt(hParallelOperation)
    @ccall libze_loader.zeRTASParallelOperationDestroyExt(hParallelOperation::ze_rtas_parallel_operation_ext_handle_t)::ze_result_t
end

@cenum _ze_device_vector_sizes_ext_version_t::UInt32 begin
    ZE_DEVICE_VECTOR_SIZES_EXT_VERSION_1_0 = 65536
    ZE_DEVICE_VECTOR_SIZES_EXT_VERSION_CURRENT = 65536
    ZE_DEVICE_VECTOR_SIZES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_device_vector_sizes_ext_version_t = _ze_device_vector_sizes_ext_version_t

@checked function zeDeviceGetVectorWidthPropertiesExt(hDevice, pCount,
                                                      pVectorWidthProperties)
    @ccall libze_loader.zeDeviceGetVectorWidthPropertiesExt(hDevice::ze_device_handle_t,
                                                            pCount::Ptr{UInt32},
                                                            pVectorWidthProperties::Ptr{ze_device_vector_width_properties_ext_t})::ze_result_t
end

@cenum _ze_cache_reservation_ext_version_t::UInt32 begin
    ZE_CACHE_RESERVATION_EXT_VERSION_1_0 = 65536
    ZE_CACHE_RESERVATION_EXT_VERSION_CURRENT = 65536
    ZE_CACHE_RESERVATION_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_cache_reservation_ext_version_t = _ze_cache_reservation_ext_version_t

@cenum _ze_cache_ext_region_t::UInt32 begin
    ZE_CACHE_EXT_REGION_ZE_CACHE_REGION_DEFAULT = 0
    ZE_CACHE_EXT_REGION_ZE_CACHE_RESERVE_REGION = 1
    ZE_CACHE_EXT_REGION_ZE_CACHE_NON_RESERVED_REGION = 2
    ZE_CACHE_EXT_REGION_DEFAULT = 0
    ZE_CACHE_EXT_REGION_RESERVED = 1
    ZE_CACHE_EXT_REGION_NON_RESERVED = 2
    ZE_CACHE_EXT_REGION_FORCE_UINT32 = 2147483647
end

const ze_cache_ext_region_t = _ze_cache_ext_region_t

@checked function zeDeviceReserveCacheExt(hDevice, cacheLevel, cacheReservationSize)
    @ccall libze_loader.zeDeviceReserveCacheExt(hDevice::ze_device_handle_t,
                                                cacheLevel::Csize_t,
                                                cacheReservationSize::Csize_t)::ze_result_t
end

@checked function zeDeviceSetCacheAdviceExt(hDevice, ptr, regionSize, cacheRegion)
    @ccall libze_loader.zeDeviceSetCacheAdviceExt(hDevice::ze_device_handle_t,
                                                  ptr::Ptr{Cvoid}, regionSize::Csize_t,
                                                  cacheRegion::ze_cache_ext_region_t)::ze_result_t
end

@cenum _ze_event_query_timestamps_exp_version_t::UInt32 begin
    ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_1_0 = 65536
    ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_CURRENT = 65536
    ZE_EVENT_QUERY_TIMESTAMPS_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_event_query_timestamps_exp_version_t = _ze_event_query_timestamps_exp_version_t

@checked function zeEventQueryTimestampsExp(hEvent, hDevice, pCount, pTimestamps)
    @ccall libze_loader.zeEventQueryTimestampsExp(hEvent::ze_event_handle_t,
                                                  hDevice::ze_device_handle_t,
                                                  pCount::Ptr{UInt32},
                                                  pTimestamps::Ptr{ze_kernel_timestamp_result_t})::ze_result_t
end

@cenum _ze_image_memory_properties_exp_version_t::UInt32 begin
    ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_1_0 = 65536
    ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_CURRENT = 65536
    ZE_IMAGE_MEMORY_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_memory_properties_exp_version_t = _ze_image_memory_properties_exp_version_t

@checked function zeImageGetMemoryPropertiesExp(hImage, pMemoryProperties)
    @ccall libze_loader.zeImageGetMemoryPropertiesExp(hImage::ze_image_handle_t,
                                                      pMemoryProperties::Ptr{ze_image_memory_properties_exp_t})::ze_result_t
end

@cenum _ze_image_view_ext_version_t::UInt32 begin
    ZE_IMAGE_VIEW_EXT_VERSION_1_0 = 65536
    ZE_IMAGE_VIEW_EXT_VERSION_CURRENT = 65536
    ZE_IMAGE_VIEW_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_view_ext_version_t = _ze_image_view_ext_version_t

@checked function zeImageViewCreateExt(hContext, hDevice, desc, hImage, phImageView)
    @ccall libze_loader.zeImageViewCreateExt(hContext::ze_context_handle_t,
                                             hDevice::ze_device_handle_t,
                                             desc::Ptr{ze_image_desc_t},
                                             hImage::ze_image_handle_t,
                                             phImageView::Ptr{ze_image_handle_t})::ze_result_t
end

@cenum _ze_image_view_exp_version_t::UInt32 begin
    ZE_IMAGE_VIEW_EXP_VERSION_1_0 = 65536
    ZE_IMAGE_VIEW_EXP_VERSION_CURRENT = 65536
    ZE_IMAGE_VIEW_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_view_exp_version_t = _ze_image_view_exp_version_t

@checked function zeImageViewCreateExp(hContext, hDevice, desc, hImage, phImageView)
    @ccall libze_loader.zeImageViewCreateExp(hContext::ze_context_handle_t,
                                             hDevice::ze_device_handle_t,
                                             desc::Ptr{ze_image_desc_t},
                                             hImage::ze_image_handle_t,
                                             phImageView::Ptr{ze_image_handle_t})::ze_result_t
end

@cenum _ze_image_view_planar_ext_version_t::UInt32 begin
    ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_1_0 = 65536
    ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_CURRENT = 65536
    ZE_IMAGE_VIEW_PLANAR_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_view_planar_ext_version_t = _ze_image_view_planar_ext_version_t

@cenum _ze_image_view_planar_exp_version_t::UInt32 begin
    ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_1_0 = 65536
    ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_CURRENT = 65536
    ZE_IMAGE_VIEW_PLANAR_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_view_planar_exp_version_t = _ze_image_view_planar_exp_version_t

@cenum _ze_scheduling_hints_exp_version_t::UInt32 begin
    ZE_SCHEDULING_HINTS_EXP_VERSION_1_0 = 65536
    ZE_SCHEDULING_HINTS_EXP_VERSION_CURRENT = 65536
    ZE_SCHEDULING_HINTS_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_scheduling_hints_exp_version_t = _ze_scheduling_hints_exp_version_t

@cenum _ze_scheduling_hint_exp_flag_t::UInt32 begin
    ZE_SCHEDULING_HINT_EXP_FLAG_OLDEST_FIRST = 1
    ZE_SCHEDULING_HINT_EXP_FLAG_ROUND_ROBIN = 2
    ZE_SCHEDULING_HINT_EXP_FLAG_STALL_BASED_ROUND_ROBIN = 4
    ZE_SCHEDULING_HINT_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_scheduling_hint_exp_flag_t = _ze_scheduling_hint_exp_flag_t

@checked function zeKernelSchedulingHintExp(hKernel, pHint)
    @ccall libze_loader.zeKernelSchedulingHintExp(hKernel::ze_kernel_handle_t,
                                                  pHint::Ptr{ze_scheduling_hint_exp_desc_t})::ze_result_t
end

@cenum _ze_linkonce_odr_ext_version_t::UInt32 begin
    ZE_LINKONCE_ODR_EXT_VERSION_1_0 = 65536
    ZE_LINKONCE_ODR_EXT_VERSION_CURRENT = 65536
    ZE_LINKONCE_ODR_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_linkonce_odr_ext_version_t = _ze_linkonce_odr_ext_version_t

@cenum _ze_power_saving_hint_exp_version_t::UInt32 begin
    ZE_POWER_SAVING_HINT_EXP_VERSION_1_0 = 65536
    ZE_POWER_SAVING_HINT_EXP_VERSION_CURRENT = 65536
    ZE_POWER_SAVING_HINT_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_power_saving_hint_exp_version_t = _ze_power_saving_hint_exp_version_t

@cenum _ze_power_saving_hint_type_t::UInt32 begin
    ZE_POWER_SAVING_HINT_TYPE_MIN = 0
    ZE_POWER_SAVING_HINT_TYPE_MAX = 100
    ZE_POWER_SAVING_HINT_TYPE_FORCE_UINT32 = 2147483647
end

const ze_power_saving_hint_type_t = _ze_power_saving_hint_type_t

@cenum _ze_subgroup_ext_version_t::UInt32 begin
    ZE_SUBGROUP_EXT_VERSION_1_0 = 65536
    ZE_SUBGROUP_EXT_VERSION_CURRENT = 65536
    ZE_SUBGROUP_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_subgroup_ext_version_t = _ze_subgroup_ext_version_t

@cenum _ze_eu_count_ext_version_t::UInt32 begin
    ZE_EU_COUNT_EXT_VERSION_1_0 = 65536
    ZE_EU_COUNT_EXT_VERSION_CURRENT = 65536
    ZE_EU_COUNT_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_eu_count_ext_version_t = _ze_eu_count_ext_version_t

@cenum _ze_pci_properties_ext_version_t::UInt32 begin
    ZE_PCI_PROPERTIES_EXT_VERSION_1_0 = 65536
    ZE_PCI_PROPERTIES_EXT_VERSION_CURRENT = 65536
    ZE_PCI_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_pci_properties_ext_version_t = _ze_pci_properties_ext_version_t

@checked function zeDevicePciGetPropertiesExt(hDevice, pPciProperties)
    @ccall libze_loader.zeDevicePciGetPropertiesExt(hDevice::ze_device_handle_t,
                                                    pPciProperties::Ptr{ze_pci_ext_properties_t})::ze_result_t
end

@cenum _ze_srgb_ext_version_t::UInt32 begin
    ZE_SRGB_EXT_VERSION_1_0 = 65536
    ZE_SRGB_EXT_VERSION_CURRENT = 65536
    ZE_SRGB_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_srgb_ext_version_t = _ze_srgb_ext_version_t

@cenum _ze_image_copy_ext_version_t::UInt32 begin
    ZE_IMAGE_COPY_EXT_VERSION_1_0 = 65536
    ZE_IMAGE_COPY_EXT_VERSION_CURRENT = 65536
    ZE_IMAGE_COPY_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_copy_ext_version_t = _ze_image_copy_ext_version_t

@checked function zeCommandListAppendImageCopyToMemoryExt(hCommandList, dstptr, hSrcImage,
                                                          pSrcRegion, destRowPitch,
                                                          destSlicePitch, hSignalEvent,
                                                          numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendImageCopyToMemoryExt(hCommandList::ze_command_list_handle_t,
                                                                dstptr::Ptr{Cvoid},
                                                                hSrcImage::ze_image_handle_t,
                                                                pSrcRegion::Ptr{ze_image_region_t},
                                                                destRowPitch::UInt32,
                                                                destSlicePitch::UInt32,
                                                                hSignalEvent::ze_event_handle_t,
                                                                numWaitEvents::UInt32,
                                                                phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListAppendImageCopyFromMemoryExt(hCommandList, hDstImage, srcptr,
                                                            pDstRegion, srcRowPitch,
                                                            srcSlicePitch, hSignalEvent,
                                                            numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListAppendImageCopyFromMemoryExt(hCommandList::ze_command_list_handle_t,
                                                                  hDstImage::ze_image_handle_t,
                                                                  srcptr::Ptr{Cvoid},
                                                                  pDstRegion::Ptr{ze_image_region_t},
                                                                  srcRowPitch::UInt32,
                                                                  srcSlicePitch::UInt32,
                                                                  hSignalEvent::ze_event_handle_t,
                                                                  numWaitEvents::UInt32,
                                                                  phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@cenum _ze_image_query_alloc_properties_ext_version_t::UInt32 begin
    ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_1_0 = 65536
    ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_CURRENT = 65536
    ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_image_query_alloc_properties_ext_version_t = _ze_image_query_alloc_properties_ext_version_t

@checked function zeImageGetAllocPropertiesExt(hContext, hImage, pImageAllocProperties)
    @ccall libze_loader.zeImageGetAllocPropertiesExt(hContext::ze_context_handle_t,
                                                     hImage::ze_image_handle_t,
                                                     pImageAllocProperties::Ptr{ze_image_allocation_ext_properties_t})::ze_result_t
end

@cenum _ze_linkage_inspection_ext_version_t::UInt32 begin
    ZE_LINKAGE_INSPECTION_EXT_VERSION_1_0 = 65536
    ZE_LINKAGE_INSPECTION_EXT_VERSION_CURRENT = 65536
    ZE_LINKAGE_INSPECTION_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_linkage_inspection_ext_version_t = _ze_linkage_inspection_ext_version_t

@cenum _ze_linkage_inspection_ext_flag_t::UInt32 begin
    ZE_LINKAGE_INSPECTION_EXT_FLAG_IMPORTS = 1
    ZE_LINKAGE_INSPECTION_EXT_FLAG_UNRESOLVABLE_IMPORTS = 2
    ZE_LINKAGE_INSPECTION_EXT_FLAG_EXPORTS = 4
    ZE_LINKAGE_INSPECTION_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_linkage_inspection_ext_flag_t = _ze_linkage_inspection_ext_flag_t

@checked function zeModuleInspectLinkageExt(pInspectDesc, numModules, phModules, phLog)
    @ccall libze_loader.zeModuleInspectLinkageExt(pInspectDesc::Ptr{ze_linkage_inspection_ext_desc_t},
                                                  numModules::UInt32,
                                                  phModules::Ptr{ze_module_handle_t},
                                                  phLog::Ptr{ze_module_build_log_handle_t})::ze_result_t
end

@cenum _ze_memory_compression_hints_ext_version_t::UInt32 begin
    ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_1_0 = 65536
    ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_CURRENT = 65536
    ZE_MEMORY_COMPRESSION_HINTS_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_memory_compression_hints_ext_version_t = _ze_memory_compression_hints_ext_version_t

@cenum _ze_memory_compression_hints_ext_flag_t::UInt32 begin
    ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_COMPRESSED = 1
    ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_UNCOMPRESSED = 2
    ZE_MEMORY_COMPRESSION_HINTS_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_memory_compression_hints_ext_flag_t = _ze_memory_compression_hints_ext_flag_t

@cenum _ze_memory_free_policies_ext_version_t::UInt32 begin
    ZE_MEMORY_FREE_POLICIES_EXT_VERSION_1_0 = 65536
    ZE_MEMORY_FREE_POLICIES_EXT_VERSION_CURRENT = 65536
    ZE_MEMORY_FREE_POLICIES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_memory_free_policies_ext_version_t = _ze_memory_free_policies_ext_version_t

@cenum _ze_driver_memory_free_policy_ext_flag_t::UInt32 begin
    ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_BLOCKING_FREE = 1
    ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE = 2
    ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_driver_memory_free_policy_ext_flag_t = _ze_driver_memory_free_policy_ext_flag_t

@checked function zeMemFreeExt(hContext, pMemFreeDesc, ptr)
    @ccall libze_loader.zeMemFreeExt(hContext::ze_context_handle_t,
                                     pMemFreeDesc::Ptr{ze_memory_free_ext_desc_t},
                                     ptr::PtrOrZePtr{Cvoid})::ze_result_t
end

@cenum _ze_device_luid_ext_version_t::UInt32 begin
    ZE_DEVICE_LUID_EXT_VERSION_1_0 = 65536
    ZE_DEVICE_LUID_EXT_VERSION_CURRENT = 65536
    ZE_DEVICE_LUID_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_device_luid_ext_version_t = _ze_device_luid_ext_version_t

@checked function zeFabricVertexGetExp(hDriver, pCount, phVertices)
    @ccall libze_loader.zeFabricVertexGetExp(hDriver::ze_driver_handle_t,
                                             pCount::Ptr{UInt32},
                                             phVertices::Ptr{ze_fabric_vertex_handle_t})::ze_result_t
end

@checked function zeFabricVertexGetSubVerticesExp(hVertex, pCount, phSubvertices)
    @ccall libze_loader.zeFabricVertexGetSubVerticesExp(hVertex::ze_fabric_vertex_handle_t,
                                                        pCount::Ptr{UInt32},
                                                        phSubvertices::Ptr{ze_fabric_vertex_handle_t})::ze_result_t
end

@checked function zeFabricVertexGetPropertiesExp(hVertex, pVertexProperties)
    @ccall libze_loader.zeFabricVertexGetPropertiesExp(hVertex::ze_fabric_vertex_handle_t,
                                                       pVertexProperties::Ptr{ze_fabric_vertex_exp_properties_t})::ze_result_t
end

@checked function zeFabricVertexGetDeviceExp(hVertex, phDevice)
    @ccall libze_loader.zeFabricVertexGetDeviceExp(hVertex::ze_fabric_vertex_handle_t,
                                                   phDevice::Ptr{ze_device_handle_t})::ze_result_t
end

@checked function zeDeviceGetFabricVertexExp(hDevice, phVertex)
    @ccall libze_loader.zeDeviceGetFabricVertexExp(hDevice::ze_device_handle_t,
                                                   phVertex::Ptr{ze_fabric_vertex_handle_t})::ze_result_t
end

@checked function zeFabricEdgeGetExp(hVertexA, hVertexB, pCount, phEdges)
    @ccall libze_loader.zeFabricEdgeGetExp(hVertexA::ze_fabric_vertex_handle_t,
                                           hVertexB::ze_fabric_vertex_handle_t,
                                           pCount::Ptr{UInt32},
                                           phEdges::Ptr{ze_fabric_edge_handle_t})::ze_result_t
end

@checked function zeFabricEdgeGetVerticesExp(hEdge, phVertexA, phVertexB)
    @ccall libze_loader.zeFabricEdgeGetVerticesExp(hEdge::ze_fabric_edge_handle_t,
                                                   phVertexA::Ptr{ze_fabric_vertex_handle_t},
                                                   phVertexB::Ptr{ze_fabric_vertex_handle_t})::ze_result_t
end

@checked function zeFabricEdgeGetPropertiesExp(hEdge, pEdgeProperties)
    @ccall libze_loader.zeFabricEdgeGetPropertiesExp(hEdge::ze_fabric_edge_handle_t,
                                                     pEdgeProperties::Ptr{ze_fabric_edge_exp_properties_t})::ze_result_t
end

@cenum _ze_device_memory_properties_ext_version_t::UInt32 begin
    ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_1_0 = 65536
    ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_CURRENT = 65536
    ZE_DEVICE_MEMORY_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_device_memory_properties_ext_version_t = _ze_device_memory_properties_ext_version_t

@cenum _ze_bfloat16_conversions_ext_version_t::UInt32 begin
    ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_1_0 = 65536
    ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_CURRENT = 65536
    ZE_BFLOAT16_CONVERSIONS_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_bfloat16_conversions_ext_version_t = _ze_bfloat16_conversions_ext_version_t

@cenum _ze_device_ip_version_version_t::UInt32 begin
    ZE_DEVICE_IP_VERSION_VERSION_1_0 = 65536
    ZE_DEVICE_IP_VERSION_VERSION_CURRENT = 65536
    ZE_DEVICE_IP_VERSION_VERSION_FORCE_UINT32 = 2147483647
end

const ze_device_ip_version_version_t = _ze_device_ip_version_version_t

@cenum _ze_kernel_max_group_size_properties_ext_version_t::UInt32 begin
    ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_1_0 = 65536
    ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_CURRENT = 65536
    ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_kernel_max_group_size_properties_ext_version_t = _ze_kernel_max_group_size_properties_ext_version_t

const ze_kernel_max_group_size_ext_properties_t = ze_kernel_max_group_size_properties_ext_t

@cenum _ze_sub_allocations_exp_version_t::UInt32 begin
    ZE_SUB_ALLOCATIONS_EXP_VERSION_1_0 = 65536
    ZE_SUB_ALLOCATIONS_EXP_VERSION_CURRENT = 65536
    ZE_SUB_ALLOCATIONS_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_sub_allocations_exp_version_t = _ze_sub_allocations_exp_version_t

@cenum _ze_event_query_kernel_timestamps_ext_version_t::UInt32 begin
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_1_0 = 65536
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_CURRENT = 65536
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_VERSION_FORCE_UINT32 = 2147483647
end

const ze_event_query_kernel_timestamps_ext_version_t = _ze_event_query_kernel_timestamps_ext_version_t

@cenum _ze_event_query_kernel_timestamps_ext_flag_t::UInt32 begin
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_KERNEL = 1
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_SYNCHRONIZED = 2
    ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_event_query_kernel_timestamps_ext_flag_t = _ze_event_query_kernel_timestamps_ext_flag_t

@checked function zeEventQueryKernelTimestampsExt(hEvent, hDevice, pCount, pResults)
    @ccall libze_loader.zeEventQueryKernelTimestampsExt(hEvent::ze_event_handle_t,
                                                        hDevice::ze_device_handle_t,
                                                        pCount::Ptr{UInt32},
                                                        pResults::Ptr{ze_event_query_kernel_timestamps_results_ext_properties_t})::ze_result_t
end

@cenum _ze_rtas_device_exp_flag_t::UInt32 begin
    ZE_RTAS_DEVICE_EXP_FLAG_RESERVED = 1
    ZE_RTAS_DEVICE_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_device_exp_flag_t = _ze_rtas_device_exp_flag_t

@cenum _ze_rtas_builder_exp_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_EXP_FLAG_RESERVED = 1
    ZE_RTAS_BUILDER_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_exp_flag_t = _ze_rtas_builder_exp_flag_t

@cenum _ze_rtas_parallel_operation_exp_flag_t::UInt32 begin
    ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_RESERVED = 1
    ZE_RTAS_PARALLEL_OPERATION_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_parallel_operation_exp_flag_t = _ze_rtas_parallel_operation_exp_flag_t

const ze_rtas_builder_geometry_exp_flags_t = UInt32

@cenum _ze_rtas_builder_geometry_exp_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_NON_OPAQUE = 1
    ZE_RTAS_BUILDER_GEOMETRY_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_geometry_exp_flag_t = _ze_rtas_builder_geometry_exp_flag_t

const ze_rtas_builder_instance_exp_flags_t = UInt32

@cenum _ze_rtas_builder_instance_exp_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_CULL_DISABLE = 1
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE = 2
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_OPAQUE = 4
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_TRIANGLE_FORCE_NON_OPAQUE = 8
    ZE_RTAS_BUILDER_INSTANCE_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_instance_exp_flag_t = _ze_rtas_builder_instance_exp_flag_t

@cenum _ze_rtas_builder_build_op_exp_flag_t::UInt32 begin
    ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_COMPACT = 1
    ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION = 2
    ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_build_op_exp_flag_t = _ze_rtas_builder_build_op_exp_flag_t

@cenum _ze_rtas_builder_geometry_type_exp_t::UInt32 begin
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES = 0
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS = 1
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL = 2
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE = 3
    ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_geometry_type_exp_t = _ze_rtas_builder_geometry_type_exp_t

@cenum _ze_rtas_builder_input_data_format_exp_t::UInt32 begin
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3 = 0
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_COLUMN_MAJOR = 1
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ALIGNED_COLUMN_MAJOR = 2
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3X4_ROW_MAJOR = 3
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_AABB = 4
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32 = 5
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32 = 6
    ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FORCE_UINT32 = 2147483647
end

const ze_rtas_builder_input_data_format_exp_t = _ze_rtas_builder_input_data_format_exp_t

mutable struct _ze_rtas_builder_exp_handle_t end

const ze_rtas_builder_exp_handle_t = Ptr{_ze_rtas_builder_exp_handle_t}

mutable struct _ze_rtas_parallel_operation_exp_handle_t end

const ze_rtas_parallel_operation_exp_handle_t = Ptr{_ze_rtas_parallel_operation_exp_handle_t}

@checked function zeRTASBuilderCreateExp(hDriver, pDescriptor, phBuilder)
    @ccall libze_loader.zeRTASBuilderCreateExp(hDriver::ze_driver_handle_t,
                                               pDescriptor::Ptr{ze_rtas_builder_exp_desc_t},
                                               phBuilder::Ptr{ze_rtas_builder_exp_handle_t})::ze_result_t
end

@checked function zeRTASBuilderGetBuildPropertiesExp(hBuilder, pBuildOpDescriptor,
                                                     pProperties)
    @ccall libze_loader.zeRTASBuilderGetBuildPropertiesExp(hBuilder::ze_rtas_builder_exp_handle_t,
                                                           pBuildOpDescriptor::Ptr{ze_rtas_builder_build_op_exp_desc_t},
                                                           pProperties::Ptr{ze_rtas_builder_exp_properties_t})::ze_result_t
end

@checked function zeDriverRTASFormatCompatibilityCheckExp(hDriver, rtasFormatA, rtasFormatB)
    @ccall libze_loader.zeDriverRTASFormatCompatibilityCheckExp(hDriver::ze_driver_handle_t,
                                                                rtasFormatA::ze_rtas_format_exp_t,
                                                                rtasFormatB::ze_rtas_format_exp_t)::ze_result_t
end

@checked function zeRTASBuilderBuildExp(hBuilder, pBuildOpDescriptor, pScratchBuffer,
                                        scratchBufferSizeBytes, pRtasBuffer,
                                        rtasBufferSizeBytes, hParallelOperation,
                                        pBuildUserPtr, pBounds, pRtasBufferSizeBytes)
    @ccall libze_loader.zeRTASBuilderBuildExp(hBuilder::ze_rtas_builder_exp_handle_t,
                                              pBuildOpDescriptor::Ptr{ze_rtas_builder_build_op_exp_desc_t},
                                              pScratchBuffer::Ptr{Cvoid},
                                              scratchBufferSizeBytes::Csize_t,
                                              pRtasBuffer::Ptr{Cvoid},
                                              rtasBufferSizeBytes::Csize_t,
                                              hParallelOperation::ze_rtas_parallel_operation_exp_handle_t,
                                              pBuildUserPtr::Ptr{Cvoid},
                                              pBounds::Ptr{ze_rtas_aabb_exp_t},
                                              pRtasBufferSizeBytes::Ptr{Csize_t})::ze_result_t
end

@checked function zeRTASBuilderDestroyExp(hBuilder)
    @ccall libze_loader.zeRTASBuilderDestroyExp(hBuilder::ze_rtas_builder_exp_handle_t)::ze_result_t
end

@checked function zeRTASParallelOperationCreateExp(hDriver, phParallelOperation)
    @ccall libze_loader.zeRTASParallelOperationCreateExp(hDriver::ze_driver_handle_t,
                                                         phParallelOperation::Ptr{ze_rtas_parallel_operation_exp_handle_t})::ze_result_t
end

@checked function zeRTASParallelOperationGetPropertiesExp(hParallelOperation, pProperties)
    @ccall libze_loader.zeRTASParallelOperationGetPropertiesExp(hParallelOperation::ze_rtas_parallel_operation_exp_handle_t,
                                                                pProperties::Ptr{ze_rtas_parallel_operation_exp_properties_t})::ze_result_t
end

@checked function zeRTASParallelOperationJoinExp(hParallelOperation)
    @ccall libze_loader.zeRTASParallelOperationJoinExp(hParallelOperation::ze_rtas_parallel_operation_exp_handle_t)::ze_result_t
end

@checked function zeRTASParallelOperationDestroyExp(hParallelOperation)
    @ccall libze_loader.zeRTASParallelOperationDestroyExp(hParallelOperation::ze_rtas_parallel_operation_exp_handle_t)::ze_result_t
end

@cenum _ze_event_pool_counter_based_exp_version_t::UInt32 begin
    ZE_EVENT_POOL_COUNTER_BASED_EXP_VERSION_1_0 = 65536
    ZE_EVENT_POOL_COUNTER_BASED_EXP_VERSION_CURRENT = 65536
    ZE_EVENT_POOL_COUNTER_BASED_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_event_pool_counter_based_exp_version_t = _ze_event_pool_counter_based_exp_version_t

@cenum _ze_event_pool_counter_based_exp_flag_t::UInt32 begin
    ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE = 1
    ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_NON_IMMEDIATE = 2
    ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_event_pool_counter_based_exp_flag_t = _ze_event_pool_counter_based_exp_flag_t

@cenum _ze_bindless_image_exp_version_t::UInt32 begin
    ZE_BINDLESS_IMAGE_EXP_VERSION_1_0 = 65536
    ZE_BINDLESS_IMAGE_EXP_VERSION_CURRENT = 65536
    ZE_BINDLESS_IMAGE_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_bindless_image_exp_version_t = _ze_bindless_image_exp_version_t

@cenum _ze_image_bindless_exp_flag_t::UInt32 begin
    ZE_IMAGE_BINDLESS_EXP_FLAG_BINDLESS = 1
    ZE_IMAGE_BINDLESS_EXP_FLAG_SAMPLED_IMAGE = 2
    ZE_IMAGE_BINDLESS_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_image_bindless_exp_flag_t = _ze_image_bindless_exp_flag_t

@checked function zeMemGetPitchFor2dImage(hContext, hDevice, imageWidth, imageHeight,
                                          elementSizeInBytes, rowPitch)
    @ccall libze_loader.zeMemGetPitchFor2dImage(hContext::ze_context_handle_t,
                                                hDevice::ze_device_handle_t,
                                                imageWidth::Csize_t, imageHeight::Csize_t,
                                                elementSizeInBytes::Cuint,
                                                rowPitch::Ptr{Csize_t})::ze_result_t
end

@checked function zeImageGetDeviceOffsetExp(hImage, pDeviceOffset)
    @ccall libze_loader.zeImageGetDeviceOffsetExp(hImage::ze_image_handle_t,
                                                  pDeviceOffset::Ptr{UInt64})::ze_result_t
end

@cenum _ze_command_list_clone_exp_version_t::UInt32 begin
    ZE_COMMAND_LIST_CLONE_EXP_VERSION_1_0 = 65536
    ZE_COMMAND_LIST_CLONE_EXP_VERSION_CURRENT = 65536
    ZE_COMMAND_LIST_CLONE_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_command_list_clone_exp_version_t = _ze_command_list_clone_exp_version_t

@checked function zeCommandListCreateCloneExp(hCommandList, phClonedCommandList)
    @ccall libze_loader.zeCommandListCreateCloneExp(hCommandList::ze_command_list_handle_t,
                                                    phClonedCommandList::Ptr{ze_command_list_handle_t})::ze_result_t
end

@cenum _ze_immediate_command_list_append_exp_version_t::UInt32 begin
    ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_VERSION_1_0 = 65536
    ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_VERSION_CURRENT = 65536
    ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_immediate_command_list_append_exp_version_t = _ze_immediate_command_list_append_exp_version_t

@checked function zeCommandListImmediateAppendCommandListsExp(hCommandListImmediate,
                                                              numCommandLists,
                                                              phCommandLists, hSignalEvent,
                                                              numWaitEvents, phWaitEvents)
    @ccall libze_loader.zeCommandListImmediateAppendCommandListsExp(hCommandListImmediate::ze_command_list_handle_t,
                                                                    numCommandLists::UInt32,
                                                                    phCommandLists::Ptr{ze_command_list_handle_t},
                                                                    hSignalEvent::ze_event_handle_t,
                                                                    numWaitEvents::UInt32,
                                                                    phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@cenum _ze_mutable_command_list_exp_version_t::UInt32 begin
    ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_0 = 65536
    ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1 = 65537
    ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_CURRENT = 65537
    ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_FORCE_UINT32 = 2147483647
end

const ze_mutable_command_list_exp_version_t = _ze_mutable_command_list_exp_version_t

@cenum _ze_mutable_command_exp_flag_t::UInt32 begin
    ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS = 1
    ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT = 2
    ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE = 4
    ZE_MUTABLE_COMMAND_EXP_FLAG_GLOBAL_OFFSET = 8
    ZE_MUTABLE_COMMAND_EXP_FLAG_SIGNAL_EVENT = 16
    ZE_MUTABLE_COMMAND_EXP_FLAG_WAIT_EVENTS = 32
    ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION = 64
    ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENTS = 128
    ZE_MUTABLE_COMMAND_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_mutable_command_exp_flag_t = _ze_mutable_command_exp_flag_t

@cenum _ze_mutable_command_list_exp_flag_t::UInt32 begin
    ZE_MUTABLE_COMMAND_LIST_EXP_FLAG_RESERVED = 1
    ZE_MUTABLE_COMMAND_LIST_EXP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_mutable_command_list_exp_flag_t = _ze_mutable_command_list_exp_flag_t

@checked function zeCommandListGetNextCommandIdExp(hCommandList, desc, pCommandId)
    @ccall libze_loader.zeCommandListGetNextCommandIdExp(hCommandList::ze_command_list_handle_t,
                                                         desc::Ptr{ze_mutable_command_id_exp_desc_t},
                                                         pCommandId::Ptr{UInt64})::ze_result_t
end

@checked function zeCommandListGetNextCommandIdWithKernelsExp(hCommandList, desc,
                                                              numKernels, phKernels,
                                                              pCommandId)
    @ccall libze_loader.zeCommandListGetNextCommandIdWithKernelsExp(hCommandList::ze_command_list_handle_t,
                                                                    desc::Ptr{ze_mutable_command_id_exp_desc_t},
                                                                    numKernels::UInt32,
                                                                    phKernels::Ptr{ze_kernel_handle_t},
                                                                    pCommandId::Ptr{UInt64})::ze_result_t
end

@checked function zeCommandListUpdateMutableCommandsExp(hCommandList, desc)
    @ccall libze_loader.zeCommandListUpdateMutableCommandsExp(hCommandList::ze_command_list_handle_t,
                                                              desc::Ptr{ze_mutable_commands_exp_desc_t})::ze_result_t
end

@checked function zeCommandListUpdateMutableCommandSignalEventExp(hCommandList, commandId,
                                                                  hSignalEvent)
    @ccall libze_loader.zeCommandListUpdateMutableCommandSignalEventExp(hCommandList::ze_command_list_handle_t,
                                                                        commandId::UInt64,
                                                                        hSignalEvent::ze_event_handle_t)::ze_result_t
end

@checked function zeCommandListUpdateMutableCommandWaitEventsExp(hCommandList, commandId,
                                                                 numWaitEvents,
                                                                 phWaitEvents)
    @ccall libze_loader.zeCommandListUpdateMutableCommandWaitEventsExp(hCommandList::ze_command_list_handle_t,
                                                                       commandId::UInt64,
                                                                       numWaitEvents::UInt32,
                                                                       phWaitEvents::Ptr{ze_event_handle_t})::ze_result_t
end

@checked function zeCommandListUpdateMutableCommandKernelsExp(hCommandList, numKernels,
                                                              pCommandId, phKernels)
    @ccall libze_loader.zeCommandListUpdateMutableCommandKernelsExp(hCommandList::ze_command_list_handle_t,
                                                                    numKernels::UInt32,
                                                                    pCommandId::Ptr{UInt64},
                                                                    phKernels::Ptr{ze_kernel_handle_t})::ze_result_t
end

struct _ze_init_params_t
    pflags::Ptr{ze_init_flags_t}
end

const ze_init_params_t = _ze_init_params_t

# typedef void ( ZE_APICALL * ze_pfnInitCb_t ) ( ze_init_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnInitCb_t = Ptr{Cvoid}

struct _ze_global_callbacks_t
    pfnInitCb::ze_pfnInitCb_t
end

const ze_global_callbacks_t = _ze_global_callbacks_t

struct _ze_driver_get_params_t
    ppCount::Ptr{Ptr{UInt32}}
    pphDrivers::Ptr{Ptr{ze_driver_handle_t}}
end

const ze_driver_get_params_t = _ze_driver_get_params_t

# typedef void ( ZE_APICALL * ze_pfnDriverGetCb_t ) ( ze_driver_get_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDriverGetCb_t = Ptr{Cvoid}

struct _ze_driver_get_api_version_params_t
    phDriver::Ptr{ze_driver_handle_t}
    pversion::Ptr{Ptr{ze_api_version_t}}
end

const ze_driver_get_api_version_params_t = _ze_driver_get_api_version_params_t

# typedef void ( ZE_APICALL * ze_pfnDriverGetApiVersionCb_t ) ( ze_driver_get_api_version_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDriverGetApiVersionCb_t = Ptr{Cvoid}

struct _ze_driver_get_properties_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppDriverProperties::Ptr{Ptr{ze_driver_properties_t}}
end

const ze_driver_get_properties_params_t = _ze_driver_get_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDriverGetPropertiesCb_t ) ( ze_driver_get_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDriverGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_driver_get_ipc_properties_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppIpcProperties::Ptr{Ptr{ze_driver_ipc_properties_t}}
end

const ze_driver_get_ipc_properties_params_t = _ze_driver_get_ipc_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDriverGetIpcPropertiesCb_t ) ( ze_driver_get_ipc_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDriverGetIpcPropertiesCb_t = Ptr{Cvoid}

struct _ze_driver_get_extension_properties_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppExtensionProperties::Ptr{Ptr{ze_driver_extension_properties_t}}
end

const ze_driver_get_extension_properties_params_t = _ze_driver_get_extension_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDriverGetExtensionPropertiesCb_t ) ( ze_driver_get_extension_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDriverGetExtensionPropertiesCb_t = Ptr{Cvoid}

struct _ze_driver_callbacks_t
    pfnGetCb::ze_pfnDriverGetCb_t
    pfnGetApiVersionCb::ze_pfnDriverGetApiVersionCb_t
    pfnGetPropertiesCb::ze_pfnDriverGetPropertiesCb_t
    pfnGetIpcPropertiesCb::ze_pfnDriverGetIpcPropertiesCb_t
    pfnGetExtensionPropertiesCb::ze_pfnDriverGetExtensionPropertiesCb_t
end

const ze_driver_callbacks_t = _ze_driver_callbacks_t

struct _ze_device_get_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    pphDevices::Ptr{Ptr{ze_device_handle_t}}
end

const ze_device_get_params_t = _ze_device_get_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetCb_t ) ( ze_device_get_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetCb_t = Ptr{Cvoid}

struct _ze_device_get_sub_devices_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    pphSubdevices::Ptr{Ptr{ze_device_handle_t}}
end

const ze_device_get_sub_devices_params_t = _ze_device_get_sub_devices_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetSubDevicesCb_t ) ( ze_device_get_sub_devices_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetSubDevicesCb_t = Ptr{Cvoid}

struct _ze_device_get_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppDeviceProperties::Ptr{Ptr{ze_device_properties_t}}
end

const ze_device_get_properties_params_t = _ze_device_get_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetPropertiesCb_t ) ( ze_device_get_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_compute_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppComputeProperties::Ptr{Ptr{ze_device_compute_properties_t}}
end

const ze_device_get_compute_properties_params_t = _ze_device_get_compute_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetComputePropertiesCb_t ) ( ze_device_get_compute_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetComputePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_module_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppModuleProperties::Ptr{Ptr{ze_device_module_properties_t}}
end

const ze_device_get_module_properties_params_t = _ze_device_get_module_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetModulePropertiesCb_t ) ( ze_device_get_module_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetModulePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_command_queue_group_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppCommandQueueGroupProperties::Ptr{Ptr{ze_command_queue_group_properties_t}}
end

const ze_device_get_command_queue_group_properties_params_t = _ze_device_get_command_queue_group_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t ) ( ze_device_get_command_queue_group_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_memory_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppMemProperties::Ptr{Ptr{ze_device_memory_properties_t}}
end

const ze_device_get_memory_properties_params_t = _ze_device_get_memory_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetMemoryPropertiesCb_t ) ( ze_device_get_memory_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetMemoryPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_memory_access_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppMemAccessProperties::Ptr{Ptr{ze_device_memory_access_properties_t}}
end

const ze_device_get_memory_access_properties_params_t = _ze_device_get_memory_access_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetMemoryAccessPropertiesCb_t ) ( ze_device_get_memory_access_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetMemoryAccessPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_cache_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppCacheProperties::Ptr{Ptr{ze_device_cache_properties_t}}
end

const ze_device_get_cache_properties_params_t = _ze_device_get_cache_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetCachePropertiesCb_t ) ( ze_device_get_cache_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetCachePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_image_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppImageProperties::Ptr{Ptr{ze_device_image_properties_t}}
end

const ze_device_get_image_properties_params_t = _ze_device_get_image_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetImagePropertiesCb_t ) ( ze_device_get_image_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetImagePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_external_memory_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppExternalMemoryProperties::Ptr{Ptr{ze_device_external_memory_properties_t}}
end

const ze_device_get_external_memory_properties_params_t = _ze_device_get_external_memory_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetExternalMemoryPropertiesCb_t ) ( ze_device_get_external_memory_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetExternalMemoryPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_p2_p_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    phPeerDevice::Ptr{ze_device_handle_t}
    ppP2PProperties::Ptr{Ptr{ze_device_p2p_properties_t}}
end

const ze_device_get_p2_p_properties_params_t = _ze_device_get_p2_p_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetP2PPropertiesCb_t ) ( ze_device_get_p2_p_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetP2PPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_can_access_peer_params_t
    phDevice::Ptr{ze_device_handle_t}
    phPeerDevice::Ptr{ze_device_handle_t}
    pvalue::Ptr{Ptr{ze_bool_t}}
end

const ze_device_can_access_peer_params_t = _ze_device_can_access_peer_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceCanAccessPeerCb_t ) ( ze_device_can_access_peer_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceCanAccessPeerCb_t = Ptr{Cvoid}

struct _ze_device_get_status_params_t
    phDevice::Ptr{ze_device_handle_t}
end

const ze_device_get_status_params_t = _ze_device_get_status_params_t

# typedef void ( ZE_APICALL * ze_pfnDeviceGetStatusCb_t ) ( ze_device_get_status_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnDeviceGetStatusCb_t = Ptr{Cvoid}

struct _ze_device_callbacks_t
    pfnGetCb::ze_pfnDeviceGetCb_t
    pfnGetSubDevicesCb::ze_pfnDeviceGetSubDevicesCb_t
    pfnGetPropertiesCb::ze_pfnDeviceGetPropertiesCb_t
    pfnGetComputePropertiesCb::ze_pfnDeviceGetComputePropertiesCb_t
    pfnGetModulePropertiesCb::ze_pfnDeviceGetModulePropertiesCb_t
    pfnGetCommandQueueGroupPropertiesCb::ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t
    pfnGetMemoryPropertiesCb::ze_pfnDeviceGetMemoryPropertiesCb_t
    pfnGetMemoryAccessPropertiesCb::ze_pfnDeviceGetMemoryAccessPropertiesCb_t
    pfnGetCachePropertiesCb::ze_pfnDeviceGetCachePropertiesCb_t
    pfnGetImagePropertiesCb::ze_pfnDeviceGetImagePropertiesCb_t
    pfnGetExternalMemoryPropertiesCb::ze_pfnDeviceGetExternalMemoryPropertiesCb_t
    pfnGetP2PPropertiesCb::ze_pfnDeviceGetP2PPropertiesCb_t
    pfnCanAccessPeerCb::ze_pfnDeviceCanAccessPeerCb_t
    pfnGetStatusCb::ze_pfnDeviceGetStatusCb_t
end

const ze_device_callbacks_t = _ze_device_callbacks_t

struct _ze_context_create_params_t
    phDriver::Ptr{ze_driver_handle_t}
    pdesc::Ptr{Ptr{ze_context_desc_t}}
    pphContext::Ptr{Ptr{ze_context_handle_t}}
end

const ze_context_create_params_t = _ze_context_create_params_t

# typedef void ( ZE_APICALL * ze_pfnContextCreateCb_t ) ( ze_context_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextCreateCb_t = Ptr{Cvoid}

struct _ze_context_destroy_params_t
    phContext::Ptr{ze_context_handle_t}
end

const ze_context_destroy_params_t = _ze_context_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnContextDestroyCb_t ) ( ze_context_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextDestroyCb_t = Ptr{Cvoid}

struct _ze_context_get_status_params_t
    phContext::Ptr{ze_context_handle_t}
end

const ze_context_get_status_params_t = _ze_context_get_status_params_t

# typedef void ( ZE_APICALL * ze_pfnContextGetStatusCb_t ) ( ze_context_get_status_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextGetStatusCb_t = Ptr{Cvoid}

struct _ze_context_system_barrier_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
end

const ze_context_system_barrier_params_t = _ze_context_system_barrier_params_t

# typedef void ( ZE_APICALL * ze_pfnContextSystemBarrierCb_t ) ( ze_context_system_barrier_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextSystemBarrierCb_t = Ptr{Cvoid}

struct _ze_context_make_memory_resident_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_context_make_memory_resident_params_t = _ze_context_make_memory_resident_params_t

# typedef void ( ZE_APICALL * ze_pfnContextMakeMemoryResidentCb_t ) ( ze_context_make_memory_resident_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextMakeMemoryResidentCb_t = Ptr{Cvoid}

struct _ze_context_evict_memory_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_context_evict_memory_params_t = _ze_context_evict_memory_params_t

# typedef void ( ZE_APICALL * ze_pfnContextEvictMemoryCb_t ) ( ze_context_evict_memory_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextEvictMemoryCb_t = Ptr{Cvoid}

struct _ze_context_make_image_resident_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    phImage::Ptr{ze_image_handle_t}
end

const ze_context_make_image_resident_params_t = _ze_context_make_image_resident_params_t

# typedef void ( ZE_APICALL * ze_pfnContextMakeImageResidentCb_t ) ( ze_context_make_image_resident_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextMakeImageResidentCb_t = Ptr{Cvoid}

struct _ze_context_evict_image_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    phImage::Ptr{ze_image_handle_t}
end

const ze_context_evict_image_params_t = _ze_context_evict_image_params_t

# typedef void ( ZE_APICALL * ze_pfnContextEvictImageCb_t ) ( ze_context_evict_image_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnContextEvictImageCb_t = Ptr{Cvoid}

struct _ze_context_callbacks_t
    pfnCreateCb::ze_pfnContextCreateCb_t
    pfnDestroyCb::ze_pfnContextDestroyCb_t
    pfnGetStatusCb::ze_pfnContextGetStatusCb_t
    pfnSystemBarrierCb::ze_pfnContextSystemBarrierCb_t
    pfnMakeMemoryResidentCb::ze_pfnContextMakeMemoryResidentCb_t
    pfnEvictMemoryCb::ze_pfnContextEvictMemoryCb_t
    pfnMakeImageResidentCb::ze_pfnContextMakeImageResidentCb_t
    pfnEvictImageCb::ze_pfnContextEvictImageCb_t
end

const ze_context_callbacks_t = _ze_context_callbacks_t

struct _ze_command_queue_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_command_queue_desc_t}}
    pphCommandQueue::Ptr{Ptr{ze_command_queue_handle_t}}
end

const ze_command_queue_create_params_t = _ze_command_queue_create_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandQueueCreateCb_t ) ( ze_command_queue_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandQueueCreateCb_t = Ptr{Cvoid}

struct _ze_command_queue_destroy_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
end

const ze_command_queue_destroy_params_t = _ze_command_queue_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandQueueDestroyCb_t ) ( ze_command_queue_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandQueueDestroyCb_t = Ptr{Cvoid}

struct _ze_command_queue_execute_command_lists_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
    pnumCommandLists::Ptr{UInt32}
    pphCommandLists::Ptr{Ptr{ze_command_list_handle_t}}
    phFence::Ptr{ze_fence_handle_t}
end

const ze_command_queue_execute_command_lists_params_t = _ze_command_queue_execute_command_lists_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandQueueExecuteCommandListsCb_t ) ( ze_command_queue_execute_command_lists_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandQueueExecuteCommandListsCb_t = Ptr{Cvoid}

struct _ze_command_queue_synchronize_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
    ptimeout::Ptr{UInt64}
end

const ze_command_queue_synchronize_params_t = _ze_command_queue_synchronize_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandQueueSynchronizeCb_t ) ( ze_command_queue_synchronize_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandQueueSynchronizeCb_t = Ptr{Cvoid}

struct _ze_command_queue_callbacks_t
    pfnCreateCb::ze_pfnCommandQueueCreateCb_t
    pfnDestroyCb::ze_pfnCommandQueueDestroyCb_t
    pfnExecuteCommandListsCb::ze_pfnCommandQueueExecuteCommandListsCb_t
    pfnSynchronizeCb::ze_pfnCommandQueueSynchronizeCb_t
end

const ze_command_queue_callbacks_t = _ze_command_queue_callbacks_t

struct _ze_command_list_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_command_list_desc_t}}
    pphCommandList::Ptr{Ptr{ze_command_list_handle_t}}
end

const ze_command_list_create_params_t = _ze_command_list_create_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListCreateCb_t ) ( ze_command_list_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListCreateCb_t = Ptr{Cvoid}

struct _ze_command_list_create_immediate_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    paltdesc::Ptr{Ptr{ze_command_queue_desc_t}}
    pphCommandList::Ptr{Ptr{ze_command_list_handle_t}}
end

const ze_command_list_create_immediate_params_t = _ze_command_list_create_immediate_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListCreateImmediateCb_t ) ( ze_command_list_create_immediate_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListCreateImmediateCb_t = Ptr{Cvoid}

struct _ze_command_list_destroy_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
end

const ze_command_list_destroy_params_t = _ze_command_list_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListDestroyCb_t ) ( ze_command_list_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListDestroyCb_t = Ptr{Cvoid}

struct _ze_command_list_close_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
end

const ze_command_list_close_params_t = _ze_command_list_close_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListCloseCb_t ) ( ze_command_list_close_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListCloseCb_t = Ptr{Cvoid}

struct _ze_command_list_reset_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
end

const ze_command_list_reset_params_t = _ze_command_list_reset_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListResetCb_t ) ( ze_command_list_reset_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListResetCb_t = Ptr{Cvoid}

struct _ze_command_list_append_write_global_timestamp_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pdstptr::Ptr{Ptr{UInt64}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_write_global_timestamp_params_t = _ze_command_list_append_write_global_timestamp_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendWriteGlobalTimestampCb_t ) ( ze_command_list_append_write_global_timestamp_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendWriteGlobalTimestampCb_t = Ptr{Cvoid}

struct _ze_command_list_append_barrier_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_barrier_params_t = _ze_command_list_append_barrier_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendBarrierCb_t ) ( ze_command_list_append_barrier_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendBarrierCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_ranges_barrier_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pnumRanges::Ptr{UInt32}
    ppRangeSizes::Ptr{Ptr{Csize_t}}
    ppRanges::Ptr{Ptr{Ptr{Cvoid}}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_memory_ranges_barrier_params_t = _ze_command_list_append_memory_ranges_barrier_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemoryRangesBarrierCb_t ) ( ze_command_list_append_memory_ranges_barrier_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemoryRangesBarrierCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_copy_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pdstptr::Ptr{Ptr{Cvoid}}
    psrcptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_memory_copy_params_t = _ze_command_list_append_memory_copy_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemoryCopyCb_t ) ( ze_command_list_append_memory_copy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemoryCopyCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_fill_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppattern::Ptr{Ptr{Cvoid}}
    ppattern_size::Ptr{Csize_t}
    psize::Ptr{Csize_t}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_memory_fill_params_t = _ze_command_list_append_memory_fill_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemoryFillCb_t ) ( ze_command_list_append_memory_fill_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemoryFillCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_copy_region_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pdstptr::Ptr{Ptr{Cvoid}}
    pdstRegion::Ptr{Ptr{ze_copy_region_t}}
    pdstPitch::Ptr{UInt32}
    pdstSlicePitch::Ptr{UInt32}
    psrcptr::Ptr{Ptr{Cvoid}}
    psrcRegion::Ptr{Ptr{ze_copy_region_t}}
    psrcPitch::Ptr{UInt32}
    psrcSlicePitch::Ptr{UInt32}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_memory_copy_region_params_t = _ze_command_list_append_memory_copy_region_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemoryCopyRegionCb_t ) ( ze_command_list_append_memory_copy_region_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemoryCopyRegionCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_copy_from_context_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pdstptr::Ptr{Ptr{Cvoid}}
    phContextSrc::Ptr{ze_context_handle_t}
    psrcptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_memory_copy_from_context_params_t = _ze_command_list_append_memory_copy_from_context_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemoryCopyFromContextCb_t ) ( ze_command_list_append_memory_copy_from_context_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemoryCopyFromContextCb_t = Ptr{Cvoid}

struct _ze_command_list_append_image_copy_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phDstImage::Ptr{ze_image_handle_t}
    phSrcImage::Ptr{ze_image_handle_t}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_image_copy_params_t = _ze_command_list_append_image_copy_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendImageCopyCb_t ) ( ze_command_list_append_image_copy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendImageCopyCb_t = Ptr{Cvoid}

struct _ze_command_list_append_image_copy_region_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phDstImage::Ptr{ze_image_handle_t}
    phSrcImage::Ptr{ze_image_handle_t}
    ppDstRegion::Ptr{Ptr{ze_image_region_t}}
    ppSrcRegion::Ptr{Ptr{ze_image_region_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_image_copy_region_params_t = _ze_command_list_append_image_copy_region_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendImageCopyRegionCb_t ) ( ze_command_list_append_image_copy_region_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendImageCopyRegionCb_t = Ptr{Cvoid}

struct _ze_command_list_append_image_copy_to_memory_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pdstptr::Ptr{Ptr{Cvoid}}
    phSrcImage::Ptr{ze_image_handle_t}
    ppSrcRegion::Ptr{Ptr{ze_image_region_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_image_copy_to_memory_params_t = _ze_command_list_append_image_copy_to_memory_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendImageCopyToMemoryCb_t ) ( ze_command_list_append_image_copy_to_memory_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendImageCopyToMemoryCb_t = Ptr{Cvoid}

struct _ze_command_list_append_image_copy_from_memory_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phDstImage::Ptr{ze_image_handle_t}
    psrcptr::Ptr{Ptr{Cvoid}}
    ppDstRegion::Ptr{Ptr{ze_image_region_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_image_copy_from_memory_params_t = _ze_command_list_append_image_copy_from_memory_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendImageCopyFromMemoryCb_t ) ( ze_command_list_append_image_copy_from_memory_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendImageCopyFromMemoryCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_prefetch_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_command_list_append_memory_prefetch_params_t = _ze_command_list_append_memory_prefetch_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemoryPrefetchCb_t ) ( ze_command_list_append_memory_prefetch_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemoryPrefetchCb_t = Ptr{Cvoid}

struct _ze_command_list_append_mem_advise_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    padvice::Ptr{ze_memory_advice_t}
end

const ze_command_list_append_mem_advise_params_t = _ze_command_list_append_mem_advise_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendMemAdviseCb_t ) ( ze_command_list_append_mem_advise_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendMemAdviseCb_t = Ptr{Cvoid}

struct _ze_command_list_append_signal_event_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phEvent::Ptr{ze_event_handle_t}
end

const ze_command_list_append_signal_event_params_t = _ze_command_list_append_signal_event_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendSignalEventCb_t ) ( ze_command_list_append_signal_event_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendSignalEventCb_t = Ptr{Cvoid}

struct _ze_command_list_append_wait_on_events_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pnumEvents::Ptr{UInt32}
    pphEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_wait_on_events_params_t = _ze_command_list_append_wait_on_events_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendWaitOnEventsCb_t ) ( ze_command_list_append_wait_on_events_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendWaitOnEventsCb_t = Ptr{Cvoid}

struct _ze_command_list_append_event_reset_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phEvent::Ptr{ze_event_handle_t}
end

const ze_command_list_append_event_reset_params_t = _ze_command_list_append_event_reset_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendEventResetCb_t ) ( ze_command_list_append_event_reset_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendEventResetCb_t = Ptr{Cvoid}

struct _ze_command_list_append_query_kernel_timestamps_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pnumEvents::Ptr{UInt32}
    pphEvents::Ptr{Ptr{ze_event_handle_t}}
    pdstptr::Ptr{Ptr{Cvoid}}
    ppOffsets::Ptr{Ptr{Csize_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_query_kernel_timestamps_params_t = _ze_command_list_append_query_kernel_timestamps_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendQueryKernelTimestampsCb_t ) ( ze_command_list_append_query_kernel_timestamps_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendQueryKernelTimestampsCb_t = Ptr{Cvoid}

struct _ze_command_list_append_launch_kernel_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phKernel::Ptr{ze_kernel_handle_t}
    ppLaunchFuncArgs::Ptr{Ptr{ze_group_count_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_launch_kernel_params_t = _ze_command_list_append_launch_kernel_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendLaunchKernelCb_t ) ( ze_command_list_append_launch_kernel_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendLaunchKernelCb_t = Ptr{Cvoid}

struct _ze_command_list_append_launch_cooperative_kernel_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phKernel::Ptr{ze_kernel_handle_t}
    ppLaunchFuncArgs::Ptr{Ptr{ze_group_count_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_launch_cooperative_kernel_params_t = _ze_command_list_append_launch_cooperative_kernel_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendLaunchCooperativeKernelCb_t ) ( ze_command_list_append_launch_cooperative_kernel_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendLaunchCooperativeKernelCb_t = Ptr{Cvoid}

struct _ze_command_list_append_launch_kernel_indirect_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phKernel::Ptr{ze_kernel_handle_t}
    ppLaunchArgumentsBuffer::Ptr{Ptr{ze_group_count_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_launch_kernel_indirect_params_t = _ze_command_list_append_launch_kernel_indirect_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendLaunchKernelIndirectCb_t ) ( ze_command_list_append_launch_kernel_indirect_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendLaunchKernelIndirectCb_t = Ptr{Cvoid}

struct _ze_command_list_append_launch_multiple_kernels_indirect_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pnumKernels::Ptr{UInt32}
    pphKernels::Ptr{Ptr{ze_kernel_handle_t}}
    ppCountBuffer::Ptr{Ptr{UInt32}}
    ppLaunchArgumentsBuffer::Ptr{Ptr{ze_group_count_t}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_launch_multiple_kernels_indirect_params_t = _ze_command_list_append_launch_multiple_kernels_indirect_params_t

# typedef void ( ZE_APICALL * ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t ) ( ze_command_list_append_launch_multiple_kernels_indirect_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t = Ptr{Cvoid}

struct _ze_command_list_callbacks_t
    pfnCreateCb::ze_pfnCommandListCreateCb_t
    pfnCreateImmediateCb::ze_pfnCommandListCreateImmediateCb_t
    pfnDestroyCb::ze_pfnCommandListDestroyCb_t
    pfnCloseCb::ze_pfnCommandListCloseCb_t
    pfnResetCb::ze_pfnCommandListResetCb_t
    pfnAppendWriteGlobalTimestampCb::ze_pfnCommandListAppendWriteGlobalTimestampCb_t
    pfnAppendBarrierCb::ze_pfnCommandListAppendBarrierCb_t
    pfnAppendMemoryRangesBarrierCb::ze_pfnCommandListAppendMemoryRangesBarrierCb_t
    pfnAppendMemoryCopyCb::ze_pfnCommandListAppendMemoryCopyCb_t
    pfnAppendMemoryFillCb::ze_pfnCommandListAppendMemoryFillCb_t
    pfnAppendMemoryCopyRegionCb::ze_pfnCommandListAppendMemoryCopyRegionCb_t
    pfnAppendMemoryCopyFromContextCb::ze_pfnCommandListAppendMemoryCopyFromContextCb_t
    pfnAppendImageCopyCb::ze_pfnCommandListAppendImageCopyCb_t
    pfnAppendImageCopyRegionCb::ze_pfnCommandListAppendImageCopyRegionCb_t
    pfnAppendImageCopyToMemoryCb::ze_pfnCommandListAppendImageCopyToMemoryCb_t
    pfnAppendImageCopyFromMemoryCb::ze_pfnCommandListAppendImageCopyFromMemoryCb_t
    pfnAppendMemoryPrefetchCb::ze_pfnCommandListAppendMemoryPrefetchCb_t
    pfnAppendMemAdviseCb::ze_pfnCommandListAppendMemAdviseCb_t
    pfnAppendSignalEventCb::ze_pfnCommandListAppendSignalEventCb_t
    pfnAppendWaitOnEventsCb::ze_pfnCommandListAppendWaitOnEventsCb_t
    pfnAppendEventResetCb::ze_pfnCommandListAppendEventResetCb_t
    pfnAppendQueryKernelTimestampsCb::ze_pfnCommandListAppendQueryKernelTimestampsCb_t
    pfnAppendLaunchKernelCb::ze_pfnCommandListAppendLaunchKernelCb_t
    pfnAppendLaunchCooperativeKernelCb::ze_pfnCommandListAppendLaunchCooperativeKernelCb_t
    pfnAppendLaunchKernelIndirectCb::ze_pfnCommandListAppendLaunchKernelIndirectCb_t
    pfnAppendLaunchMultipleKernelsIndirectCb::ze_pfnCommandListAppendLaunchMultipleKernelsIndirectCb_t
end

const ze_command_list_callbacks_t = _ze_command_list_callbacks_t

struct _ze_image_get_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_image_desc_t}}
    ppImageProperties::Ptr{Ptr{ze_image_properties_t}}
end

const ze_image_get_properties_params_t = _ze_image_get_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnImageGetPropertiesCb_t ) ( ze_image_get_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnImageGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_image_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_image_desc_t}}
    pphImage::Ptr{Ptr{ze_image_handle_t}}
end

const ze_image_create_params_t = _ze_image_create_params_t

# typedef void ( ZE_APICALL * ze_pfnImageCreateCb_t ) ( ze_image_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnImageCreateCb_t = Ptr{Cvoid}

struct _ze_image_destroy_params_t
    phImage::Ptr{ze_image_handle_t}
end

const ze_image_destroy_params_t = _ze_image_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnImageDestroyCb_t ) ( ze_image_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnImageDestroyCb_t = Ptr{Cvoid}

struct _ze_image_callbacks_t
    pfnGetPropertiesCb::ze_pfnImageGetPropertiesCb_t
    pfnCreateCb::ze_pfnImageCreateCb_t
    pfnDestroyCb::ze_pfnImageDestroyCb_t
end

const ze_image_callbacks_t = _ze_image_callbacks_t

struct _ze_mem_alloc_shared_params_t
    phContext::Ptr{ze_context_handle_t}
    pdevice_desc::Ptr{Ptr{ze_device_mem_alloc_desc_t}}
    phost_desc::Ptr{Ptr{ze_host_mem_alloc_desc_t}}
    psize::Ptr{Csize_t}
    palignment::Ptr{Csize_t}
    phDevice::Ptr{ze_device_handle_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_mem_alloc_shared_params_t = _ze_mem_alloc_shared_params_t

# typedef void ( ZE_APICALL * ze_pfnMemAllocSharedCb_t ) ( ze_mem_alloc_shared_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemAllocSharedCb_t = Ptr{Cvoid}

struct _ze_mem_alloc_device_params_t
    phContext::Ptr{ze_context_handle_t}
    pdevice_desc::Ptr{Ptr{ze_device_mem_alloc_desc_t}}
    psize::Ptr{Csize_t}
    palignment::Ptr{Csize_t}
    phDevice::Ptr{ze_device_handle_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_mem_alloc_device_params_t = _ze_mem_alloc_device_params_t

# typedef void ( ZE_APICALL * ze_pfnMemAllocDeviceCb_t ) ( ze_mem_alloc_device_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemAllocDeviceCb_t = Ptr{Cvoid}

struct _ze_mem_alloc_host_params_t
    phContext::Ptr{ze_context_handle_t}
    phost_desc::Ptr{Ptr{ze_host_mem_alloc_desc_t}}
    psize::Ptr{Csize_t}
    palignment::Ptr{Csize_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_mem_alloc_host_params_t = _ze_mem_alloc_host_params_t

# typedef void ( ZE_APICALL * ze_pfnMemAllocHostCb_t ) ( ze_mem_alloc_host_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemAllocHostCb_t = Ptr{Cvoid}

struct _ze_mem_free_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
end

const ze_mem_free_params_t = _ze_mem_free_params_t

# typedef void ( ZE_APICALL * ze_pfnMemFreeCb_t ) ( ze_mem_free_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemFreeCb_t = Ptr{Cvoid}

struct _ze_mem_get_alloc_properties_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppMemAllocProperties::Ptr{Ptr{ze_memory_allocation_properties_t}}
    pphDevice::Ptr{Ptr{ze_device_handle_t}}
end

const ze_mem_get_alloc_properties_params_t = _ze_mem_get_alloc_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnMemGetAllocPropertiesCb_t ) ( ze_mem_get_alloc_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemGetAllocPropertiesCb_t = Ptr{Cvoid}

struct _ze_mem_get_address_range_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppBase::Ptr{Ptr{Ptr{Cvoid}}}
    ppSize::Ptr{Ptr{Csize_t}}
end

const ze_mem_get_address_range_params_t = _ze_mem_get_address_range_params_t

# typedef void ( ZE_APICALL * ze_pfnMemGetAddressRangeCb_t ) ( ze_mem_get_address_range_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemGetAddressRangeCb_t = Ptr{Cvoid}

struct _ze_mem_get_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppIpcHandle::Ptr{Ptr{ze_ipc_mem_handle_t}}
end

const ze_mem_get_ipc_handle_params_t = _ze_mem_get_ipc_handle_params_t

# typedef void ( ZE_APICALL * ze_pfnMemGetIpcHandleCb_t ) ( ze_mem_get_ipc_handle_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemGetIpcHandleCb_t = Ptr{Cvoid}

struct _ze_mem_open_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    phandle::Ptr{ze_ipc_mem_handle_t}
    pflags::Ptr{ze_ipc_memory_flags_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_mem_open_ipc_handle_params_t = _ze_mem_open_ipc_handle_params_t

# typedef void ( ZE_APICALL * ze_pfnMemOpenIpcHandleCb_t ) ( ze_mem_open_ipc_handle_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemOpenIpcHandleCb_t = Ptr{Cvoid}

struct _ze_mem_close_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
end

const ze_mem_close_ipc_handle_params_t = _ze_mem_close_ipc_handle_params_t

# typedef void ( ZE_APICALL * ze_pfnMemCloseIpcHandleCb_t ) ( ze_mem_close_ipc_handle_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnMemCloseIpcHandleCb_t = Ptr{Cvoid}

struct _ze_mem_callbacks_t
    pfnAllocSharedCb::ze_pfnMemAllocSharedCb_t
    pfnAllocDeviceCb::ze_pfnMemAllocDeviceCb_t
    pfnAllocHostCb::ze_pfnMemAllocHostCb_t
    pfnFreeCb::ze_pfnMemFreeCb_t
    pfnGetAllocPropertiesCb::ze_pfnMemGetAllocPropertiesCb_t
    pfnGetAddressRangeCb::ze_pfnMemGetAddressRangeCb_t
    pfnGetIpcHandleCb::ze_pfnMemGetIpcHandleCb_t
    pfnOpenIpcHandleCb::ze_pfnMemOpenIpcHandleCb_t
    pfnCloseIpcHandleCb::ze_pfnMemCloseIpcHandleCb_t
end

const ze_mem_callbacks_t = _ze_mem_callbacks_t

struct _ze_fence_create_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
    pdesc::Ptr{Ptr{ze_fence_desc_t}}
    pphFence::Ptr{Ptr{ze_fence_handle_t}}
end

const ze_fence_create_params_t = _ze_fence_create_params_t

# typedef void ( ZE_APICALL * ze_pfnFenceCreateCb_t ) ( ze_fence_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnFenceCreateCb_t = Ptr{Cvoid}

struct _ze_fence_destroy_params_t
    phFence::Ptr{ze_fence_handle_t}
end

const ze_fence_destroy_params_t = _ze_fence_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnFenceDestroyCb_t ) ( ze_fence_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnFenceDestroyCb_t = Ptr{Cvoid}

struct _ze_fence_host_synchronize_params_t
    phFence::Ptr{ze_fence_handle_t}
    ptimeout::Ptr{UInt64}
end

const ze_fence_host_synchronize_params_t = _ze_fence_host_synchronize_params_t

# typedef void ( ZE_APICALL * ze_pfnFenceHostSynchronizeCb_t ) ( ze_fence_host_synchronize_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnFenceHostSynchronizeCb_t = Ptr{Cvoid}

struct _ze_fence_query_status_params_t
    phFence::Ptr{ze_fence_handle_t}
end

const ze_fence_query_status_params_t = _ze_fence_query_status_params_t

# typedef void ( ZE_APICALL * ze_pfnFenceQueryStatusCb_t ) ( ze_fence_query_status_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnFenceQueryStatusCb_t = Ptr{Cvoid}

struct _ze_fence_reset_params_t
    phFence::Ptr{ze_fence_handle_t}
end

const ze_fence_reset_params_t = _ze_fence_reset_params_t

# typedef void ( ZE_APICALL * ze_pfnFenceResetCb_t ) ( ze_fence_reset_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnFenceResetCb_t = Ptr{Cvoid}

struct _ze_fence_callbacks_t
    pfnCreateCb::ze_pfnFenceCreateCb_t
    pfnDestroyCb::ze_pfnFenceDestroyCb_t
    pfnHostSynchronizeCb::ze_pfnFenceHostSynchronizeCb_t
    pfnQueryStatusCb::ze_pfnFenceQueryStatusCb_t
    pfnResetCb::ze_pfnFenceResetCb_t
end

const ze_fence_callbacks_t = _ze_fence_callbacks_t

struct _ze_event_pool_create_params_t
    phContext::Ptr{ze_context_handle_t}
    pdesc::Ptr{Ptr{ze_event_pool_desc_t}}
    pnumDevices::Ptr{UInt32}
    pphDevices::Ptr{Ptr{ze_device_handle_t}}
    pphEventPool::Ptr{Ptr{ze_event_pool_handle_t}}
end

const ze_event_pool_create_params_t = _ze_event_pool_create_params_t

# typedef void ( ZE_APICALL * ze_pfnEventPoolCreateCb_t ) ( ze_event_pool_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventPoolCreateCb_t = Ptr{Cvoid}

struct _ze_event_pool_destroy_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
end

const ze_event_pool_destroy_params_t = _ze_event_pool_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnEventPoolDestroyCb_t ) ( ze_event_pool_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventPoolDestroyCb_t = Ptr{Cvoid}

struct _ze_event_pool_get_ipc_handle_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
    pphIpc::Ptr{Ptr{ze_ipc_event_pool_handle_t}}
end

const ze_event_pool_get_ipc_handle_params_t = _ze_event_pool_get_ipc_handle_params_t

# typedef void ( ZE_APICALL * ze_pfnEventPoolGetIpcHandleCb_t ) ( ze_event_pool_get_ipc_handle_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventPoolGetIpcHandleCb_t = Ptr{Cvoid}

struct _ze_event_pool_open_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    phIpc::Ptr{ze_ipc_event_pool_handle_t}
    pphEventPool::Ptr{Ptr{ze_event_pool_handle_t}}
end

const ze_event_pool_open_ipc_handle_params_t = _ze_event_pool_open_ipc_handle_params_t

# typedef void ( ZE_APICALL * ze_pfnEventPoolOpenIpcHandleCb_t ) ( ze_event_pool_open_ipc_handle_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventPoolOpenIpcHandleCb_t = Ptr{Cvoid}

struct _ze_event_pool_close_ipc_handle_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
end

const ze_event_pool_close_ipc_handle_params_t = _ze_event_pool_close_ipc_handle_params_t

# typedef void ( ZE_APICALL * ze_pfnEventPoolCloseIpcHandleCb_t ) ( ze_event_pool_close_ipc_handle_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventPoolCloseIpcHandleCb_t = Ptr{Cvoid}

struct _ze_event_pool_callbacks_t
    pfnCreateCb::ze_pfnEventPoolCreateCb_t
    pfnDestroyCb::ze_pfnEventPoolDestroyCb_t
    pfnGetIpcHandleCb::ze_pfnEventPoolGetIpcHandleCb_t
    pfnOpenIpcHandleCb::ze_pfnEventPoolOpenIpcHandleCb_t
    pfnCloseIpcHandleCb::ze_pfnEventPoolCloseIpcHandleCb_t
end

const ze_event_pool_callbacks_t = _ze_event_pool_callbacks_t

struct _ze_event_create_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
    pdesc::Ptr{Ptr{ze_event_desc_t}}
    pphEvent::Ptr{Ptr{ze_event_handle_t}}
end

const ze_event_create_params_t = _ze_event_create_params_t

# typedef void ( ZE_APICALL * ze_pfnEventCreateCb_t ) ( ze_event_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventCreateCb_t = Ptr{Cvoid}

struct _ze_event_destroy_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_destroy_params_t = _ze_event_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnEventDestroyCb_t ) ( ze_event_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventDestroyCb_t = Ptr{Cvoid}

struct _ze_event_host_signal_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_host_signal_params_t = _ze_event_host_signal_params_t

# typedef void ( ZE_APICALL * ze_pfnEventHostSignalCb_t ) ( ze_event_host_signal_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventHostSignalCb_t = Ptr{Cvoid}

struct _ze_event_host_synchronize_params_t
    phEvent::Ptr{ze_event_handle_t}
    ptimeout::Ptr{UInt64}
end

const ze_event_host_synchronize_params_t = _ze_event_host_synchronize_params_t

# typedef void ( ZE_APICALL * ze_pfnEventHostSynchronizeCb_t ) ( ze_event_host_synchronize_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventHostSynchronizeCb_t = Ptr{Cvoid}

struct _ze_event_query_status_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_query_status_params_t = _ze_event_query_status_params_t

# typedef void ( ZE_APICALL * ze_pfnEventQueryStatusCb_t ) ( ze_event_query_status_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventQueryStatusCb_t = Ptr{Cvoid}

struct _ze_event_host_reset_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_host_reset_params_t = _ze_event_host_reset_params_t

# typedef void ( ZE_APICALL * ze_pfnEventHostResetCb_t ) ( ze_event_host_reset_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventHostResetCb_t = Ptr{Cvoid}

struct _ze_event_query_kernel_timestamp_params_t
    phEvent::Ptr{ze_event_handle_t}
    pdstptr::Ptr{Ptr{ze_kernel_timestamp_result_t}}
end

const ze_event_query_kernel_timestamp_params_t = _ze_event_query_kernel_timestamp_params_t

# typedef void ( ZE_APICALL * ze_pfnEventQueryKernelTimestampCb_t ) ( ze_event_query_kernel_timestamp_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnEventQueryKernelTimestampCb_t = Ptr{Cvoid}

struct _ze_event_callbacks_t
    pfnCreateCb::ze_pfnEventCreateCb_t
    pfnDestroyCb::ze_pfnEventDestroyCb_t
    pfnHostSignalCb::ze_pfnEventHostSignalCb_t
    pfnHostSynchronizeCb::ze_pfnEventHostSynchronizeCb_t
    pfnQueryStatusCb::ze_pfnEventQueryStatusCb_t
    pfnHostResetCb::ze_pfnEventHostResetCb_t
    pfnQueryKernelTimestampCb::ze_pfnEventQueryKernelTimestampCb_t
end

const ze_event_callbacks_t = _ze_event_callbacks_t

struct _ze_module_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_module_desc_t}}
    pphModule::Ptr{Ptr{ze_module_handle_t}}
    pphBuildLog::Ptr{Ptr{ze_module_build_log_handle_t}}
end

const ze_module_create_params_t = _ze_module_create_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleCreateCb_t ) ( ze_module_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleCreateCb_t = Ptr{Cvoid}

struct _ze_module_destroy_params_t
    phModule::Ptr{ze_module_handle_t}
end

const ze_module_destroy_params_t = _ze_module_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleDestroyCb_t ) ( ze_module_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleDestroyCb_t = Ptr{Cvoid}

struct _ze_module_dynamic_link_params_t
    pnumModules::Ptr{UInt32}
    pphModules::Ptr{Ptr{ze_module_handle_t}}
    pphLinkLog::Ptr{Ptr{ze_module_build_log_handle_t}}
end

const ze_module_dynamic_link_params_t = _ze_module_dynamic_link_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleDynamicLinkCb_t ) ( ze_module_dynamic_link_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleDynamicLinkCb_t = Ptr{Cvoid}

struct _ze_module_get_native_binary_params_t
    phModule::Ptr{ze_module_handle_t}
    ppSize::Ptr{Ptr{Csize_t}}
    ppModuleNativeBinary::Ptr{Ptr{UInt8}}
end

const ze_module_get_native_binary_params_t = _ze_module_get_native_binary_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleGetNativeBinaryCb_t ) ( ze_module_get_native_binary_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleGetNativeBinaryCb_t = Ptr{Cvoid}

struct _ze_module_get_global_pointer_params_t
    phModule::Ptr{ze_module_handle_t}
    ppGlobalName::Ptr{Ptr{Cchar}}
    ppSize::Ptr{Ptr{Csize_t}}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_module_get_global_pointer_params_t = _ze_module_get_global_pointer_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleGetGlobalPointerCb_t ) ( ze_module_get_global_pointer_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleGetGlobalPointerCb_t = Ptr{Cvoid}

struct _ze_module_get_kernel_names_params_t
    phModule::Ptr{ze_module_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppNames::Ptr{Ptr{Ptr{Cchar}}}
end

const ze_module_get_kernel_names_params_t = _ze_module_get_kernel_names_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleGetKernelNamesCb_t ) ( ze_module_get_kernel_names_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleGetKernelNamesCb_t = Ptr{Cvoid}

struct _ze_module_get_properties_params_t
    phModule::Ptr{ze_module_handle_t}
    ppModuleProperties::Ptr{Ptr{ze_module_properties_t}}
end

const ze_module_get_properties_params_t = _ze_module_get_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleGetPropertiesCb_t ) ( ze_module_get_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_module_get_function_pointer_params_t
    phModule::Ptr{ze_module_handle_t}
    ppFunctionName::Ptr{Ptr{Cchar}}
    ppfnFunction::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_module_get_function_pointer_params_t = _ze_module_get_function_pointer_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleGetFunctionPointerCb_t ) ( ze_module_get_function_pointer_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleGetFunctionPointerCb_t = Ptr{Cvoid}

struct _ze_module_callbacks_t
    pfnCreateCb::ze_pfnModuleCreateCb_t
    pfnDestroyCb::ze_pfnModuleDestroyCb_t
    pfnDynamicLinkCb::ze_pfnModuleDynamicLinkCb_t
    pfnGetNativeBinaryCb::ze_pfnModuleGetNativeBinaryCb_t
    pfnGetGlobalPointerCb::ze_pfnModuleGetGlobalPointerCb_t
    pfnGetKernelNamesCb::ze_pfnModuleGetKernelNamesCb_t
    pfnGetPropertiesCb::ze_pfnModuleGetPropertiesCb_t
    pfnGetFunctionPointerCb::ze_pfnModuleGetFunctionPointerCb_t
end

const ze_module_callbacks_t = _ze_module_callbacks_t

struct _ze_module_build_log_destroy_params_t
    phModuleBuildLog::Ptr{ze_module_build_log_handle_t}
end

const ze_module_build_log_destroy_params_t = _ze_module_build_log_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleBuildLogDestroyCb_t ) ( ze_module_build_log_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleBuildLogDestroyCb_t = Ptr{Cvoid}

struct _ze_module_build_log_get_string_params_t
    phModuleBuildLog::Ptr{ze_module_build_log_handle_t}
    ppSize::Ptr{Ptr{Csize_t}}
    ppBuildLog::Ptr{Ptr{Cchar}}
end

const ze_module_build_log_get_string_params_t = _ze_module_build_log_get_string_params_t

# typedef void ( ZE_APICALL * ze_pfnModuleBuildLogGetStringCb_t ) ( ze_module_build_log_get_string_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnModuleBuildLogGetStringCb_t = Ptr{Cvoid}

struct _ze_module_build_log_callbacks_t
    pfnDestroyCb::ze_pfnModuleBuildLogDestroyCb_t
    pfnGetStringCb::ze_pfnModuleBuildLogGetStringCb_t
end

const ze_module_build_log_callbacks_t = _ze_module_build_log_callbacks_t

struct _ze_kernel_create_params_t
    phModule::Ptr{ze_module_handle_t}
    pdesc::Ptr{Ptr{ze_kernel_desc_t}}
    pphKernel::Ptr{Ptr{ze_kernel_handle_t}}
end

const ze_kernel_create_params_t = _ze_kernel_create_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelCreateCb_t ) ( ze_kernel_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelCreateCb_t = Ptr{Cvoid}

struct _ze_kernel_destroy_params_t
    phKernel::Ptr{ze_kernel_handle_t}
end

const ze_kernel_destroy_params_t = _ze_kernel_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelDestroyCb_t ) ( ze_kernel_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelDestroyCb_t = Ptr{Cvoid}

struct _ze_kernel_set_cache_config_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pflags::Ptr{ze_cache_config_flags_t}
end

const ze_kernel_set_cache_config_params_t = _ze_kernel_set_cache_config_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelSetCacheConfigCb_t ) ( ze_kernel_set_cache_config_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelSetCacheConfigCb_t = Ptr{Cvoid}

struct _ze_kernel_set_group_size_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pgroupSizeX::Ptr{UInt32}
    pgroupSizeY::Ptr{UInt32}
    pgroupSizeZ::Ptr{UInt32}
end

const ze_kernel_set_group_size_params_t = _ze_kernel_set_group_size_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelSetGroupSizeCb_t ) ( ze_kernel_set_group_size_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelSetGroupSizeCb_t = Ptr{Cvoid}

struct _ze_kernel_suggest_group_size_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pglobalSizeX::Ptr{UInt32}
    pglobalSizeY::Ptr{UInt32}
    pglobalSizeZ::Ptr{UInt32}
    pgroupSizeX::Ptr{Ptr{UInt32}}
    pgroupSizeY::Ptr{Ptr{UInt32}}
    pgroupSizeZ::Ptr{Ptr{UInt32}}
end

const ze_kernel_suggest_group_size_params_t = _ze_kernel_suggest_group_size_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelSuggestGroupSizeCb_t ) ( ze_kernel_suggest_group_size_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelSuggestGroupSizeCb_t = Ptr{Cvoid}

struct _ze_kernel_suggest_max_cooperative_group_count_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ptotalGroupCount::Ptr{Ptr{UInt32}}
end

const ze_kernel_suggest_max_cooperative_group_count_params_t = _ze_kernel_suggest_max_cooperative_group_count_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t ) ( ze_kernel_suggest_max_cooperative_group_count_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t = Ptr{Cvoid}

struct _ze_kernel_set_argument_value_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pargIndex::Ptr{UInt32}
    pargSize::Ptr{Csize_t}
    ppArgValue::Ptr{Ptr{Cvoid}}
end

const ze_kernel_set_argument_value_params_t = _ze_kernel_set_argument_value_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelSetArgumentValueCb_t ) ( ze_kernel_set_argument_value_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelSetArgumentValueCb_t = Ptr{Cvoid}

struct _ze_kernel_set_indirect_access_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pflags::Ptr{ze_kernel_indirect_access_flags_t}
end

const ze_kernel_set_indirect_access_params_t = _ze_kernel_set_indirect_access_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelSetIndirectAccessCb_t ) ( ze_kernel_set_indirect_access_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelSetIndirectAccessCb_t = Ptr{Cvoid}

struct _ze_kernel_get_indirect_access_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppFlags::Ptr{Ptr{ze_kernel_indirect_access_flags_t}}
end

const ze_kernel_get_indirect_access_params_t = _ze_kernel_get_indirect_access_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelGetIndirectAccessCb_t ) ( ze_kernel_get_indirect_access_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelGetIndirectAccessCb_t = Ptr{Cvoid}

struct _ze_kernel_get_source_attributes_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppSize::Ptr{Ptr{UInt32}}
    ppString::Ptr{Ptr{Ptr{Cchar}}}
end

const ze_kernel_get_source_attributes_params_t = _ze_kernel_get_source_attributes_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelGetSourceAttributesCb_t ) ( ze_kernel_get_source_attributes_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelGetSourceAttributesCb_t = Ptr{Cvoid}

struct _ze_kernel_get_properties_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppKernelProperties::Ptr{Ptr{ze_kernel_properties_t}}
end

const ze_kernel_get_properties_params_t = _ze_kernel_get_properties_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelGetPropertiesCb_t ) ( ze_kernel_get_properties_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_kernel_get_name_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppSize::Ptr{Ptr{Csize_t}}
    ppName::Ptr{Ptr{Cchar}}
end

const ze_kernel_get_name_params_t = _ze_kernel_get_name_params_t

# typedef void ( ZE_APICALL * ze_pfnKernelGetNameCb_t ) ( ze_kernel_get_name_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnKernelGetNameCb_t = Ptr{Cvoid}

struct _ze_kernel_callbacks_t
    pfnCreateCb::ze_pfnKernelCreateCb_t
    pfnDestroyCb::ze_pfnKernelDestroyCb_t
    pfnSetCacheConfigCb::ze_pfnKernelSetCacheConfigCb_t
    pfnSetGroupSizeCb::ze_pfnKernelSetGroupSizeCb_t
    pfnSuggestGroupSizeCb::ze_pfnKernelSuggestGroupSizeCb_t
    pfnSuggestMaxCooperativeGroupCountCb::ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t
    pfnSetArgumentValueCb::ze_pfnKernelSetArgumentValueCb_t
    pfnSetIndirectAccessCb::ze_pfnKernelSetIndirectAccessCb_t
    pfnGetIndirectAccessCb::ze_pfnKernelGetIndirectAccessCb_t
    pfnGetSourceAttributesCb::ze_pfnKernelGetSourceAttributesCb_t
    pfnGetPropertiesCb::ze_pfnKernelGetPropertiesCb_t
    pfnGetNameCb::ze_pfnKernelGetNameCb_t
end

const ze_kernel_callbacks_t = _ze_kernel_callbacks_t

struct _ze_sampler_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_sampler_desc_t}}
    pphSampler::Ptr{Ptr{ze_sampler_handle_t}}
end

const ze_sampler_create_params_t = _ze_sampler_create_params_t

# typedef void ( ZE_APICALL * ze_pfnSamplerCreateCb_t ) ( ze_sampler_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnSamplerCreateCb_t = Ptr{Cvoid}

struct _ze_sampler_destroy_params_t
    phSampler::Ptr{ze_sampler_handle_t}
end

const ze_sampler_destroy_params_t = _ze_sampler_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnSamplerDestroyCb_t ) ( ze_sampler_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnSamplerDestroyCb_t = Ptr{Cvoid}

struct _ze_sampler_callbacks_t
    pfnCreateCb::ze_pfnSamplerCreateCb_t
    pfnDestroyCb::ze_pfnSamplerDestroyCb_t
end

const ze_sampler_callbacks_t = _ze_sampler_callbacks_t

struct _ze_physical_mem_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_physical_mem_desc_t}}
    pphPhysicalMemory::Ptr{Ptr{ze_physical_mem_handle_t}}
end

const ze_physical_mem_create_params_t = _ze_physical_mem_create_params_t

# typedef void ( ZE_APICALL * ze_pfnPhysicalMemCreateCb_t ) ( ze_physical_mem_create_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnPhysicalMemCreateCb_t = Ptr{Cvoid}

struct _ze_physical_mem_destroy_params_t
    phContext::Ptr{ze_context_handle_t}
    phPhysicalMemory::Ptr{ze_physical_mem_handle_t}
end

const ze_physical_mem_destroy_params_t = _ze_physical_mem_destroy_params_t

# typedef void ( ZE_APICALL * ze_pfnPhysicalMemDestroyCb_t ) ( ze_physical_mem_destroy_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnPhysicalMemDestroyCb_t = Ptr{Cvoid}

struct _ze_physical_mem_callbacks_t
    pfnCreateCb::ze_pfnPhysicalMemCreateCb_t
    pfnDestroyCb::ze_pfnPhysicalMemDestroyCb_t
end

const ze_physical_mem_callbacks_t = _ze_physical_mem_callbacks_t

struct _ze_virtual_mem_reserve_params_t
    phContext::Ptr{ze_context_handle_t}
    ppStart::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_virtual_mem_reserve_params_t = _ze_virtual_mem_reserve_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemReserveCb_t ) ( ze_virtual_mem_reserve_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemReserveCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_free_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_virtual_mem_free_params_t = _ze_virtual_mem_free_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemFreeCb_t ) ( ze_virtual_mem_free_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemFreeCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_query_page_size_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    psize::Ptr{Csize_t}
    ppagesize::Ptr{Ptr{Csize_t}}
end

const ze_virtual_mem_query_page_size_params_t = _ze_virtual_mem_query_page_size_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemQueryPageSizeCb_t ) ( ze_virtual_mem_query_page_size_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemQueryPageSizeCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_map_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    phPhysicalMemory::Ptr{ze_physical_mem_handle_t}
    poffset::Ptr{Csize_t}
    paccess::Ptr{ze_memory_access_attribute_t}
end

const ze_virtual_mem_map_params_t = _ze_virtual_mem_map_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemMapCb_t ) ( ze_virtual_mem_map_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemMapCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_unmap_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_virtual_mem_unmap_params_t = _ze_virtual_mem_unmap_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemUnmapCb_t ) ( ze_virtual_mem_unmap_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemUnmapCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_set_access_attribute_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    paccess::Ptr{ze_memory_access_attribute_t}
end

const ze_virtual_mem_set_access_attribute_params_t = _ze_virtual_mem_set_access_attribute_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemSetAccessAttributeCb_t ) ( ze_virtual_mem_set_access_attribute_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemSetAccessAttributeCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_get_access_attribute_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    paccess::Ptr{Ptr{ze_memory_access_attribute_t}}
    poutSize::Ptr{Ptr{Csize_t}}
end

const ze_virtual_mem_get_access_attribute_params_t = _ze_virtual_mem_get_access_attribute_params_t

# typedef void ( ZE_APICALL * ze_pfnVirtualMemGetAccessAttributeCb_t ) ( ze_virtual_mem_get_access_attribute_params_t * params , ze_result_t result , void * pTracerUserData , void * * ppTracerInstanceUserData )
const ze_pfnVirtualMemGetAccessAttributeCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_callbacks_t
    pfnReserveCb::ze_pfnVirtualMemReserveCb_t
    pfnFreeCb::ze_pfnVirtualMemFreeCb_t
    pfnQueryPageSizeCb::ze_pfnVirtualMemQueryPageSizeCb_t
    pfnMapCb::ze_pfnVirtualMemMapCb_t
    pfnUnmapCb::ze_pfnVirtualMemUnmapCb_t
    pfnSetAccessAttributeCb::ze_pfnVirtualMemSetAccessAttributeCb_t
    pfnGetAccessAttributeCb::ze_pfnVirtualMemGetAccessAttributeCb_t
end

const ze_virtual_mem_callbacks_t = _ze_virtual_mem_callbacks_t

struct _ze_callbacks_t
    Global::ze_global_callbacks_t
    Driver::ze_driver_callbacks_t
    Device::ze_device_callbacks_t
    Context::ze_context_callbacks_t
    CommandQueue::ze_command_queue_callbacks_t
    CommandList::ze_command_list_callbacks_t
    Fence::ze_fence_callbacks_t
    EventPool::ze_event_pool_callbacks_t
    Event::ze_event_callbacks_t
    Image::ze_image_callbacks_t
    Module::ze_module_callbacks_t
    ModuleBuildLog::ze_module_build_log_callbacks_t
    Kernel::ze_kernel_callbacks_t
    Sampler::ze_sampler_callbacks_t
    PhysicalMem::ze_physical_mem_callbacks_t
    Mem::ze_mem_callbacks_t
    VirtualMem::ze_virtual_mem_callbacks_t
end

const ze_callbacks_t = _ze_callbacks_t

# Skipping MacroDefinition: ZE_APIEXPORT __attribute__ ( ( visibility ( "default" ) ) )

# Skipping MacroDefinition: ZE_DLLEXPORT __attribute__ ( ( visibility ( "default" ) ) )

const ZE_MAX_IPC_HANDLE_SIZE = 64

const ZE_MAX_UUID_SIZE = 16

const ZE_API_VERSION_CURRENT_M = ZE_MAKE_VERSION(1, 13)

const ZE_MAX_DRIVER_UUID_SIZE = 16

const ZE_MAX_EXTENSION_NAME = 256

const ZE_MAX_DEVICE_UUID_SIZE = 16

const ZE_MAX_DEVICE_NAME = 256

const ZE_SUBGROUPSIZE_COUNT = 8

const ZE_MAX_NATIVE_KERNEL_UUID_SIZE = 16

const ZE_MAX_KERNEL_UUID_SIZE = 16

const ZE_MAX_MODULE_UUID_SIZE = 16

const ZE_MODULE_PROGRAM_EXP_NAME = "ZE_experimental_module_program"

const ZE_RAYTRACING_EXT_NAME = "ZE_extension_raytracing"

const ZE_FLOAT_ATOMICS_EXT_NAME = "ZE_extension_float_atomics"

const ZE_GLOBAL_OFFSET_EXP_NAME = "ZE_experimental_global_offset"

const ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME = "ZE_experimental_relaxed_allocation_limits"

const ZE_GET_KERNEL_BINARY_EXP_NAME = "ZE_extension_kernel_binary_exp"

const ZE_DRIVER_DDI_HANDLES_EXT_NAME = "ZE_extension_driver_ddi_handles"

const ZE_EXTERNAL_SEMAPHORES_EXTENSION_NAME = "ZE_extension_external_semaphores"

const ZE_CACHELINE_SIZE_EXT_NAME = "ZE_extension_device_cache_line_size"

const ZE_RTAS_EXT_NAME = "ZE_extension_rtas"

const ZE_DEVICE_VECTOR_SIZES_EXT_NAME = "ZE_extension_device_vector_sizes"

const ZE_CACHE_RESERVATION_EXT_NAME = "ZE_extension_cache_reservation"

const ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME = "ZE_experimental_event_query_timestamps"

const ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME = "ZE_experimental_image_memory_properties"

const ZE_IMAGE_VIEW_EXT_NAME = "ZE_extension_image_view"

const ZE_IMAGE_VIEW_EXP_NAME = "ZE_experimental_image_view"

const ZE_IMAGE_VIEW_PLANAR_EXT_NAME = "ZE_extension_image_view_planar"

const ZE_IMAGE_VIEW_PLANAR_EXP_NAME = "ZE_experimental_image_view_planar"

const ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME = "ZE_experimental_scheduling_hints"

const ZE_LINKONCE_ODR_EXT_NAME = "ZE_extension_linkonce_odr"

const ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME = "ZE_experimental_power_saving_hint"

const ZE_SUBGROUPS_EXT_NAME = "ZE_extension_subgroups"

const ZE_EU_COUNT_EXT_NAME = "ZE_extension_eu_count"

const ZE_PCI_PROPERTIES_EXT_NAME = "ZE_extension_pci_properties"

const ZE_SRGB_EXT_NAME = "ZE_extension_srgb"

const ZE_IMAGE_COPY_EXT_NAME = "ZE_extension_image_copy"

const ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME = "ZE_extension_image_query_alloc_properties"

const ZE_LINKAGE_INSPECTION_EXT_NAME = "ZE_extension_linkage_inspection"

const ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME = "ZE_extension_memory_compression_hints"

const ZE_MEMORY_FREE_POLICIES_EXT_NAME = "ZE_extension_memory_free_policies"

const ZE_BANDWIDTH_PROPERTIES_EXP_NAME = "ZE_experimental_bandwidth_properties"

const ZE_DEVICE_LUID_EXT_NAME = "ZE_extension_device_luid"

const ZE_MAX_DEVICE_LUID_SIZE_EXT = 8

const ZE_FABRIC_EXP_NAME = "ZE_experimental_fabric"

const ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE = 256

const ZE_DEVICE_MEMORY_PROPERTIES_EXT_NAME = "ZE_extension_device_memory_properties"

const ZE_BFLOAT16_CONVERSIONS_EXT_NAME = "ZE_extension_bfloat16_conversions"

const ZE_DEVICE_IP_VERSION_EXT_NAME = "ZE_extension_device_ip_version"

const ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_NAME = "ZE_extension_kernel_max_group_size_properties"

const ZE_SUB_ALLOCATIONS_EXP_NAME = "ZE_experimental_sub_allocations"

const ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_NAME = "ZE_extension_event_query_kernel_timestamps"

const ZE_RTAS_BUILDER_EXP_NAME = "ZE_experimental_rtas_builder"

const ZE_EVENT_POOL_COUNTER_BASED_EXP_NAME = "ZE_experimental_event_pool_counter_based"

const ZE_BINDLESS_IMAGE_EXP_NAME = "ZE_experimental_bindless_image"

const ZE_COMMAND_LIST_CLONE_EXP_NAME = "ZE_experimental_command_list_clone"

const ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_NAME = "ZE_experimental_immediate_command_list_append"

const ZE_MUTABLE_COMMAND_LIST_EXP_NAME = "ZE_experimental_mutable_command_list"
