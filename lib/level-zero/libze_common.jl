# Automatically generated using Clang.jl

# Skipping MacroDefinition: ZE_MAKE_VERSION ( _major , _minor ) ( ( _major << 16 ) | ( _minor & 0x0000ffff ) )
# Skipping MacroDefinition: ZE_MAJOR_VERSION ( _ver ) ( _ver >> 16 )
# Skipping MacroDefinition: ZE_MINOR_VERSION ( _ver ) ( _ver & 0x0000ffff )
# Skipping MacroDefinition: ZE_DLLEXPORT __attribute__ ( ( visibility ( "default" ) ) )

const ZE_MAX_IPC_HANDLE_SIZE = 64

# Skipping MacroDefinition: ZE_BIT ( _i ) ( 1 << _i )

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
const ze_bool_t = UInt8
const _ze_driver_handle_t = Cvoid
const ze_driver_handle_t = Ptr{_ze_driver_handle_t}
const _ze_device_handle_t = Cvoid
const ze_device_handle_t = Ptr{_ze_device_handle_t}
const _ze_context_handle_t = Cvoid
const ze_context_handle_t = Ptr{_ze_context_handle_t}
const _ze_command_queue_handle_t = Cvoid
const ze_command_queue_handle_t = Ptr{_ze_command_queue_handle_t}
const _ze_command_list_handle_t = Cvoid
const ze_command_list_handle_t = Ptr{_ze_command_list_handle_t}
const _ze_fence_handle_t = Cvoid
const ze_fence_handle_t = Ptr{_ze_fence_handle_t}
const _ze_event_pool_handle_t = Cvoid
const ze_event_pool_handle_t = Ptr{_ze_event_pool_handle_t}
const _ze_event_handle_t = Cvoid
const ze_event_handle_t = Ptr{_ze_event_handle_t}
const _ze_image_handle_t = Cvoid
const ze_image_handle_t = Ptr{_ze_image_handle_t}
const _ze_module_handle_t = Cvoid
const ze_module_handle_t = Ptr{_ze_module_handle_t}
const _ze_module_build_log_handle_t = Cvoid
const ze_module_build_log_handle_t = Ptr{_ze_module_build_log_handle_t}
const _ze_kernel_handle_t = Cvoid
const ze_kernel_handle_t = Ptr{_ze_kernel_handle_t}
const _ze_sampler_handle_t = Cvoid
const ze_sampler_handle_t = Ptr{_ze_sampler_handle_t}
const _ze_physical_mem_handle_t = Cvoid
const ze_physical_mem_handle_t = Ptr{_ze_physical_mem_handle_t}

struct _ze_ipc_mem_handle_t
    data::NTuple{64, UInt8}
end

const ze_ipc_mem_handle_t = _ze_ipc_mem_handle_t

struct _ze_ipc_event_pool_handle_t
    data::NTuple{64, UInt8}
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
    ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 1879113728
    ZE_RESULT_ERROR_NOT_AVAILABLE = 1879113729
    ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 1879179264
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
    ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES = 65537
    ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC = 65538
    ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES = 65539
    ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC = 131073
    ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC = 131074
    ZE_STRUCTURE_TYPE_FORCE_UINT32 = 2147483647
end

const ze_structure_type_t = _ze_structure_type_t
const ze_external_memory_type_flags_t = UInt32

@cenum _ze_external_memory_type_flag_t::UInt32 begin
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD = 1
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 2
    ZE_EXTERNAL_MEMORY_TYPE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_external_memory_type_flag_t = _ze_external_memory_type_flag_t

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

struct _ze_driver_uuid_t
    id::NTuple{16, UInt8}
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
    name::NTuple{256, UInt8}
    version::UInt32
end

const ze_driver_extension_properties_t = _ze_driver_extension_properties_t

struct _ze_device_uuid_t
    id::NTuple{16, UInt8}
end

const ze_device_uuid_t = _ze_device_uuid_t

@cenum _ze_device_type_t::UInt32 begin
    ZE_DEVICE_TYPE_GPU = 1
    ZE_DEVICE_TYPE_CPU = 2
    ZE_DEVICE_TYPE_FPGA = 3
    ZE_DEVICE_TYPE_MCA = 4
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
    name::NTuple{256, UInt8}
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
    subGroupSizes::NTuple{8, UInt32}
end

const ze_device_compute_properties_t = _ze_device_compute_properties_t

struct _ze_native_kernel_uuid_t
    id::NTuple{16, UInt8}
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
    name::NTuple{256, UInt8}
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
    pBuildFlags::Cstring
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
    pKernelName::Cstring
end

const ze_kernel_desc_t = _ze_kernel_desc_t

struct _ze_kernel_uuid_t
    kid::NTuple{16, UInt8}
    mid::NTuple{16, UInt8}
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
    pBuildFlags::Ptr{Cstring}
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
const ze_init_flags_t = UInt32

@cenum _ze_init_flag_t::UInt32 begin
    ZE_INIT_FLAG_GPU_ONLY = 1
    ZE_INIT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_init_flag_t = _ze_init_flag_t

@cenum _ze_api_version_t::UInt32 begin
    ZE_API_VERSION_1_0 = 65536
    ZE_API_VERSION_1_1 = 65537
    ZE_API_VERSION_CURRENT = 65537
    ZE_API_VERSION_FORCE_UINT32 = 2147483647
end

const ze_api_version_t = _ze_api_version_t

@cenum _ze_ipc_property_flag_t::UInt32 begin
    ZE_IPC_PROPERTY_FLAG_MEMORY = 1
    ZE_IPC_PROPERTY_FLAG_EVENT_POOL = 2
    ZE_IPC_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_ipc_property_flag_t = _ze_ipc_property_flag_t

@cenum _ze_device_property_flag_t::UInt32 begin
    ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1
    ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE = 2
    ZE_DEVICE_PROPERTY_FLAG_ECC = 4
    ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = 8
    ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_property_flag_t = _ze_device_property_flag_t

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

@cenum _ze_command_queue_group_property_flag_t::UInt32 begin
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 1
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY = 2
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS = 4
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS = 8
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_command_queue_group_property_flag_t = _ze_command_queue_group_property_flag_t

@cenum _ze_device_memory_property_flag_t::UInt32 begin
    ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD = 1
    ZE_DEVICE_MEMORY_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_memory_property_flag_t = _ze_device_memory_property_flag_t

@cenum _ze_memory_access_cap_flag_t::UInt32 begin
    ZE_MEMORY_ACCESS_CAP_FLAG_RW = 1
    ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC = 2
    ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT = 4
    ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC = 8
    ZE_MEMORY_ACCESS_CAP_FLAG_FORCE_UINT32 = 2147483647
end

const ze_memory_access_cap_flag_t = _ze_memory_access_cap_flag_t

@cenum _ze_device_cache_property_flag_t::UInt32 begin
    ZE_DEVICE_CACHE_PROPERTY_FLAG_USER_CONTROL = 1
    ZE_DEVICE_CACHE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_cache_property_flag_t = _ze_device_cache_property_flag_t

@cenum _ze_device_p2p_property_flag_t::UInt32 begin
    ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS = 1
    ZE_DEVICE_P2P_PROPERTY_FLAG_ATOMICS = 2
    ZE_DEVICE_P2P_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_p2p_property_flag_t = _ze_device_p2p_property_flag_t

@cenum _ze_context_flag_t::UInt32 begin
    ZE_CONTEXT_FLAG_TBD = 1
    ZE_CONTEXT_FLAG_FORCE_UINT32 = 2147483647
end

const ze_context_flag_t = _ze_context_flag_t

@cenum _ze_command_queue_flag_t::UInt32 begin
    ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY = 1
    ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_command_queue_flag_t = _ze_command_queue_flag_t

@cenum _ze_command_list_flag_t::UInt32 begin
    ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = 1
    ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = 2
    ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = 4
    ZE_COMMAND_LIST_FLAG_FORCE_UINT32 = 2147483647
end

const ze_command_list_flag_t = _ze_command_list_flag_t

@cenum _ze_memory_advice_t::UInt32 begin
    ZE_MEMORY_ADVICE_SET_READ_MOSTLY = 0
    ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY = 1
    ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION = 2
    ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION = 3
    ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY = 4
    ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 5
    ZE_MEMORY_ADVICE_BIAS_CACHED = 6
    ZE_MEMORY_ADVICE_BIAS_UNCACHED = 7
    ZE_MEMORY_ADVICE_FORCE_UINT32 = 2147483647
end

const ze_memory_advice_t = _ze_memory_advice_t

@cenum _ze_event_pool_flag_t::UInt32 begin
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE = 1
    ZE_EVENT_POOL_FLAG_IPC = 2
    ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP = 4
    ZE_EVENT_POOL_FLAG_FORCE_UINT32 = 2147483647
end

const ze_event_pool_flag_t = _ze_event_pool_flag_t

@cenum _ze_event_scope_flag_t::UInt32 begin
    ZE_EVENT_SCOPE_FLAG_SUBDEVICE = 1
    ZE_EVENT_SCOPE_FLAG_DEVICE = 2
    ZE_EVENT_SCOPE_FLAG_HOST = 4
    ZE_EVENT_SCOPE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_event_scope_flag_t = _ze_event_scope_flag_t

@cenum _ze_fence_flag_t::UInt32 begin
    ZE_FENCE_FLAG_SIGNALED = 1
    ZE_FENCE_FLAG_FORCE_UINT32 = 2147483647
end

const ze_fence_flag_t = _ze_fence_flag_t

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

@cenum _ze_device_mem_alloc_flag_t::UInt32 begin
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED = 1
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
    ZE_DEVICE_MEM_ALLOC_FLAG_FORCE_UINT32 = 2147483647
end

const ze_device_mem_alloc_flag_t = _ze_device_mem_alloc_flag_t

@cenum _ze_host_mem_alloc_flag_t::UInt32 begin
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = 1
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED = 4
    ZE_HOST_MEM_ALLOC_FLAG_FORCE_UINT32 = 2147483647
end

const ze_host_mem_alloc_flag_t = _ze_host_mem_alloc_flag_t
const ze_ipc_memory_flags_t = UInt32

@cenum _ze_ipc_memory_flag_t::UInt32 begin
    ZE_IPC_MEMORY_FLAG_TBD = 1
    ZE_IPC_MEMORY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_ipc_memory_flag_t = _ze_ipc_memory_flag_t

@cenum _ze_module_property_flag_t::UInt32 begin
    ZE_MODULE_PROPERTY_FLAG_IMPORTS = 1
    ZE_MODULE_PROPERTY_FLAG_FORCE_UINT32 = 2147483647
end

const ze_module_property_flag_t = _ze_module_property_flag_t

@cenum _ze_kernel_flag_t::UInt32 begin
    ZE_KERNEL_FLAG_FORCE_RESIDENCY = 1
    ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY = 2
    ZE_KERNEL_FLAG_FORCE_UINT32 = 2147483647
end

const ze_kernel_flag_t = _ze_kernel_flag_t
const ze_kernel_indirect_access_flags_t = UInt32

@cenum _ze_kernel_indirect_access_flag_t::UInt32 begin
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST = 1
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE = 2
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED = 4
    ZE_KERNEL_INDIRECT_ACCESS_FLAG_FORCE_UINT32 = 2147483647
end

const ze_kernel_indirect_access_flag_t = _ze_kernel_indirect_access_flag_t
const ze_cache_config_flags_t = UInt32

@cenum _ze_cache_config_flag_t::UInt32 begin
    ZE_CACHE_CONFIG_FLAG_LARGE_SLM = 1
    ZE_CACHE_CONFIG_FLAG_LARGE_DATA = 2
    ZE_CACHE_CONFIG_FLAG_FORCE_UINT32 = 2147483647
end

const ze_cache_config_flag_t = _ze_cache_config_flag_t

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

@cenum _ze_memory_access_attribute_t::UInt32 begin
    ZE_MEMORY_ACCESS_ATTRIBUTE_NONE = 0
    ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1
    ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY = 2
    ZE_MEMORY_ACCESS_ATTRIBUTE_FORCE_UINT32 = 2147483647
end

const ze_memory_access_attribute_t = _ze_memory_access_attribute_t

@cenum _ze_physical_mem_flag_t::UInt32 begin
    ZE_PHYSICAL_MEM_FLAG_TBD = 1
    ZE_PHYSICAL_MEM_FLAG_FORCE_UINT32 = 2147483647
end

const ze_physical_mem_flag_t = _ze_physical_mem_flag_t

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

struct _ze_init_params_t
    pflags::Ptr{ze_init_flags_t}
end

const ze_init_params_t = _ze_init_params_t
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
const ze_pfnDriverGetCb_t = Ptr{Cvoid}

struct _ze_driver_get_api_version_params_t
    phDriver::Ptr{ze_driver_handle_t}
    pversion::Ptr{Ptr{ze_api_version_t}}
end

const ze_driver_get_api_version_params_t = _ze_driver_get_api_version_params_t
const ze_pfnDriverGetApiVersionCb_t = Ptr{Cvoid}

struct _ze_driver_get_properties_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppDriverProperties::Ptr{Ptr{ze_driver_properties_t}}
end

const ze_driver_get_properties_params_t = _ze_driver_get_properties_params_t
const ze_pfnDriverGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_driver_get_ipc_properties_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppIpcProperties::Ptr{Ptr{ze_driver_ipc_properties_t}}
end

const ze_driver_get_ipc_properties_params_t = _ze_driver_get_ipc_properties_params_t
const ze_pfnDriverGetIpcPropertiesCb_t = Ptr{Cvoid}

struct _ze_driver_get_extension_properties_params_t
    phDriver::Ptr{ze_driver_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppExtensionProperties::Ptr{Ptr{ze_driver_extension_properties_t}}
end

const ze_driver_get_extension_properties_params_t = _ze_driver_get_extension_properties_params_t
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
const ze_pfnDeviceGetCb_t = Ptr{Cvoid}

struct _ze_device_get_sub_devices_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    pphSubdevices::Ptr{Ptr{ze_device_handle_t}}
end

const ze_device_get_sub_devices_params_t = _ze_device_get_sub_devices_params_t
const ze_pfnDeviceGetSubDevicesCb_t = Ptr{Cvoid}

struct _ze_device_get_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppDeviceProperties::Ptr{Ptr{ze_device_properties_t}}
end

const ze_device_get_properties_params_t = _ze_device_get_properties_params_t
const ze_pfnDeviceGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_compute_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppComputeProperties::Ptr{Ptr{ze_device_compute_properties_t}}
end

const ze_device_get_compute_properties_params_t = _ze_device_get_compute_properties_params_t
const ze_pfnDeviceGetComputePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_module_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppModuleProperties::Ptr{Ptr{ze_device_module_properties_t}}
end

const ze_device_get_module_properties_params_t = _ze_device_get_module_properties_params_t
const ze_pfnDeviceGetModulePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_command_queue_group_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppCommandQueueGroupProperties::Ptr{Ptr{ze_command_queue_group_properties_t}}
end

const ze_device_get_command_queue_group_properties_params_t = _ze_device_get_command_queue_group_properties_params_t
const ze_pfnDeviceGetCommandQueueGroupPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_memory_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppMemProperties::Ptr{Ptr{ze_device_memory_properties_t}}
end

const ze_device_get_memory_properties_params_t = _ze_device_get_memory_properties_params_t
const ze_pfnDeviceGetMemoryPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_memory_access_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppMemAccessProperties::Ptr{Ptr{ze_device_memory_access_properties_t}}
end

const ze_device_get_memory_access_properties_params_t = _ze_device_get_memory_access_properties_params_t
const ze_pfnDeviceGetMemoryAccessPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_cache_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppCacheProperties::Ptr{Ptr{ze_device_cache_properties_t}}
end

const ze_device_get_cache_properties_params_t = _ze_device_get_cache_properties_params_t
const ze_pfnDeviceGetCachePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_image_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppImageProperties::Ptr{Ptr{ze_device_image_properties_t}}
end

const ze_device_get_image_properties_params_t = _ze_device_get_image_properties_params_t
const ze_pfnDeviceGetImagePropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_external_memory_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    ppExternalMemoryProperties::Ptr{Ptr{ze_device_external_memory_properties_t}}
end

const ze_device_get_external_memory_properties_params_t = _ze_device_get_external_memory_properties_params_t
const ze_pfnDeviceGetExternalMemoryPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_get_p2_p_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    phPeerDevice::Ptr{ze_device_handle_t}
    ppP2PProperties::Ptr{Ptr{ze_device_p2p_properties_t}}
end

const ze_device_get_p2_p_properties_params_t = _ze_device_get_p2_p_properties_params_t
const ze_pfnDeviceGetP2PPropertiesCb_t = Ptr{Cvoid}

struct _ze_device_can_access_peer_params_t
    phDevice::Ptr{ze_device_handle_t}
    phPeerDevice::Ptr{ze_device_handle_t}
    pvalue::Ptr{Ptr{ze_bool_t}}
end

const ze_device_can_access_peer_params_t = _ze_device_can_access_peer_params_t
const ze_pfnDeviceCanAccessPeerCb_t = Ptr{Cvoid}

struct _ze_device_get_status_params_t
    phDevice::Ptr{ze_device_handle_t}
end

const ze_device_get_status_params_t = _ze_device_get_status_params_t
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
const ze_pfnContextCreateCb_t = Ptr{Cvoid}

struct _ze_context_destroy_params_t
    phContext::Ptr{ze_context_handle_t}
end

const ze_context_destroy_params_t = _ze_context_destroy_params_t
const ze_pfnContextDestroyCb_t = Ptr{Cvoid}

struct _ze_context_get_status_params_t
    phContext::Ptr{ze_context_handle_t}
end

const ze_context_get_status_params_t = _ze_context_get_status_params_t
const ze_pfnContextGetStatusCb_t = Ptr{Cvoid}

struct _ze_context_system_barrier_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
end

const ze_context_system_barrier_params_t = _ze_context_system_barrier_params_t
const ze_pfnContextSystemBarrierCb_t = Ptr{Cvoid}

struct _ze_context_make_memory_resident_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_context_make_memory_resident_params_t = _ze_context_make_memory_resident_params_t
const ze_pfnContextMakeMemoryResidentCb_t = Ptr{Cvoid}

struct _ze_context_evict_memory_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_context_evict_memory_params_t = _ze_context_evict_memory_params_t
const ze_pfnContextEvictMemoryCb_t = Ptr{Cvoid}

struct _ze_context_make_image_resident_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    phImage::Ptr{ze_image_handle_t}
end

const ze_context_make_image_resident_params_t = _ze_context_make_image_resident_params_t
const ze_pfnContextMakeImageResidentCb_t = Ptr{Cvoid}

struct _ze_context_evict_image_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    phImage::Ptr{ze_image_handle_t}
end

const ze_context_evict_image_params_t = _ze_context_evict_image_params_t
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
const ze_pfnCommandQueueCreateCb_t = Ptr{Cvoid}

struct _ze_command_queue_destroy_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
end

const ze_command_queue_destroy_params_t = _ze_command_queue_destroy_params_t
const ze_pfnCommandQueueDestroyCb_t = Ptr{Cvoid}

struct _ze_command_queue_execute_command_lists_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
    pnumCommandLists::Ptr{UInt32}
    pphCommandLists::Ptr{Ptr{ze_command_list_handle_t}}
    phFence::Ptr{ze_fence_handle_t}
end

const ze_command_queue_execute_command_lists_params_t = _ze_command_queue_execute_command_lists_params_t
const ze_pfnCommandQueueExecuteCommandListsCb_t = Ptr{Cvoid}

struct _ze_command_queue_synchronize_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
    ptimeout::Ptr{UInt64}
end

const ze_command_queue_synchronize_params_t = _ze_command_queue_synchronize_params_t
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
const ze_pfnCommandListCreateCb_t = Ptr{Cvoid}

struct _ze_command_list_create_immediate_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    paltdesc::Ptr{Ptr{ze_command_queue_desc_t}}
    pphCommandList::Ptr{Ptr{ze_command_list_handle_t}}
end

const ze_command_list_create_immediate_params_t = _ze_command_list_create_immediate_params_t
const ze_pfnCommandListCreateImmediateCb_t = Ptr{Cvoid}

struct _ze_command_list_destroy_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
end

const ze_command_list_destroy_params_t = _ze_command_list_destroy_params_t
const ze_pfnCommandListDestroyCb_t = Ptr{Cvoid}

struct _ze_command_list_close_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
end

const ze_command_list_close_params_t = _ze_command_list_close_params_t
const ze_pfnCommandListCloseCb_t = Ptr{Cvoid}

struct _ze_command_list_reset_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
end

const ze_command_list_reset_params_t = _ze_command_list_reset_params_t
const ze_pfnCommandListResetCb_t = Ptr{Cvoid}

struct _ze_command_list_append_write_global_timestamp_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pdstptr::Ptr{Ptr{UInt64}}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_write_global_timestamp_params_t = _ze_command_list_append_write_global_timestamp_params_t
const ze_pfnCommandListAppendWriteGlobalTimestampCb_t = Ptr{Cvoid}

struct _ze_command_list_append_barrier_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phSignalEvent::Ptr{ze_event_handle_t}
    pnumWaitEvents::Ptr{UInt32}
    pphWaitEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_barrier_params_t = _ze_command_list_append_barrier_params_t
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
const ze_pfnCommandListAppendImageCopyFromMemoryCb_t = Ptr{Cvoid}

struct _ze_command_list_append_memory_prefetch_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_command_list_append_memory_prefetch_params_t = _ze_command_list_append_memory_prefetch_params_t
const ze_pfnCommandListAppendMemoryPrefetchCb_t = Ptr{Cvoid}

struct _ze_command_list_append_mem_advise_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    padvice::Ptr{ze_memory_advice_t}
end

const ze_command_list_append_mem_advise_params_t = _ze_command_list_append_mem_advise_params_t
const ze_pfnCommandListAppendMemAdviseCb_t = Ptr{Cvoid}

struct _ze_command_list_append_signal_event_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phEvent::Ptr{ze_event_handle_t}
end

const ze_command_list_append_signal_event_params_t = _ze_command_list_append_signal_event_params_t
const ze_pfnCommandListAppendSignalEventCb_t = Ptr{Cvoid}

struct _ze_command_list_append_wait_on_events_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    pnumEvents::Ptr{UInt32}
    pphEvents::Ptr{Ptr{ze_event_handle_t}}
end

const ze_command_list_append_wait_on_events_params_t = _ze_command_list_append_wait_on_events_params_t
const ze_pfnCommandListAppendWaitOnEventsCb_t = Ptr{Cvoid}

struct _ze_command_list_append_event_reset_params_t
    phCommandList::Ptr{ze_command_list_handle_t}
    phEvent::Ptr{ze_event_handle_t}
end

const ze_command_list_append_event_reset_params_t = _ze_command_list_append_event_reset_params_t
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

struct _ze_fence_create_params_t
    phCommandQueue::Ptr{ze_command_queue_handle_t}
    pdesc::Ptr{Ptr{ze_fence_desc_t}}
    pphFence::Ptr{Ptr{ze_fence_handle_t}}
end

const ze_fence_create_params_t = _ze_fence_create_params_t
const ze_pfnFenceCreateCb_t = Ptr{Cvoid}

struct _ze_fence_destroy_params_t
    phFence::Ptr{ze_fence_handle_t}
end

const ze_fence_destroy_params_t = _ze_fence_destroy_params_t
const ze_pfnFenceDestroyCb_t = Ptr{Cvoid}

struct _ze_fence_host_synchronize_params_t
    phFence::Ptr{ze_fence_handle_t}
    ptimeout::Ptr{UInt64}
end

const ze_fence_host_synchronize_params_t = _ze_fence_host_synchronize_params_t
const ze_pfnFenceHostSynchronizeCb_t = Ptr{Cvoid}

struct _ze_fence_query_status_params_t
    phFence::Ptr{ze_fence_handle_t}
end

const ze_fence_query_status_params_t = _ze_fence_query_status_params_t
const ze_pfnFenceQueryStatusCb_t = Ptr{Cvoid}

struct _ze_fence_reset_params_t
    phFence::Ptr{ze_fence_handle_t}
end

const ze_fence_reset_params_t = _ze_fence_reset_params_t
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
const ze_pfnEventPoolCreateCb_t = Ptr{Cvoid}

struct _ze_event_pool_destroy_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
end

const ze_event_pool_destroy_params_t = _ze_event_pool_destroy_params_t
const ze_pfnEventPoolDestroyCb_t = Ptr{Cvoid}

struct _ze_event_pool_get_ipc_handle_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
    pphIpc::Ptr{Ptr{ze_ipc_event_pool_handle_t}}
end

const ze_event_pool_get_ipc_handle_params_t = _ze_event_pool_get_ipc_handle_params_t
const ze_pfnEventPoolGetIpcHandleCb_t = Ptr{Cvoid}

struct _ze_event_pool_open_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    phIpc::Ptr{ze_ipc_event_pool_handle_t}
    pphEventPool::Ptr{Ptr{ze_event_pool_handle_t}}
end

const ze_event_pool_open_ipc_handle_params_t = _ze_event_pool_open_ipc_handle_params_t
const ze_pfnEventPoolOpenIpcHandleCb_t = Ptr{Cvoid}

struct _ze_event_pool_close_ipc_handle_params_t
    phEventPool::Ptr{ze_event_pool_handle_t}
end

const ze_event_pool_close_ipc_handle_params_t = _ze_event_pool_close_ipc_handle_params_t
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
const ze_pfnEventCreateCb_t = Ptr{Cvoid}

struct _ze_event_destroy_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_destroy_params_t = _ze_event_destroy_params_t
const ze_pfnEventDestroyCb_t = Ptr{Cvoid}

struct _ze_event_host_signal_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_host_signal_params_t = _ze_event_host_signal_params_t
const ze_pfnEventHostSignalCb_t = Ptr{Cvoid}

struct _ze_event_host_synchronize_params_t
    phEvent::Ptr{ze_event_handle_t}
    ptimeout::Ptr{UInt64}
end

const ze_event_host_synchronize_params_t = _ze_event_host_synchronize_params_t
const ze_pfnEventHostSynchronizeCb_t = Ptr{Cvoid}

struct _ze_event_query_status_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_query_status_params_t = _ze_event_query_status_params_t
const ze_pfnEventQueryStatusCb_t = Ptr{Cvoid}

struct _ze_event_host_reset_params_t
    phEvent::Ptr{ze_event_handle_t}
end

const ze_event_host_reset_params_t = _ze_event_host_reset_params_t
const ze_pfnEventHostResetCb_t = Ptr{Cvoid}

struct _ze_event_query_kernel_timestamp_params_t
    phEvent::Ptr{ze_event_handle_t}
    pdstptr::Ptr{Ptr{ze_kernel_timestamp_result_t}}
end

const ze_event_query_kernel_timestamp_params_t = _ze_event_query_kernel_timestamp_params_t
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

struct _ze_image_get_properties_params_t
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_image_desc_t}}
    ppImageProperties::Ptr{Ptr{ze_image_properties_t}}
end

const ze_image_get_properties_params_t = _ze_image_get_properties_params_t
const ze_pfnImageGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_image_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_image_desc_t}}
    pphImage::Ptr{Ptr{ze_image_handle_t}}
end

const ze_image_create_params_t = _ze_image_create_params_t
const ze_pfnImageCreateCb_t = Ptr{Cvoid}

struct _ze_image_destroy_params_t
    phImage::Ptr{ze_image_handle_t}
end

const ze_image_destroy_params_t = _ze_image_destroy_params_t
const ze_pfnImageDestroyCb_t = Ptr{Cvoid}

struct _ze_image_callbacks_t
    pfnGetPropertiesCb::ze_pfnImageGetPropertiesCb_t
    pfnCreateCb::ze_pfnImageCreateCb_t
    pfnDestroyCb::ze_pfnImageDestroyCb_t
end

const ze_image_callbacks_t = _ze_image_callbacks_t

struct _ze_module_create_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    pdesc::Ptr{Ptr{ze_module_desc_t}}
    pphModule::Ptr{Ptr{ze_module_handle_t}}
    pphBuildLog::Ptr{Ptr{ze_module_build_log_handle_t}}
end

const ze_module_create_params_t = _ze_module_create_params_t
const ze_pfnModuleCreateCb_t = Ptr{Cvoid}

struct _ze_module_destroy_params_t
    phModule::Ptr{ze_module_handle_t}
end

const ze_module_destroy_params_t = _ze_module_destroy_params_t
const ze_pfnModuleDestroyCb_t = Ptr{Cvoid}

struct _ze_module_dynamic_link_params_t
    pnumModules::Ptr{UInt32}
    pphModules::Ptr{Ptr{ze_module_handle_t}}
    pphLinkLog::Ptr{Ptr{ze_module_build_log_handle_t}}
end

const ze_module_dynamic_link_params_t = _ze_module_dynamic_link_params_t
const ze_pfnModuleDynamicLinkCb_t = Ptr{Cvoid}

struct _ze_module_get_native_binary_params_t
    phModule::Ptr{ze_module_handle_t}
    ppSize::Ptr{Ptr{Csize_t}}
    ppModuleNativeBinary::Ptr{Ptr{UInt8}}
end

const ze_module_get_native_binary_params_t = _ze_module_get_native_binary_params_t
const ze_pfnModuleGetNativeBinaryCb_t = Ptr{Cvoid}

struct _ze_module_get_global_pointer_params_t
    phModule::Ptr{ze_module_handle_t}
    ppGlobalName::Ptr{Cstring}
    ppSize::Ptr{Ptr{Csize_t}}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_module_get_global_pointer_params_t = _ze_module_get_global_pointer_params_t
const ze_pfnModuleGetGlobalPointerCb_t = Ptr{Cvoid}

struct _ze_module_get_kernel_names_params_t
    phModule::Ptr{ze_module_handle_t}
    ppCount::Ptr{Ptr{UInt32}}
    ppNames::Ptr{Ptr{Cstring}}
end

const ze_module_get_kernel_names_params_t = _ze_module_get_kernel_names_params_t
const ze_pfnModuleGetKernelNamesCb_t = Ptr{Cvoid}

struct _ze_module_get_properties_params_t
    phModule::Ptr{ze_module_handle_t}
    ppModuleProperties::Ptr{Ptr{ze_module_properties_t}}
end

const ze_module_get_properties_params_t = _ze_module_get_properties_params_t
const ze_pfnModuleGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_module_get_function_pointer_params_t
    phModule::Ptr{ze_module_handle_t}
    ppFunctionName::Ptr{Cstring}
    ppfnFunction::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_module_get_function_pointer_params_t = _ze_module_get_function_pointer_params_t
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
const ze_pfnModuleBuildLogDestroyCb_t = Ptr{Cvoid}

struct _ze_module_build_log_get_string_params_t
    phModuleBuildLog::Ptr{ze_module_build_log_handle_t}
    ppSize::Ptr{Ptr{Csize_t}}
    ppBuildLog::Ptr{Cstring}
end

const ze_module_build_log_get_string_params_t = _ze_module_build_log_get_string_params_t
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
const ze_pfnKernelCreateCb_t = Ptr{Cvoid}

struct _ze_kernel_destroy_params_t
    phKernel::Ptr{ze_kernel_handle_t}
end

const ze_kernel_destroy_params_t = _ze_kernel_destroy_params_t
const ze_pfnKernelDestroyCb_t = Ptr{Cvoid}

struct _ze_kernel_set_cache_config_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pflags::Ptr{ze_cache_config_flags_t}
end

const ze_kernel_set_cache_config_params_t = _ze_kernel_set_cache_config_params_t
const ze_pfnKernelSetCacheConfigCb_t = Ptr{Cvoid}

struct _ze_kernel_set_group_size_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pgroupSizeX::Ptr{UInt32}
    pgroupSizeY::Ptr{UInt32}
    pgroupSizeZ::Ptr{UInt32}
end

const ze_kernel_set_group_size_params_t = _ze_kernel_set_group_size_params_t
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
const ze_pfnKernelSuggestGroupSizeCb_t = Ptr{Cvoid}

struct _ze_kernel_suggest_max_cooperative_group_count_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ptotalGroupCount::Ptr{Ptr{UInt32}}
end

const ze_kernel_suggest_max_cooperative_group_count_params_t = _ze_kernel_suggest_max_cooperative_group_count_params_t
const ze_pfnKernelSuggestMaxCooperativeGroupCountCb_t = Ptr{Cvoid}

struct _ze_kernel_set_argument_value_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pargIndex::Ptr{UInt32}
    pargSize::Ptr{Csize_t}
    ppArgValue::Ptr{Ptr{Cvoid}}
end

const ze_kernel_set_argument_value_params_t = _ze_kernel_set_argument_value_params_t
const ze_pfnKernelSetArgumentValueCb_t = Ptr{Cvoid}

struct _ze_kernel_set_indirect_access_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    pflags::Ptr{ze_kernel_indirect_access_flags_t}
end

const ze_kernel_set_indirect_access_params_t = _ze_kernel_set_indirect_access_params_t
const ze_pfnKernelSetIndirectAccessCb_t = Ptr{Cvoid}

struct _ze_kernel_get_indirect_access_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppFlags::Ptr{Ptr{ze_kernel_indirect_access_flags_t}}
end

const ze_kernel_get_indirect_access_params_t = _ze_kernel_get_indirect_access_params_t
const ze_pfnKernelGetIndirectAccessCb_t = Ptr{Cvoid}

struct _ze_kernel_get_source_attributes_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppSize::Ptr{Ptr{UInt32}}
    ppString::Ptr{Ptr{Cstring}}
end

const ze_kernel_get_source_attributes_params_t = _ze_kernel_get_source_attributes_params_t
const ze_pfnKernelGetSourceAttributesCb_t = Ptr{Cvoid}

struct _ze_kernel_get_properties_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppKernelProperties::Ptr{Ptr{ze_kernel_properties_t}}
end

const ze_kernel_get_properties_params_t = _ze_kernel_get_properties_params_t
const ze_pfnKernelGetPropertiesCb_t = Ptr{Cvoid}

struct _ze_kernel_get_name_params_t
    phKernel::Ptr{ze_kernel_handle_t}
    ppSize::Ptr{Ptr{Csize_t}}
    ppName::Ptr{Cstring}
end

const ze_kernel_get_name_params_t = _ze_kernel_get_name_params_t
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
const ze_pfnSamplerCreateCb_t = Ptr{Cvoid}

struct _ze_sampler_destroy_params_t
    phSampler::Ptr{ze_sampler_handle_t}
end

const ze_sampler_destroy_params_t = _ze_sampler_destroy_params_t
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
const ze_pfnPhysicalMemCreateCb_t = Ptr{Cvoid}

struct _ze_physical_mem_destroy_params_t
    phContext::Ptr{ze_context_handle_t}
    phPhysicalMemory::Ptr{ze_physical_mem_handle_t}
end

const ze_physical_mem_destroy_params_t = _ze_physical_mem_destroy_params_t
const ze_pfnPhysicalMemDestroyCb_t = Ptr{Cvoid}

struct _ze_physical_mem_callbacks_t
    pfnCreateCb::ze_pfnPhysicalMemCreateCb_t
    pfnDestroyCb::ze_pfnPhysicalMemDestroyCb_t
end

const ze_physical_mem_callbacks_t = _ze_physical_mem_callbacks_t

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
const ze_pfnMemAllocDeviceCb_t = Ptr{Cvoid}

struct _ze_mem_alloc_host_params_t
    phContext::Ptr{ze_context_handle_t}
    phost_desc::Ptr{Ptr{ze_host_mem_alloc_desc_t}}
    psize::Ptr{Csize_t}
    palignment::Ptr{Csize_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_mem_alloc_host_params_t = _ze_mem_alloc_host_params_t
const ze_pfnMemAllocHostCb_t = Ptr{Cvoid}

struct _ze_mem_free_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
end

const ze_mem_free_params_t = _ze_mem_free_params_t
const ze_pfnMemFreeCb_t = Ptr{Cvoid}

struct _ze_mem_get_alloc_properties_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppMemAllocProperties::Ptr{Ptr{ze_memory_allocation_properties_t}}
    pphDevice::Ptr{Ptr{ze_device_handle_t}}
end

const ze_mem_get_alloc_properties_params_t = _ze_mem_get_alloc_properties_params_t
const ze_pfnMemGetAllocPropertiesCb_t = Ptr{Cvoid}

struct _ze_mem_get_address_range_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppBase::Ptr{Ptr{Ptr{Cvoid}}}
    ppSize::Ptr{Ptr{Csize_t}}
end

const ze_mem_get_address_range_params_t = _ze_mem_get_address_range_params_t
const ze_pfnMemGetAddressRangeCb_t = Ptr{Cvoid}

struct _ze_mem_get_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    ppIpcHandle::Ptr{Ptr{ze_ipc_mem_handle_t}}
end

const ze_mem_get_ipc_handle_params_t = _ze_mem_get_ipc_handle_params_t
const ze_pfnMemGetIpcHandleCb_t = Ptr{Cvoid}

struct _ze_mem_open_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    phandle::Ptr{ze_ipc_mem_handle_t}
    pflags::Ptr{ze_ipc_memory_flags_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_mem_open_ipc_handle_params_t = _ze_mem_open_ipc_handle_params_t
const ze_pfnMemOpenIpcHandleCb_t = Ptr{Cvoid}

struct _ze_mem_close_ipc_handle_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
end

const ze_mem_close_ipc_handle_params_t = _ze_mem_close_ipc_handle_params_t
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

struct _ze_virtual_mem_reserve_params_t
    phContext::Ptr{ze_context_handle_t}
    ppStart::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    ppptr::Ptr{Ptr{Ptr{Cvoid}}}
end

const ze_virtual_mem_reserve_params_t = _ze_virtual_mem_reserve_params_t
const ze_pfnVirtualMemReserveCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_free_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_virtual_mem_free_params_t = _ze_virtual_mem_free_params_t
const ze_pfnVirtualMemFreeCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_query_page_size_params_t
    phContext::Ptr{ze_context_handle_t}
    phDevice::Ptr{ze_device_handle_t}
    psize::Ptr{Csize_t}
    ppagesize::Ptr{Ptr{Csize_t}}
end

const ze_virtual_mem_query_page_size_params_t = _ze_virtual_mem_query_page_size_params_t
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
const ze_pfnVirtualMemMapCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_unmap_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
end

const ze_virtual_mem_unmap_params_t = _ze_virtual_mem_unmap_params_t
const ze_pfnVirtualMemUnmapCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_set_access_attribute_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    paccess::Ptr{ze_memory_access_attribute_t}
end

const ze_virtual_mem_set_access_attribute_params_t = _ze_virtual_mem_set_access_attribute_params_t
const ze_pfnVirtualMemSetAccessAttributeCb_t = Ptr{Cvoid}

struct _ze_virtual_mem_get_access_attribute_params_t
    phContext::Ptr{ze_context_handle_t}
    pptr::Ptr{Ptr{Cvoid}}
    psize::Ptr{Csize_t}
    paccess::Ptr{Ptr{ze_memory_access_attribute_t}}
    poutSize::Ptr{Ptr{Csize_t}}
end

const ze_virtual_mem_get_access_attribute_params_t = _ze_virtual_mem_get_access_attribute_params_t
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
