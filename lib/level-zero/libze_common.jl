# Automatically generated using Clang.jl


# Skipping MacroDefinition: ZE_MAKE_VERSION ( _major , _minor ) ( ( _major << 16 ) | ( _minor & 0x0000ffff ) )
# Skipping MacroDefinition: ZE_MAJOR_VERSION ( _ver ) ( _ver >> 16 )
# Skipping MacroDefinition: ZE_MINOR_VERSION ( _ver ) ( _ver & 0x0000ffff )

const ZE_ENABLE_OCL_INTEROP = 0
const ZE_MAX_IPC_HANDLE_SIZE = 64

# Skipping MacroDefinition: ZE_BIT ( _i ) ( 1 << _i )

const ZE_MAX_DRIVER_UUID_SIZE = 16
const ZE_MAX_DEVICE_UUID_SIZE = 16
const ZE_MAX_DEVICE_NAME = 256
const ZE_SUBGROUPSIZE_COUNT = 8
const ZE_MAX_NATIVE_KERNEL_UUID_SIZE = 16
const ZE_MAX_KERNEL_NAME = 256
const ze_bool_t = UInt8
const _ze_driver_handle_t = Cvoid
const ze_driver_handle_t = Ptr{_ze_driver_handle_t}
const _ze_device_handle_t = Cvoid
const ze_device_handle_t = Ptr{_ze_device_handle_t}
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
    ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 1879113728
    ZE_RESULT_ERROR_NOT_AVAILABLE = 1879113729
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
    ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 2013265944
    ZE_RESULT_ERROR_OVERLAPPING_REGIONS = 2013265945
    ZE_RESULT_ERROR_UNKNOWN = 2147483647
end


const ze_result_t = _ze_result_t

struct _ze_driver_uuid_t
    id::NTuple{16, UInt8}
end

const ze_driver_uuid_t = _ze_driver_uuid_t

@cenum _ze_driver_properties_version_t::UInt32 begin
    ZE_DRIVER_PROPERTIES_VERSION_CURRENT = 91
end


const ze_driver_properties_version_t = _ze_driver_properties_version_t

struct _ze_driver_properties_t
    version::ze_driver_properties_version_t
    uuid::ze_driver_uuid_t
    driverVersion::UInt32
end

const ze_driver_properties_t = _ze_driver_properties_t

@cenum _ze_driver_ipc_properties_version_t::UInt32 begin
    ZE_DRIVER_IPC_PROPERTIES_VERSION_CURRENT = 91
end


const ze_driver_ipc_properties_version_t = _ze_driver_ipc_properties_version_t

struct _ze_driver_ipc_properties_t
    version::ze_driver_ipc_properties_version_t
    memsSupported::ze_bool_t
    eventsSupported::ze_bool_t
end

const ze_driver_ipc_properties_t = _ze_driver_ipc_properties_t

struct _ze_device_uuid_t
    id::NTuple{16, UInt8}
end

const ze_device_uuid_t = _ze_device_uuid_t

@cenum _ze_device_properties_version_t::UInt32 begin
    ZE_DEVICE_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_properties_version_t = _ze_device_properties_version_t

@cenum _ze_device_type_t::UInt32 begin
    ZE_DEVICE_TYPE_GPU = 1
    ZE_DEVICE_TYPE_FPGA = 2
end


const ze_device_type_t = _ze_device_type_t

struct _ze_device_properties_t
    version::ze_device_properties_version_t
    type::ze_device_type_t
    vendorId::UInt32
    deviceId::UInt32
    uuid::ze_device_uuid_t
    isSubdevice::ze_bool_t
    subdeviceId::UInt32
    coreClockRate::UInt32
    unifiedMemorySupported::ze_bool_t
    eccMemorySupported::ze_bool_t
    onDemandPageFaultsSupported::ze_bool_t
    maxCommandQueues::UInt32
    numAsyncComputeEngines::UInt32
    numAsyncCopyEngines::UInt32
    maxCommandQueuePriority::UInt32
    numThreadsPerEU::UInt32
    physicalEUSimdWidth::UInt32
    numEUsPerSubslice::UInt32
    numSubslicesPerSlice::UInt32
    numSlices::UInt32
    timerResolution::UInt64
    name::NTuple{256, UInt8}
end

const ze_device_properties_t = _ze_device_properties_t

@cenum _ze_device_compute_properties_version_t::UInt32 begin
    ZE_DEVICE_COMPUTE_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_compute_properties_version_t = _ze_device_compute_properties_version_t

struct _ze_device_compute_properties_t
    version::ze_device_compute_properties_version_t
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

@cenum _ze_device_kernel_properties_version_t::UInt32 begin
    ZE_DEVICE_KERNEL_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_kernel_properties_version_t = _ze_device_kernel_properties_version_t

@cenum _ze_fp_capabilities_t::UInt32 begin
    ZE_FP_CAPS_NONE = 0
    ZE_FP_CAPS_DENORM = 1
    ZE_FP_CAPS_INF_NAN = 2
    ZE_FP_CAPS_ROUND_TO_NEAREST = 4
    ZE_FP_CAPS_ROUND_TO_ZERO = 8
    ZE_FP_CAPS_ROUND_TO_INF = 16
    ZE_FP_CAPS_FMA = 32
    ZE_FP_CAPS_ROUNDED_DIVIDE_SQRT = 64
    ZE_FP_CAPS_SOFT_FLOAT = 128
end


const ze_fp_capabilities_t = _ze_fp_capabilities_t

struct _ze_device_kernel_properties_t
    version::ze_device_kernel_properties_version_t
    spirvVersionSupported::UInt32
    nativeKernelSupported::ze_native_kernel_uuid_t
    fp16Supported::ze_bool_t
    fp64Supported::ze_bool_t
    int64AtomicsSupported::ze_bool_t
    dp4aSupported::ze_bool_t
    halfFpCapabilities::ze_fp_capabilities_t
    singleFpCapabilities::ze_fp_capabilities_t
    doubleFpCapabilities::ze_fp_capabilities_t
    maxArgumentsSize::UInt32
    printfBufferSize::UInt32
end

const ze_device_kernel_properties_t = _ze_device_kernel_properties_t

@cenum _ze_device_memory_properties_version_t::UInt32 begin
    ZE_DEVICE_MEMORY_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_memory_properties_version_t = _ze_device_memory_properties_version_t

struct _ze_device_memory_properties_t
    version::ze_device_memory_properties_version_t
    maxClockRate::UInt32
    maxBusWidth::UInt32
    totalSize::UInt64
end

const ze_device_memory_properties_t = _ze_device_memory_properties_t

@cenum _ze_device_memory_access_properties_version_t::UInt32 begin
    ZE_DEVICE_MEMORY_ACCESS_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_memory_access_properties_version_t = _ze_device_memory_access_properties_version_t

@cenum _ze_memory_access_capabilities_t::UInt32 begin
    ZE_MEMORY_ACCESS_NONE = 0
    ZE_MEMORY_ACCESS = 1
    ZE_MEMORY_ATOMIC_ACCESS = 2
    ZE_MEMORY_CONCURRENT_ACCESS = 4
    ZE_MEMORY_CONCURRENT_ATOMIC_ACCESS = 8
end


const ze_memory_access_capabilities_t = _ze_memory_access_capabilities_t

struct _ze_device_memory_access_properties_t
    version::ze_device_memory_access_properties_version_t
    hostAllocCapabilities::ze_memory_access_capabilities_t
    deviceAllocCapabilities::ze_memory_access_capabilities_t
    sharedSingleDeviceAllocCapabilities::ze_memory_access_capabilities_t
    sharedCrossDeviceAllocCapabilities::ze_memory_access_capabilities_t
    sharedSystemAllocCapabilities::ze_memory_access_capabilities_t
end

const ze_device_memory_access_properties_t = _ze_device_memory_access_properties_t

@cenum _ze_device_cache_properties_version_t::UInt32 begin
    ZE_DEVICE_CACHE_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_cache_properties_version_t = _ze_device_cache_properties_version_t

struct _ze_device_cache_properties_t
    version::ze_device_cache_properties_version_t
    intermediateCacheControlSupported::ze_bool_t
    intermediateCacheSize::Csize_t
    intermediateCachelineSize::UInt32
    lastLevelCacheSizeControlSupported::ze_bool_t
    lastLevelCacheSize::Csize_t
    lastLevelCachelineSize::UInt32
end

const ze_device_cache_properties_t = _ze_device_cache_properties_t

@cenum _ze_device_image_properties_version_t::UInt32 begin
    ZE_DEVICE_IMAGE_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_image_properties_version_t = _ze_device_image_properties_version_t

struct _ze_device_image_properties_t
    version::ze_device_image_properties_version_t
    supported::ze_bool_t
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

@cenum _ze_device_p2p_properties_version_t::UInt32 begin
    ZE_DEVICE_P2P_PROPERTIES_VERSION_CURRENT = 91
end


const ze_device_p2p_properties_version_t = _ze_device_p2p_properties_version_t

struct _ze_device_p2p_properties_t
    version::ze_device_p2p_properties_version_t
    accessSupported::ze_bool_t
    atomicsSupported::ze_bool_t
end

const ze_device_p2p_properties_t = _ze_device_p2p_properties_t

@cenum _ze_command_queue_desc_version_t::UInt32 begin
    ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT = 91
end


const ze_command_queue_desc_version_t = _ze_command_queue_desc_version_t

@cenum _ze_command_queue_flag_t::UInt32 begin
    ZE_COMMAND_QUEUE_FLAG_NONE = 0
    ZE_COMMAND_QUEUE_FLAG_COPY_ONLY = 1
    ZE_COMMAND_QUEUE_FLAG_LOGICAL_ONLY = 2
    ZE_COMMAND_QUEUE_FLAG_SINGLE_SLICE_ONLY = 4
    ZE_COMMAND_QUEUE_FLAG_SUPPORTS_COOPERATIVE_KERNELS = 8
end


const ze_command_queue_flag_t = _ze_command_queue_flag_t

@cenum _ze_command_queue_mode_t::UInt32 begin
    ZE_COMMAND_QUEUE_MODE_DEFAULT = 0
    ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2
end


const ze_command_queue_mode_t = _ze_command_queue_mode_t

@cenum _ze_command_queue_priority_t::UInt32 begin
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
    ZE_COMMAND_QUEUE_PRIORITY_LOW = 1
    ZE_COMMAND_QUEUE_PRIORITY_HIGH = 2
end


const ze_command_queue_priority_t = _ze_command_queue_priority_t

struct _ze_command_queue_desc_t
    version::ze_command_queue_desc_version_t
    flags::ze_command_queue_flag_t
    mode::ze_command_queue_mode_t
    priority::ze_command_queue_priority_t
    ordinal::UInt32
end

const ze_command_queue_desc_t = _ze_command_queue_desc_t

@cenum _ze_command_list_desc_version_t::UInt32 begin
    ZE_COMMAND_LIST_DESC_VERSION_CURRENT = 91
end


const ze_command_list_desc_version_t = _ze_command_list_desc_version_t

@cenum _ze_command_list_flag_t::UInt32 begin
    ZE_COMMAND_LIST_FLAG_NONE = 0
    ZE_COMMAND_LIST_FLAG_COPY_ONLY = 1
    ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING = 2
    ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT = 4
    ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY = 8
end


const ze_command_list_flag_t = _ze_command_list_flag_t

struct _ze_command_list_desc_t
    version::ze_command_list_desc_version_t
    flags::ze_command_list_flag_t
end

const ze_command_list_desc_t = _ze_command_list_desc_t

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
    ZE_IMAGE_FORMAT_LAYOUT_YUAV = 21
    ZE_IMAGE_FORMAT_LAYOUT_P010 = 22
    ZE_IMAGE_FORMAT_LAYOUT_Y410 = 23
    ZE_IMAGE_FORMAT_LAYOUT_P012 = 24
    ZE_IMAGE_FORMAT_LAYOUT_Y16 = 25
    ZE_IMAGE_FORMAT_LAYOUT_P016 = 26
    ZE_IMAGE_FORMAT_LAYOUT_Y216 = 27
    ZE_IMAGE_FORMAT_LAYOUT_P216 = 28
    ZE_IMAGE_FORMAT_LAYOUT_P416 = 29
end


const ze_image_format_layout_t = _ze_image_format_layout_t

@cenum _ze_image_format_type_t::UInt32 begin
    ZE_IMAGE_FORMAT_TYPE_UINT = 0
    ZE_IMAGE_FORMAT_TYPE_SINT = 1
    ZE_IMAGE_FORMAT_TYPE_UNORM = 2
    ZE_IMAGE_FORMAT_TYPE_SNORM = 3
    ZE_IMAGE_FORMAT_TYPE_FLOAT = 4
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
end


const ze_image_format_swizzle_t = _ze_image_format_swizzle_t

struct _ze_image_format_desc_t
    layout::ze_image_format_layout_t
    type::ze_image_format_type_t
    x::ze_image_format_swizzle_t
    y::ze_image_format_swizzle_t
    z::ze_image_format_swizzle_t
    w::ze_image_format_swizzle_t
end

const ze_image_format_desc_t = _ze_image_format_desc_t

@cenum _ze_image_desc_version_t::UInt32 begin
    ZE_IMAGE_DESC_VERSION_CURRENT = 91
end


const ze_image_desc_version_t = _ze_image_desc_version_t

@cenum _ze_image_flag_t::UInt32 begin
    ZE_IMAGE_FLAG_PROGRAM_READ = 1
    ZE_IMAGE_FLAG_PROGRAM_WRITE = 2
    ZE_IMAGE_FLAG_BIAS_CACHED = 4
    ZE_IMAGE_FLAG_BIAS_UNCACHED = 8
end


const ze_image_flag_t = _ze_image_flag_t

@cenum _ze_image_type_t::UInt32 begin
    ZE_IMAGE_TYPE_1D = 0
    ZE_IMAGE_TYPE_1DARRAY = 1
    ZE_IMAGE_TYPE_2D = 2
    ZE_IMAGE_TYPE_2DARRAY = 3
    ZE_IMAGE_TYPE_3D = 4
    ZE_IMAGE_TYPE_BUFFER = 5
end


const ze_image_type_t = _ze_image_type_t

struct _ze_image_desc_t
    version::ze_image_desc_version_t
    flags::ze_image_flag_t
    type::ze_image_type_t
    format::ze_image_format_desc_t
    width::UInt64
    height::UInt32
    depth::UInt32
    arraylevels::UInt32
    miplevels::UInt32
end

const ze_image_desc_t = _ze_image_desc_t

@cenum _ze_image_properties_version_t::UInt32 begin
    ZE_IMAGE_PROPERTIES_VERSION_CURRENT = 91
end


const ze_image_properties_version_t = _ze_image_properties_version_t

@cenum _ze_image_sampler_filter_flags_t::UInt32 begin
    ZE_IMAGE_SAMPLER_FILTER_FLAGS_NONE = 0
    ZE_IMAGE_SAMPLER_FILTER_FLAGS_POINT = 1
    ZE_IMAGE_SAMPLER_FILTER_FLAGS_LINEAR = 2
end


const ze_image_sampler_filter_flags_t = _ze_image_sampler_filter_flags_t

struct _ze_image_properties_t
    version::ze_image_properties_version_t
    samplerFilterFlags::ze_image_sampler_filter_flags_t
end

const ze_image_properties_t = _ze_image_properties_t

struct _ze_module_constants_t
    numConstants::UInt32
    pConstantIds::Ptr{UInt32}
    pConstantValues::Ptr{UInt64}
end

const ze_module_constants_t = _ze_module_constants_t

@cenum _ze_module_desc_version_t::UInt32 begin
    ZE_MODULE_DESC_VERSION_CURRENT = 91
end


const ze_module_desc_version_t = _ze_module_desc_version_t

@cenum _ze_module_format_t::UInt32 begin
    ZE_MODULE_FORMAT_IL_SPIRV = 0
    ZE_MODULE_FORMAT_NATIVE = 1
end


const ze_module_format_t = _ze_module_format_t

struct _ze_module_desc_t
    version::ze_module_desc_version_t
    format::ze_module_format_t
    inputSize::Csize_t
    pInputModule::Ptr{UInt8}
    pBuildFlags::Cstring
    pConstants::Ptr{ze_module_constants_t}
end

const ze_module_desc_t = _ze_module_desc_t

@cenum _ze_kernel_desc_version_t::UInt32 begin
    ZE_KERNEL_DESC_VERSION_CURRENT = 91
end


const ze_kernel_desc_version_t = _ze_kernel_desc_version_t

@cenum _ze_kernel_flag_t::UInt32 begin
    ZE_KERNEL_FLAG_NONE = 0
    ZE_KERNEL_FLAG_FORCE_RESIDENCY = 1
end


const ze_kernel_flag_t = _ze_kernel_flag_t

struct _ze_kernel_desc_t
    version::ze_kernel_desc_version_t
    flags::ze_kernel_flag_t
    pKernelName::Cstring
end

const ze_kernel_desc_t = _ze_kernel_desc_t

@cenum _ze_kernel_properties_version_t::UInt32 begin
    ZE_KERNEL_PROPERTIES_VERSION_CURRENT = 91
end


const ze_kernel_properties_version_t = _ze_kernel_properties_version_t

struct _ze_kernel_properties_t
    version::ze_kernel_properties_version_t
    name::NTuple{256, UInt8}
    numKernelArgs::UInt32
    requiredGroupSizeX::UInt32
    requiredGroupSizeY::UInt32
    requiredGroupSizeZ::UInt32
end

const ze_kernel_properties_t = _ze_kernel_properties_t

struct _ze_group_count_t
    groupCountX::UInt32
    groupCountY::UInt32
    groupCountZ::UInt32
end

const ze_group_count_t = _ze_group_count_t

@cenum _ze_event_pool_desc_version_t::UInt32 begin
    ZE_EVENT_POOL_DESC_VERSION_CURRENT = 91
end


const ze_event_pool_desc_version_t = _ze_event_pool_desc_version_t

@cenum _ze_event_pool_flag_t::UInt32 begin
    ZE_EVENT_POOL_FLAG_DEFAULT = 0
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE = 1
    ZE_EVENT_POOL_FLAG_IPC = 2
    ZE_EVENT_POOL_FLAG_TIMESTAMP = 4
end


const ze_event_pool_flag_t = _ze_event_pool_flag_t

struct _ze_event_pool_desc_t
    version::ze_event_pool_desc_version_t
    flags::ze_event_pool_flag_t
    count::UInt32
end

const ze_event_pool_desc_t = _ze_event_pool_desc_t

@cenum _ze_event_desc_version_t::UInt32 begin
    ZE_EVENT_DESC_VERSION_CURRENT = 91
end


const ze_event_desc_version_t = _ze_event_desc_version_t

@cenum _ze_event_scope_flag_t::UInt32 begin
    ZE_EVENT_SCOPE_FLAG_NONE = 0
    ZE_EVENT_SCOPE_FLAG_SUBDEVICE = 1
    ZE_EVENT_SCOPE_FLAG_DEVICE = 2
    ZE_EVENT_SCOPE_FLAG_HOST = 4
end


const ze_event_scope_flag_t = _ze_event_scope_flag_t

struct _ze_event_desc_t
    version::ze_event_desc_version_t
    index::UInt32
    signal::ze_event_scope_flag_t
    wait::ze_event_scope_flag_t
end

const ze_event_desc_t = _ze_event_desc_t

@cenum _ze_sampler_desc_version_t::UInt32 begin
    ZE_SAMPLER_DESC_VERSION_CURRENT = 91
end


const ze_sampler_desc_version_t = _ze_sampler_desc_version_t

@cenum _ze_sampler_address_mode_t::UInt32 begin
    ZE_SAMPLER_ADDRESS_MODE_NONE = 0
    ZE_SAMPLER_ADDRESS_MODE_REPEAT = 1
    ZE_SAMPLER_ADDRESS_MODE_CLAMP = 2
    ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3
    ZE_SAMPLER_ADDRESS_MODE_MIRROR = 4
end


const ze_sampler_address_mode_t = _ze_sampler_address_mode_t

@cenum _ze_sampler_filter_mode_t::UInt32 begin
    ZE_SAMPLER_FILTER_MODE_NEAREST = 0
    ZE_SAMPLER_FILTER_MODE_LINEAR = 1
end


const ze_sampler_filter_mode_t = _ze_sampler_filter_mode_t

struct _ze_sampler_desc_t
    version::ze_sampler_desc_version_t
    addressMode::ze_sampler_address_mode_t
    filterMode::ze_sampler_filter_mode_t
    isNormalized::ze_bool_t
end

const ze_sampler_desc_t = _ze_sampler_desc_t

@cenum _ze_device_mem_alloc_desc_version_t::UInt32 begin
    ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT = 91
end


const ze_device_mem_alloc_desc_version_t = _ze_device_mem_alloc_desc_version_t

@cenum _ze_device_mem_alloc_flag_t::UInt32 begin
    ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT = 0
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED = 1
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
end


const ze_device_mem_alloc_flag_t = _ze_device_mem_alloc_flag_t

struct _ze_device_mem_alloc_desc_t
    version::ze_device_mem_alloc_desc_version_t
    flags::ze_device_mem_alloc_flag_t
    ordinal::UInt32
end

const ze_device_mem_alloc_desc_t = _ze_device_mem_alloc_desc_t

@cenum _ze_host_mem_alloc_desc_version_t::UInt32 begin
    ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT = 91
end


const ze_host_mem_alloc_desc_version_t = _ze_host_mem_alloc_desc_version_t

@cenum _ze_host_mem_alloc_flag_t::UInt32 begin
    ZE_HOST_MEM_ALLOC_FLAG_DEFAULT = 0
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = 1
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED = 2
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED = 4
end


const ze_host_mem_alloc_flag_t = _ze_host_mem_alloc_flag_t

struct _ze_host_mem_alloc_desc_t
    version::ze_host_mem_alloc_desc_version_t
    flags::ze_host_mem_alloc_flag_t
end

const ze_host_mem_alloc_desc_t = _ze_host_mem_alloc_desc_t

@cenum _ze_memory_allocation_properties_version_t::UInt32 begin
    ZE_MEMORY_ALLOCATION_PROPERTIES_VERSION_CURRENT = 91
end


const ze_memory_allocation_properties_version_t = _ze_memory_allocation_properties_version_t

@cenum _ze_memory_type_t::UInt32 begin
    ZE_MEMORY_TYPE_UNKNOWN = 0
    ZE_MEMORY_TYPE_HOST = 1
    ZE_MEMORY_TYPE_DEVICE = 2
    ZE_MEMORY_TYPE_SHARED = 3
end


const ze_memory_type_t = _ze_memory_type_t

struct _ze_memory_allocation_properties_t
    version::ze_memory_allocation_properties_version_t
    type::ze_memory_type_t
    id::UInt64
end

const ze_memory_allocation_properties_t = _ze_memory_allocation_properties_t

@cenum _ze_fence_desc_version_t::UInt32 begin
    ZE_FENCE_DESC_VERSION_CURRENT = 91
end


const ze_fence_desc_version_t = _ze_fence_desc_version_t

@cenum _ze_fence_flag_t::UInt32 begin
    ZE_FENCE_FLAG_NONE = 0
end


const ze_fence_flag_t = _ze_fence_flag_t

struct _ze_fence_desc_t
    version::ze_fence_desc_version_t
    flags::ze_fence_flag_t
end

const ze_fence_desc_t = _ze_fence_desc_t

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

@cenum _ze_init_flag_t::UInt32 begin
    ZE_INIT_FLAG_NONE = 0
    ZE_INIT_FLAG_GPU_ONLY = 1
end


const ze_init_flag_t = _ze_init_flag_t

@cenum _ze_api_version_t::UInt32 begin
    ZE_API_VERSION_1_0 = 91
end


const ze_api_version_t = _ze_api_version_t

@cenum _ze_cache_config_t::UInt32 begin
    ZE_CACHE_CONFIG_DEFAULT = 1
    ZE_CACHE_CONFIG_LARGE_SLM = 2
    ZE_CACHE_CONFIG_LARGE_DATA = 4
end


const ze_cache_config_t = _ze_cache_config_t

@cenum _ze_kernel_attribute_t::UInt32 begin
    ZE_KERNEL_ATTR_INDIRECT_HOST_ACCESS = 0
    ZE_KERNEL_ATTR_INDIRECT_DEVICE_ACCESS = 1
    ZE_KERNEL_ATTR_INDIRECT_SHARED_ACCESS = 2
    ZE_KERNEL_ATTR_SOURCE_ATTRIBUTE = 3
end


const ze_kernel_attribute_t = _ze_kernel_attribute_t

@cenum _ze_event_timestamp_type_t::UInt32 begin
    ZE_EVENT_TIMESTAMP_GLOBAL_START = 0
    ZE_EVENT_TIMESTAMP_GLOBAL_END = 1
    ZE_EVENT_TIMESTAMP_CONTEXT_START = 2
    ZE_EVENT_TIMESTAMP_CONTEXT_END = 3
end


const ze_event_timestamp_type_t = _ze_event_timestamp_type_t

@cenum _ze_ipc_memory_flag_t::UInt32 begin
    ZE_IPC_MEMORY_FLAG_NONE = 0
end


const ze_ipc_memory_flag_t = _ze_ipc_memory_flag_t

@cenum _ze_memory_advice_t::UInt32 begin
    ZE_MEMORY_ADVICE_SET_READ_MOSTLY = 0
    ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY = 1
    ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION = 2
    ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION = 3
    ZE_MEMORY_ADVICE_SET_ACCESSED_BY = 4
    ZE_MEMORY_ADVICE_CLEAR_ACCESSED_BY = 5
    ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY = 6
    ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY = 7
    ZE_MEMORY_ADVICE_BIAS_CACHED = 8
    ZE_MEMORY_ADVICE_BIAS_UNCACHED = 9
end


const ze_memory_advice_t = _ze_memory_advice_t

