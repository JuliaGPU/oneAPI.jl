using CEnum: CEnum, @cenum

# outlined functionality to avoid GC frame allocation
@noinline throw_api_error(res) = throw(PTIError(res))

function check(f)
    res = f()
    if res != PTI_SUCCESS
        throw_api_error(res)
    end
    return
end

@cenum pti_result::UInt32 begin
    PTI_SUCCESS = 0
    PTI_STATUS_END_OF_BUFFER = 1
    PTI_ERROR_NOT_IMPLEMENTED = 2
    PTI_ERROR_BAD_ARGUMENT = 3
    PTI_ERROR_NO_CALLBACKS_SET = 4
    PTI_ERROR_EXTERNAL_ID_QUEUE_EMPTY = 5
    PTI_ERROR_BAD_TIMESTAMP = 6
    PTI_ERROR_BAD_API_ID = 7
    PTI_ERROR_NO_GPU_VIEWS_ENABLED = 8
    PTI_ERROR_DRIVER = 50
    PTI_ERROR_TRACING_NOT_INITIALIZED = 51
    PTI_ERROR_L0_LOCAL_PROFILING_NOT_SUPPORTED = 52
    PTI_ERROR_METRICS_ENABLE_FAILED = 98
    PTI_ERROR_METRICS_RUNTIME_ENABLE_UNSUPPORTED = 99
    PTI_ERROR_METRICS_COLLECTION_NOT_ENABLED = 100
    PTI_ERROR_METRICS_COLLECTION_NOT_DISABLED = 101
    PTI_ERROR_METRICS_COLLECTION_NOT_PAUSED = 102
    PTI_ERROR_METRICS_COLLECTION_ALREADY_PAUSED = 103
    PTI_ERROR_METRICS_COLLECTION_ALREADY_ENABLED = 104
    PTI_ERROR_METRICS_BAD_COLLECTION_CONFIGURATION = 105
    PTI_ERROR_METRICS_NO_DATA_COLLECTED = 106
    PTI_ERROR_METRICS_SCOPE_METRIC_NOT_FOUND = 151
    PTI_ERROR_METRICS_SCOPE_NOT_A_SINGLE_GROUP = 152
    PTI_ERROR_METRICS_SCOPE_OUT_OF_MEMORY = 153
    PTI_ERROR_METRICS_SCOPE_COLLECTION_BUFFER_TOO_SMALL = 154
    PTI_ERROR_METRICS_SCOPE_INSUFFICIENT_BUFFER = 155
    PTI_ERROR_METRICS_SCOPE_INVALID_COLLECTION_BUFFER = 156
    PTI_WARN_METRICS_SCOPE_PARTIAL_BUFFER = 157
    PTI_WARN_BUFFER_NOT_FINALIZED = 158
    PTI_ERROR_METRICS_SCOPE_DEVICE_TYPE_NOT_UNIFORM = 159
    PTI_ERROR_INTERNAL = 200
    PTI_ERROR_PC_SAMPLING_ALREADY_ENABLED = 250
    PTI_ERROR_PC_SAMPLING_NOT_CONFIGURED = 251
    PTI_ERROR_PC_SAMPLING_ALREADY_CONFIGURED = 252
    PTI_ERROR_PC_SAMPLING_ALREADY_STARTED = 253
    PTI_ERROR_PC_SAMPLING_NOT_STARTED = 254
    PTI_ERROR_PC_SAMPLING_ALREADY_STOPPED = 255
    PTI_ERROR_PC_SAMPLING_NOT_STOPPED = 256
    PTI_ERROR_PC_SAMPLING_UNSUPPORTED = 257
    PTI_ERROR_PC_SAMPLING_CONFIGURATION_FAIL = 258
    PTI_RESULT_FORCE_UINT32 = 2147483647
end

function ptiResultTypeToString(result_value)
    @gcsafe_ccall libpti_view.ptiResultTypeToString(result_value::pti_result)::Ptr{Cchar}
end

const pti_device_handle_t = Ptr{Cvoid}

const pti_backend_ctx_t = Ptr{Cvoid}

const pti_backend_queue_t = Ptr{Cvoid}

const pti_backend_evt_t = Ptr{Cvoid}

const pti_backend_command_list_t = Ptr{Cvoid}

const pti_backend_kernel_t = Ptr{Cvoid}

const pti_backend_module_t = Ptr{Cvoid}

struct pti_version
    _major::UInt32
    _minor::UInt32
    _patch::UInt32
end

function ptiVersionString()
    @gcsafe_ccall libpti_view.ptiVersionString()::Ptr{Cchar}
end

function ptiVersion()
    @gcsafe_ccall libpti_view.ptiVersion()::pti_version
end

@cenum _pti_view_kind::UInt32 begin
    PTI_VIEW_INVALID = 0
    PTI_VIEW_DEVICE_GPU_KERNEL = 1
    PTI_VIEW_DEVICE_CPU_KERNEL = 2
    PTI_VIEW_DRIVER_API = 3
    PTI_VIEW_RESERVED = 4
    PTI_VIEW_COLLECTION_OVERHEAD = 5
    PTI_VIEW_RUNTIME_API = 6
    PTI_VIEW_EXTERNAL_CORRELATION = 7
    PTI_VIEW_DEVICE_GPU_MEM_COPY = 8
    PTI_VIEW_DEVICE_GPU_MEM_FILL = 9
    PTI_VIEW_DEVICE_GPU_MEM_COPY_P2P = 10
    PTI_VIEW_DEVICE_SYNCHRONIZATION = 11
    PTI_VIEW_COMMUNICATION = 12
    PTI_VIEW_KIND_FORCE_UINT32 = 2147483647
end

const pti_view_kind = _pti_view_kind

@cenum _pti_view_synchronization_type::UInt32 begin
    PTI_VIEW_SYNCHRONIZATION_TYPE_UNKNOWN = 0
    PTI_VIEW_SYNCHRONIZATION_TYPE_GPU_BARRIER_EXECUTION = 1
    PTI_VIEW_SYNCHRONIZATION_TYPE_GPU_BARRIER_MEMORY = 2
    PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_FENCE = 3
    PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_EVENT = 4
    PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_COMMAND_LIST = 5
    PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_COMMAND_QUEUE = 6
    PTI_VIEW_SYNCHRONIZATION_TYPE_HOST_DEVICE = 7
    PTI_VIEW_SYNCHRONIZATION_TYPE_FORCE_UINT32 = 2147483647
end

const pti_view_synchronization_type = _pti_view_synchronization_type

@cenum _pti_view_memory_type::UInt32 begin
    PTI_VIEW_MEMORY_TYPE_MEMORY = 0
    PTI_VIEW_MEMORY_TYPE_HOST = 1
    PTI_VIEW_MEMORY_TYPE_DEVICE = 2
    PTI_VIEW_MEMORY_TYPE_SHARED = 3
    PTI_VIEW_MEMORY_TYPE_FORCE_UINT32 = 2147483647
end

const pti_view_memory_type = _pti_view_memory_type

@cenum _pti_view_memcpy_type::UInt32 begin
    PTI_VIEW_MEMCPY_TYPE_M2M = 0
    PTI_VIEW_MEMCPY_TYPE_M2H = 1
    PTI_VIEW_MEMCPY_TYPE_M2D = 2
    PTI_VIEW_MEMCPY_TYPE_M2S = 3
    PTI_VIEW_MEMCPY_TYPE_H2M = 4
    PTI_VIEW_MEMCPY_TYPE_H2H = 5
    PTI_VIEW_MEMCPY_TYPE_H2D = 6
    PTI_VIEW_MEMCPY_TYPE_H2S = 7
    PTI_VIEW_MEMCPY_TYPE_D2M = 8
    PTI_VIEW_MEMCPY_TYPE_D2H = 9
    PTI_VIEW_MEMCPY_TYPE_D2D = 10
    PTI_VIEW_MEMCPY_TYPE_D2S = 11
    PTI_VIEW_MEMCPY_TYPE_S2M = 12
    PTI_VIEW_MEMCPY_TYPE_S2H = 13
    PTI_VIEW_MEMCPY_TYPE_S2D = 14
    PTI_VIEW_MEMCPY_TYPE_S2S = 15
    PTI_VIEW_MEMCPY_TYPE_FORCE_UINT32 = 2147483647
end

const pti_view_memcpy_type = _pti_view_memcpy_type

@cenum _pti_view_external_kind::UInt32 begin
    PTI_VIEW_EXTERNAL_KIND_INVALID = 0
    PTI_VIEW_EXTERNAL_KIND_UNKNOWN = 1
    PTI_VIEW_EXTERNAL_KIND_CUSTOM_0 = 2
    PTI_VIEW_EXTERNAL_KIND_CUSTOM_1 = 3
    PTI_VIEW_EXTERNAL_KIND_CUSTOM_2 = 4
    PTI_VIEW_EXTERNAL_KIND_CUSTOM_3 = 5
    PTI_VIEW_EXTERNAL_KIND_FORCE_UINT32 = 2147483647
end

const pti_view_external_kind = _pti_view_external_kind

@cenum _pti_view_overhead_kind::UInt32 begin
    PTI_VIEW_OVERHEAD_KIND_INVALID = 0
    PTI_VIEW_OVERHEAD_KIND_UNKNOWN = 1
    PTI_VIEW_OVERHEAD_KIND_RESOURCE = 2
    PTI_VIEW_OVERHEAD_KIND_BUFFER_FLUSH = 3
    PTI_VIEW_OVERHEAD_KIND_DRIVER = 4
    PTI_VIEW_OVERHEAD_KIND_TIME = 5
    PTI_VIEW_OVERHEAD_KIND_FORCE_UINT32 = 2147483647
end

const pti_view_overhead_kind = _pti_view_overhead_kind

@cenum _pti_api_group_id::UInt32 begin
    PTI_API_GROUP_RESERVED = 0
    PTI_API_GROUP_LEVELZERO = 1
    PTI_API_GROUP_OPENCL = 2
    PTI_API_GROUP_SYCL = 3
    PTI_API_GROUP_HYBRID_SYCL_LEVELZERO = 4
    PTI_API_GROUP_HYBRID_SYCL_OPENCL = 5
    PTI_API_GROUP_ALL = 2147483647
end

const pti_api_group_id = _pti_api_group_id

@cenum _pti_api_class::UInt32 begin
    PTI_API_CLASS_RESERVED = 0
    PTI_API_CLASS_GPU_OPERATION_CORE = 1
    PTI_API_CLASS_HOST_OPERATION_SYNCHRONIZATION = 2
    PTI_API_CLASS_ALL = 2147483647
end

const pti_api_class = _pti_api_class

struct pti_view_record_base
    _view_kind::pti_view_kind
end

struct pti_view_record_kernel
    _view_kind::pti_view_record_base
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _source_file_name::Ptr{Cchar}
    _source_line_number::UInt64
    _kernel_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _pci_address::NTuple{16,Cchar}
    _device_uuid::NTuple{16,UInt8}
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _sycl_task_begin_timestamp::UInt64
    _sycl_enqk_begin_timestamp::UInt64
    _sycl_node_id::UInt64
    _sycl_queue_id::UInt64
    _sycl_invocation_id::UInt32
end

struct pti_view_record_kernel_v2
    _view_kind::pti_view_record_base
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _source_file_name::Ptr{Cchar}
    _source_line_number::UInt64
    _kernel_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _pci_address::NTuple{16,Cchar}
    _device_uuid::NTuple{16,UInt8}
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _sycl_task_begin_timestamp::UInt64
    _sycl_enqk_begin_timestamp::UInt64
    _sycl_node_id::UInt64
    _sycl_queue_id::UInt64
    _sycl_invocation_id::UInt32
    _device_handle::pti_device_handle_t
    _engine_ordinal::UInt32
    _engine_index::UInt32
end

struct pti_view_record_synchronization
    _view_kind::pti_view_record_base
    _synch_type::pti_view_synchronization_type
    _context_handle::pti_backend_ctx_t
    _queue_handle::pti_backend_queue_t
    _event_handle::pti_backend_evt_t
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _thread_id::UInt32
    _correlation_id::UInt32
    _number_wait_events::UInt32
    _return_code::UInt32
    _api_id::UInt32
    _api_group::pti_api_group_id
end

struct pti_view_record_memory_copy
    _view_kind::pti_view_record_base
    _memcpy_type::pti_view_memcpy_type
    _mem_src::pti_view_memory_type
    _mem_dst::pti_view_memory_type
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _pci_address::NTuple{16,Cchar}
    _device_uuid::NTuple{16,UInt8}
    _mem_op_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _bytes::UInt64
    _sycl_queue_id::UInt64
end

struct pti_view_record_memory_copy_v2
    _view_kind::pti_view_record_base
    _memcpy_type::pti_view_memcpy_type
    _mem_src::pti_view_memory_type
    _mem_dst::pti_view_memory_type
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _pci_address::NTuple{16,Cchar}
    _device_uuid::NTuple{16,UInt8}
    _mem_op_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _bytes::UInt64
    _sycl_queue_id::UInt64
    _device_handle::pti_device_handle_t
    _engine_ordinal::UInt32
    _engine_index::UInt32
end

struct pti_view_record_memory_copy_p2p
    _view_kind::pti_view_record_base
    _memcpy_type::pti_view_memcpy_type
    _mem_src::pti_view_memory_type
    _mem_dst::pti_view_memory_type
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _src_pci_address::NTuple{16,Cchar}
    _dst_pci_address::NTuple{16,Cchar}
    _src_uuid::NTuple{16,UInt8}
    _dst_uuid::NTuple{16,UInt8}
    _mem_op_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _bytes::UInt64
    _sycl_queue_id::UInt64
end

struct pti_view_record_memory_copy_p2p_v2
    _view_kind::pti_view_record_base
    _memcpy_type::pti_view_memcpy_type
    _mem_src::pti_view_memory_type
    _mem_dst::pti_view_memory_type
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _src_pci_address::NTuple{16,Cchar}
    _dst_pci_address::NTuple{16,Cchar}
    _src_uuid::NTuple{16,UInt8}
    _dst_uuid::NTuple{16,UInt8}
    _mem_op_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _bytes::UInt64
    _sycl_queue_id::UInt64
    _src_device_handle::pti_device_handle_t
    _dst_device_handle::pti_device_handle_t
    _engine_ordinal::UInt32
    _engine_index::UInt32
end

struct pti_view_record_memory_fill
    _view_kind::pti_view_record_base
    _mem_type::pti_view_memory_type
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _pci_address::NTuple{16,Cchar}
    _device_uuid::NTuple{16,UInt8}
    _mem_op_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _bytes::UInt64
    _value_for_set::UInt64
    _sycl_queue_id::UInt64
end

struct pti_view_record_memory_fill_v2
    _view_kind::pti_view_record_base
    _mem_type::pti_view_memory_type
    _queue_handle::pti_backend_queue_t
    _context_handle::pti_backend_ctx_t
    _name::Ptr{Cchar}
    _pci_address::NTuple{16,Cchar}
    _device_uuid::NTuple{16,UInt8}
    _mem_op_id::UInt64
    _correlation_id::UInt32
    _thread_id::UInt32
    _append_timestamp::UInt64
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _submit_timestamp::UInt64
    _bytes::UInt64
    _value_for_set::UInt64
    _sycl_queue_id::UInt64
    _device_handle::pti_device_handle_t
    _engine_ordinal::UInt32
    _engine_index::UInt32
end

struct pti_view_record_external_correlation
    _view_kind::pti_view_record_base
    _correlation_id::UInt32
    _external_id::UInt64
    _external_kind::pti_view_external_kind
end

struct pti_view_record_overhead
    _view_kind::pti_view_record_base
    _overhead_start_timestamp_ns::UInt64
    _overhead_end_timestamp_ns::UInt64
    _overhead_thread_id::UInt32
    _api_id::UInt32
    _overhead_count::UInt64
    _overhead_duration_ns::UInt64
    _overhead_kind::pti_view_overhead_kind
end

struct pti_view_record_api
    _view_kind::pti_view_record_base
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _api_group::pti_api_group_id
    _api_id::UInt32
    _process_id::UInt32
    _thread_id::UInt32
    _correlation_id::UInt32
    _return_code::UInt32
end

struct pti_view_record_comms
    _view_kind::pti_view_record_base
    _start_timestamp::UInt64
    _end_timestamp::UInt64
    _process_id::UInt32
    _thread_id::UInt32
    _send_size::UInt64
    _recv_size::UInt64
    _communicator_id::UInt64
    _name::Ptr{Cchar}
end

# typedef void ( * pti_fptr_buffer_completed ) ( unsigned char * buffer , size_t buffer_size_in_bytes , size_t used_bytes )
const pti_fptr_buffer_completed = Ptr{Cvoid}

# typedef void ( * pti_fptr_buffer_requested ) ( unsigned char * * buffer_ptr , size_t * buffer_size_in_bytes )
const pti_fptr_buffer_requested = Ptr{Cvoid}

@checked function ptiViewSetCallbacks(fptr_bufferRequested, fptr_bufferCompleted)
    @gcsafe_ccall libpti_view.ptiViewSetCallbacks(fptr_bufferRequested::pti_fptr_buffer_requested,
                                                  fptr_bufferCompleted::pti_fptr_buffer_completed)::pti_result
end

@checked function ptiViewEnable(view_kind)
    @gcsafe_ccall libpti_view.ptiViewEnable(view_kind::pti_view_kind)::pti_result
end

@checked function ptiViewDisable(view_kind)
    @gcsafe_ccall libpti_view.ptiViewDisable(view_kind::pti_view_kind)::pti_result
end

@checked function ptiViewGPULocalAvailable()
    @gcsafe_ccall libpti_view.ptiViewGPULocalAvailable()::pti_result
end

@checked function ptiFlushAllViews()
    @gcsafe_ccall libpti_view.ptiFlushAllViews()::pti_result
end

@checked function ptiViewGetNextRecord(buffer, valid_bytes, record)
    @gcsafe_ccall libpti_view.ptiViewGetNextRecord(buffer::Ptr{UInt8}, valid_bytes::Csize_t,
                                                   record::Ptr{Ptr{pti_view_record_base}})::pti_result
end

@checked function ptiViewPushExternalCorrelationId(external_kind, external_id)
    @gcsafe_ccall libpti_view.ptiViewPushExternalCorrelationId(external_kind::pti_view_external_kind,
                                                               external_id::UInt64)::pti_result
end

@checked function ptiViewPopExternalCorrelationId(external_kind, p_external_id)
    @gcsafe_ccall libpti_view.ptiViewPopExternalCorrelationId(external_kind::pti_view_external_kind,
                                                              p_external_id::Ptr{UInt64})::pti_result
end

function ptiViewOverheadKindToString(type)
    @gcsafe_ccall libpti_view.ptiViewOverheadKindToString(type::pti_view_overhead_kind)::Ptr{Cchar}
end

function ptiViewMemoryTypeToString(type)
    @gcsafe_ccall libpti_view.ptiViewMemoryTypeToString(type::pti_view_memory_type)::Ptr{Cchar}
end

function ptiViewMemcpyTypeToString(type)
    @gcsafe_ccall libpti_view.ptiViewMemcpyTypeToString(type::pti_view_memcpy_type)::Ptr{Cchar}
end

function ptiViewGetTimestamp()
    @gcsafe_ccall libpti_view.ptiViewGetTimestamp()::UInt64
end

# typedef uint64_t ( * pti_fptr_get_timestamp ) ( void )
const pti_fptr_get_timestamp = Ptr{Cvoid}

@checked function ptiViewSetTimestampCallback(fptr_timestampRequested)
    @gcsafe_ccall libpti_view.ptiViewSetTimestampCallback(fptr_timestampRequested::pti_fptr_get_timestamp)::pti_result
end

@checked function ptiViewGetApiIdName(type, unique_id, name)
    @gcsafe_ccall libpti_view.ptiViewGetApiIdName(type::pti_api_group_id, unique_id::UInt32,
                                                  name::Ptr{Ptr{Cchar}})::pti_result
end

@checked function ptiViewEnableDriverApi(enable, api_group_id, api_id)
    @gcsafe_ccall libpti_view.ptiViewEnableDriverApi(enable::UInt32,
                                                     api_group_id::pti_api_group_id,
                                                     api_id::UInt32)::pti_result
end

@checked function ptiViewEnableRuntimeApi(enable, api_group_id, api_id)
    @gcsafe_ccall libpti_view.ptiViewEnableRuntimeApi(enable::UInt32,
                                                      api_group_id::pti_api_group_id,
                                                      api_id::UInt32)::pti_result
end

@checked function ptiViewEnableDriverApiClass(enable, api_class, group)
    @gcsafe_ccall libpti_view.ptiViewEnableDriverApiClass(enable::UInt32,
                                                          api_class::pti_api_class,
                                                          group::pti_api_group_id)::pti_result
end

@checked function ptiViewEnableRuntimeApiClass(enable, api_class, group)
    @gcsafe_ccall libpti_view.ptiViewEnableRuntimeApiClass(enable::UInt32,
                                                           api_class::pti_api_class,
                                                           group::pti_api_group_id)::pti_result
end

const PTI_MAX_DEVICE_UUID_SIZE = 16

# Skipping MacroDefinition: PTI_VERSION 1.2.0

const PTI_VERSION_STRING = "1.2.0"

const PTI_VERSION_MAJOR = 1

const PTI_VERSION_MINOR = 2

const PTI_VERSION_PATCH = 0

const PTI_COMMIT_HASH_STRING = "fdc44f65ccfc4c75f951a8f01b1ef9326223aaaf"

const PTI_MAX_PCI_ADDRESS_SIZE = 16

const PTI_INVALID_QUEUE_ID = 0xffffffffffffffff - 1
