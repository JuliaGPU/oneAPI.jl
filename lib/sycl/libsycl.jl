using CEnum

mutable struct syclPlatform_st end

const syclPlatform_t = Ptr{syclPlatform_st}

function syclPlatformCreate(obj, driver)
    @ccall liboneapi_support.syclPlatformCreate(obj::Ptr{syclPlatform_t},
                                           driver::ze_driver_handle_t)::Cint
end

function syclPlatformDestroy(obj)
    @ccall liboneapi_support.syclPlatformDestroy(obj::syclPlatform_t)::Cint
end

mutable struct syclDevice_st end

const syclDevice_t = Ptr{syclDevice_st}

function syclDeviceCreate(obj, platform, device)
    @ccall liboneapi_support.syclDeviceCreate(obj::Ptr{syclDevice_t}, platform::syclPlatform_t,
                                         device::ze_device_handle_t)::Cint
end

function syclDeviceDestroy(obj)
    @ccall liboneapi_support.syclDeviceDestroy(obj::syclDevice_t)::Cint
end

mutable struct syclContext_st end

const syclContext_t = Ptr{syclContext_st}

function syclContextCreate(obj, devices, ndevices, context, keep_ownership)
    @ccall liboneapi_support.syclContextCreate(obj::Ptr{syclContext_t},
                                          devices::Ptr{syclDevice_t}, ndevices::Csize_t,
                                          context::ze_context_handle_t,
                                          keep_ownership::Cint)::Cint
end

function syclContextDestroy(obj)
    @ccall liboneapi_support.syclContextDestroy(obj::syclContext_t)::Cint
end

mutable struct syclQueue_st end

const syclQueue_t = Ptr{syclQueue_st}

function syclQueueCreate(obj, context, device, queue, keep_ownership)
    @ccall liboneapi_support.syclQueueCreate(obj::Ptr{syclQueue_t}, context::syclContext_t,
                                        device::syclDevice_t,
                                        queue::ze_command_queue_handle_t,
                                        keep_ownership::Cint)::Cint
end

function syclQueueDestroy(obj)
    @ccall liboneapi_support.syclQueueDestroy(obj::syclQueue_t)::Cint
end

mutable struct syclEvent_st end

const syclEvent_t = Ptr{syclEvent_st}

function syclEventCreate(obj, context, event, keep_ownership)
    @ccall liboneapi_support.syclEventCreate(obj::Ptr{syclEvent_t}, context::syclContext_t,
                                        event::ze_event_handle_t,
                                        keep_ownership::Cint)::Cint
end

function syclEventDestroy(obj)
    @ccall liboneapi_support.syclEventDestroy(obj::syclEvent_t)::Cint
end

function onemklDestroy()
    @ccall liboneapi_support.onemklDestroy()::Cvoid
end
