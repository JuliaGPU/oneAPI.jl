using CEnum

mutable struct syclPlatform_st end

const syclPlatform_t = Ptr{syclPlatform_st}

function syclPlatformCreate(obj, driver)
    @ccall liboneapilib.syclPlatformCreate(obj::Ptr{syclPlatform_t},
                                           driver::ze_driver_handle_t)::Cint
end

function syclPlatformDestroy(obj)
    @ccall liboneapilib.syclPlatformDestroy(obj::syclPlatform_t)::Cint
end

mutable struct syclDevice_st end

const syclDevice_t = Ptr{syclDevice_st}

function syclDeviceCreate(obj, platform, device)
    @ccall liboneapilib.syclDeviceCreate(obj::Ptr{syclDevice_t}, platform::syclPlatform_t,
                                         device::ze_device_handle_t)::Cint
end

function syclDeviceDestroy(obj)
    @ccall liboneapilib.syclDeviceDestroy(obj::syclDevice_t)::Cint
end

mutable struct syclContext_st end

const syclContext_t = Ptr{syclContext_st}

function syclContextCreate(obj, devices, ndevices, context, keep_ownership)
    @ccall liboneapilib.syclContextCreate(obj::Ptr{syclContext_t},
                                          devices::Ptr{syclDevice_t}, ndevices::Csize_t,
                                          context::ze_context_handle_t,
                                          keep_ownership::Cint)::Cint
end

function syclContextDestroy(obj)
    @ccall liboneapilib.syclContextDestroy(obj::syclContext_t)::Cint
end

mutable struct syclQueue_st end

const syclQueue_t = Ptr{syclQueue_st}

function syclQueueCreate(obj, context, queue, keep_ownership)
    @ccall liboneapilib.syclQueueCreate(obj::Ptr{syclQueue_t}, context::syclContext_t,
                                        queue::ze_command_queue_handle_t,
                                        keep_ownership::Cint)::Cint
end

function syclQueueDestroy(obj)
    @ccall liboneapilib.syclQueueDestroy(obj::syclQueue_t)::Cint
end

mutable struct syclEvent_st end

const syclEvent_t = Ptr{syclEvent_st}

function syclEventCreate(obj, context, event, keep_ownership)
    @ccall liboneapilib.syclEventCreate(obj::Ptr{syclEvent_t}, context::syclContext_t,
                                        event::ze_event_handle_t,
                                        keep_ownership::Cint)::Cint
end

function syclEventDestroy(obj)
    @ccall liboneapilib.syclEventDestroy(obj::syclEvent_t)::Cint
end
