mutable struct syclPlatform_st end

const syclPlatform_t = Ptr{syclPlatform_st}

function syclPlatformCreate(obj, driver)
    ccall((:syclPlatformCreate, liboneapilib), Cint,
          (Ptr{syclPlatform_t}, ze_driver_handle_t),
          obj, driver)
end

function syclPlatformDestroy(obj)
    ccall((:syclPlatformDestroy, liboneapilib), Cint,
          (syclPlatform_t,),
          obj)
end

mutable struct syclDevice_st end

const syclDevice_t = Ptr{syclDevice_st}

function syclDeviceCreate(obj, platform, device)
    ccall((:syclDeviceCreate, liboneapilib), Cint,
          (Ptr{syclDevice_t}, syclPlatform_t, ze_device_handle_t),
          obj, platform, device)
end

function syclDeviceDestroy(obj)
    ccall((:syclDeviceDestroy, liboneapilib), Cint,
          (syclDevice_t,),
          obj)
end

mutable struct syclContext_st end

const syclContext_t = Ptr{syclContext_st}

function syclContextCreate(obj, devices, ndevices, context, keep_ownership)
    ccall((:syclContextCreate, liboneapilib), Cint,
          (Ptr{syclContext_t}, Ptr{syclDevice_t}, Csize_t, ze_context_handle_t, Cint),
          obj, devices, ndevices, context, keep_ownership)
end

function syclContextDestroy(obj)
    ccall((:syclContextDestroy, liboneapilib), Cint,
          (syclContext_t,),
          obj)
end

mutable struct syclQueue_st end

const syclQueue_t = Ptr{syclQueue_st}

function syclQueueCreate(obj, context, queue, keep_ownership)
    ccall((:syclQueueCreate, liboneapilib), Cint,
          (Ptr{syclQueue_t}, syclContext_t, ze_command_queue_handle_t, Cint),
          obj, context, queue, keep_ownership)
end

function syclQueueDestroy(obj)
    ccall((:syclQueueDestroy, liboneapilib), Cint,
          (syclQueue_t,),
          obj)
end

mutable struct syclEvent_st end

const syclEvent_t = Ptr{syclEvent_st}

function syclEventCreate(obj, context, event, keep_ownership)
    ccall((:syclEventCreate, liboneapilib), Cint,
          (Ptr{syclEvent_t}, syclContext_t, ze_event_handle_t, Cint),
          obj, context, event, keep_ownership)
end

function syclEventDestroy(obj)
    ccall((:syclEventDestroy, liboneapilib), Cint,
          (syclEvent_t,),
          obj)
end
