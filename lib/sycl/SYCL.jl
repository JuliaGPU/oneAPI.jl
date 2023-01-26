module SYCL

using ..oneAPI
using ..oneAPI: liboneapi_support

using ..oneAPI.oneL0
using ..oneAPI.oneL0:
  ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t,
  ze_command_queue_handle_t, ze_event_handle_t

include("libsycl.jl")

export syclPlatform, syclDevice, syclContext, syclQueue, syclEvent

mutable struct syclPlatform
  handle::syclPlatform_t

  function syclPlatform(drv::ZeDriver)
    handle = Ref{syclPlatform_t}()
    syclPlatformCreate(handle, drv)
    obj = new(handle[])
    finalizer(obj) do sycl_platform
      syclPlatformDestroy(sycl_platform)
    end
  end
end

Base.unsafe_convert(::Type{syclPlatform_t}, sycl_platform::syclPlatform) =
  sycl_platform.handle

mutable struct syclDevice
  handle::syclDevice_t
  ze_dev::ZeDevice

  function syclDevice(platform::syclPlatform, ze_dev::ZeDevice)
    handle = Ref{syclDevice_t}()
    syclDeviceCreate(handle, platform, ze_dev)
    obj = new(handle[], ze_dev)
    finalizer(obj) do dev
      syclDeviceDestroy(dev)
    end
  end
end

Base.unsafe_convert(::Type{syclDevice_t}, dev::syclDevice) =
  dev.handle

mutable struct syclContext
  handle::syclContext_t
  devs::Vector{syclDevice}
  ze_ctx::ZeContext

  function syclContext(devs::Vector{syclDevice}, ze_ctx::ZeContext)
    handle = Ref{syclContext_t}()
    syclContextCreate(handle, devs, length(devs), ze_ctx, true)
    obj = new(handle[], devs, ze_ctx)
    finalizer(obj) do ctx
      onemklDestroy()
      syclContextDestroy(ctx)
    end
  end
end

Base.unsafe_convert(::Type{syclContext_t}, ctx::syclContext) =
  ctx.handle

mutable struct syclQueue
  handle::syclQueue_t
  ctx::syclContext
  dev::syclDevice
  ze_queue::ZeCommandQueue

  function syclQueue(ctx::syclContext, dev::syclDevice, ze_queue::ZeCommandQueue)
    handle = Ref{syclQueue_t}()
    syclQueueCreate(handle, ctx, dev, ze_queue, true)
    obj = new(handle[], ctx, dev, ze_queue)
    finalizer(obj) do queue
      syclQueueDestroy(queue)
    end
  end
end

Base.unsafe_convert(::Type{syclQueue_t}, queue::syclQueue) =
  queue.handle

mutable struct syclEvent
  handle::syclEvent_t
  ctx::syclContext
  ze_event::ZeEvent

  function syclEvent(ctx::syclContext, ze_event::ZeEvent)
    handle = Ref{syclEvent_t}()
    syclEventCreate(handle, ctx, ze_event, true)
    obj = new(handle[], ctx, ze_event)
    finalizer(obj) do event
      syclEventDestroy(event)
    end
  end
end

Base.unsafe_convert(::Type{syclEvent_t}, event::syclEvent) =
  event.handle

end
