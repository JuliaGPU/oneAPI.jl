#include "sycl.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

// https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/oneapi/backend/level_zero.hpp

extern "C" int syclPlatformCreate(syclPlatform_t *obj,
                                  ze_driver_handle_t driver) {
    auto sycl_platform =
        sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(driver);
    *obj = new syclPlatform_st({sycl_platform});
    return 0;
}

extern "C" int syclPlatformDestroy(syclPlatform_t obj) {
    delete obj;
    return 0;
}

extern "C" int syclDeviceCreate(syclDevice_t *obj, syclPlatform_t platform,
                                ze_device_handle_t device) {
    auto sycl_device =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(device);
    *obj = new syclDevice_st({sycl_device});
    return 0;
}

extern "C" int syclDeviceDestroy(syclDevice_t obj) {
    delete obj;
    return 0;
}

extern "C" int syclContextCreate(syclContext_t *obj, syclDevice_t *devices,
                                 size_t ndevices, ze_context_handle_t context,
                                 int keep_ownership) {
    std::vector<sycl::device> sycl_devices(ndevices);
    for (size_t i = 0; i < ndevices; i++)
        sycl_devices[i] = devices[i]->val;
    auto sycl_ownership =
        keep_ownership ? sycl::ext::oneapi::level_zero::ownership::keep
                       : sycl::ext::oneapi::level_zero::ownership::transfer;
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>
        sycl_context_input = {context, sycl_devices, sycl_ownership};

    auto sycl_context =
        sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
            sycl_context_input);
    *obj = new syclContext_st({sycl_context});
    return 0;
}

extern "C" int syclContextDestroy(syclContext_t obj) {
    delete obj;
    return 0;
}

extern "C" int syclQueueCreate(syclQueue_t *obj, syclContext_t context,
                               syclDevice_t device,
                               ze_command_queue_handle_t queue,
                               int keep_ownership) {
    auto sycl_ownership =
        keep_ownership ? sycl::ext::oneapi::level_zero::ownership::keep
                       : sycl::ext::oneapi::level_zero::ownership::transfer;
    auto sycl_queue_input =
        sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero,
                              sycl::queue>{queue, device->val, sycl_ownership};

    auto sycl_queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(
        sycl_queue_input, context->val);
    *obj = new syclQueue_st({sycl_queue});
    return 0;
}

extern "C" int syclQueueDestroy(syclQueue_t obj) {
    delete obj;
    return 0;
}

extern "C" int syclQueueWait(syclQueue_t obj) {
    obj->val.wait();
    return 0;
}

extern "C" int syclEventCreate(syclEvent_t *obj, syclContext_t context,
                               ze_event_handle_t event, int keep_ownership) {
    auto sycl_ownership =
        keep_ownership ? sycl::ext::oneapi::level_zero::ownership::keep
                       : sycl::ext::oneapi::level_zero::ownership::transfer;
    auto sycl_event_input =
        sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero,
                              sycl::event>{event, sycl_ownership};

    auto sycl_event = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        sycl_event_input, context->val);
    *obj = new syclEvent_st({sycl_event});
    return 0;
}

extern "C" int syclEventDestroy(syclEvent_t obj) {
   delete obj;
   return 0;
}
