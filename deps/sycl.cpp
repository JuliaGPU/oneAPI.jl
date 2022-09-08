#include "sycl.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

// https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/oneapi/backend/level_zero.hpp

extern "C" int syclPlatformCreate(syclPlatform_t *obj,
                                  ze_driver_handle_t driver) {
    auto sycl_platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle) driver);
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
        sycl::ext::oneapi::level_zero::make_device(platform->val, (pi_native_handle) device);
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
    
    auto sycl_context =
        sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle) context, keep_ownership);
    *obj = new syclContext_st({sycl_context});
    return 0;
}

extern "C" int syclContextDestroy(syclContext_t obj) {
    delete obj;
    return 0;
}

extern "C" int syclQueueCreate(syclQueue_t *obj, syclContext_t context,
                               ze_command_queue_handle_t queue,
                               int keep_ownership) {
    // XXX: ownership argument only used on master
    auto sycl_queue = sycl::ext::oneapi::level_zero::make_queue(context->val, (pi_native_handle) queue, keep_ownership);
    *obj = new syclQueue_st({sycl_queue});
    return 0;
}

extern "C" int syclQueueDestroy(syclQueue_t obj) {
    delete obj;
    return 0;
}

// XXX: make_event only available on master
// extern "C" int syclEventCreate(syclEvent_t *obj, syclContext_t context,
//                               ze_event_handle_t event, int keep_ownership) {
//    auto ownership = keep_ownership ? sycl::level_zero::ownership::keep
//                                    : sycl::level_zero::ownership::transfer;
//    auto sycl_event = sycl::level_zero::make<sycl::event>(
//        context->val, ze_event_handle_t event, ownership);
//    *obj = new syclEvent_st({sycl_event});
//    return 0;
//}
//
// extern "C" int syclEventDestroy(syclEvent_t obj) {
//    delete obj;
//    return 0;
//}
