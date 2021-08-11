#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include <level_zero/ze_api.h>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>

// https://github.com/intel/llvm/blob/sycl/sycl/include/CL/sycl/backend/level_zero.hpp

jlcxx::BoxedValue<sycl::platform> syclMakePlatform(void *handle) {
    auto platform =
        sycl::level_zero::make<sycl::platform>((ze_driver_handle_t)handle);
    return jlcxx::create<sycl::platform>(platform);
}

jlcxx::BoxedValue<sycl::device> syclMakeDevice(sycl::platform platform,
                                               void *handle) {
    auto device = sycl::level_zero::make<sycl::device>(
        platform, (ze_device_handle_t)handle);
    return jlcxx::create<sycl::device>(device);
}

jlcxx::BoxedValue<sycl::context>
syclMakeContext(const std::vector<sycl::device> &devices, void *handle,
                bool keep_ownership) {
    auto ownership = keep_ownership ? sycl::level_zero::ownership::keep
                                    : sycl::level_zero::ownership::transfer;
    auto context = sycl::level_zero::make<sycl::context>(
        devices, (ze_context_handle_t)handle, ownership);
    return jlcxx::create<sycl::context>(context);
}

jlcxx::BoxedValue<sycl::queue>
syclMakeQueue(sycl::context context, void *handle, bool keep_ownership) {
    auto ownership = keep_ownership ? sycl::level_zero::ownership::keep
                                    : sycl::level_zero::ownership::transfer;
    // XXX: ownership argument only used on master
    auto queue = sycl::level_zero::make<sycl::queue>(
        context, (ze_command_queue_handle_t)handle);
    return jlcxx::create<sycl::queue>(queue);
}

// XXX: make_event only available on master
// jlcxx::BoxedValue<sycl::event> syclMakeEvent(sycl::context context, void
// *handle, bool keep_ownership) {
//    auto ownership = keep_ownership ? sycl::level_zero::ownership::keep :
//    sycl::level_zero::ownership::transfer; auto event =
//    sycl::level_zero::make<sycl::event>(context, (ze_event_handle_t)handle,
//    ownership); return jlcxx::create<sycl::event>(event);
//}

JLCXX_MODULE define_module_sycl(jlcxx::Module &mod) {
    mod.add_type<sycl::platform>("Platform");
    mod.add_type<sycl::context>("Context");
    mod.add_type<sycl::device>("Device");
    mod.add_type<sycl::queue>("Queue");

    mod.method("syclMakePlatform", syclMakePlatform);
    mod.method("syclMakeDevice", syclMakeDevice);
    mod.method("syclMakeContext", syclMakeContext);
    mod.method("syclMakeQueue", syclMakeQueue);
    // mod.method("syclMakeEvent", syclMakeEvent);
}
