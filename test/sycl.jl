if Sys.iswindows()
@warn "Skipping unsupported SYCL tests"
else

using oneAPI.oneL0, oneAPI.SYCL

@test sycl_platform() isa syclPlatform

ze_dev = device()
sycl_dev = sycl_device(ze_dev)
@test sycl_dev isa syclDevice

ze_ctx = context()
sycl_ctx = sycl_context(ze_ctx, ze_dev)
@test sycl_ctx isa syclContext

ze_queue = ZeCommandQueue(ze_ctx, ze_dev)
@test sycl_queue(ze_queue) isa syclQueue

ze_event_pool = ZeEventPool(ze_ctx, 1, ze_dev)
ze_event = ze_event_pool[1]
sycl_event = oneAPI.SYCL.syclEvent(sycl_ctx, ze_event)

end