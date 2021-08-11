module SYCL

using CxxWrap

using ..oneAPI.oneL0

@wrapmodule(joinpath(@__DIR__, "../../deps/liboneapilib.so"), :define_module_sycl)

# XXX: handles are passed as void* to work around JuliaInterop/CxxWrap.jl#302
syclMakePlatform(drv::ZeDriver) = syclMakePlatform(
convert(Ptr{Cvoid}, Base.unsafe_convert(oneL0.ze_driver_handle_t, drv)))
syclMakeDevice(platform, dev::ZeDevice) = syclMakeDevice(platform,
convert(Ptr{Cvoid}, Base.unsafe_convert(oneL0.ze_device_handle_t, dev)))
syclMakeContext(devs::Vector, ctx::ZeContext) = syclMakeContext(StdVector(devs),
convert(Ptr{Cvoid}, Base.unsafe_convert(oneL0.ze_context_handle_t, ctx)), true)
syclMakeQueue(ctx, queue::ZeCommandQueue) = syclMakeQueue(ctx,
convert(Ptr{Cvoid}, Base.unsafe_convert(oneL0.ze_command_queue_handle_t, queue)), true)

function __init__()
  @initcxx
end

end
