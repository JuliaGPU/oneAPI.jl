@testset "oneL0" begin

using oneAPI.oneL0

## driver

drvs = drivers()
@assert !isempty(drvs)
drv = first(drvs)

api_version(drv)


## device

devs = devices(drv)
@assert !isempty(devs)
dev = first(devs)

properties(dev)
compute_properties(dev)
kernel_properties(dev)
memory_properties(dev)
memory_access_properties(dev)
cache_properties(dev)
image_properties(dev)
p2p_properties(dev, dev)


## command

queue = ZeCommandQueue(dev)

list = ZeCommandList(dev)
close(list)
execute!(queue, [list])
synchronize(queue)
reset(list)


## event

ZeEventPool(drv, 1)
ZeEventPool(drv, 1, dev)

pool = ZeEventPool(drv, 1)

event = pool[1]
@test !query(event)

signal(event)
append_signal!(list, event)
@test query(event)

wait(event, 1)
append_wait!(list, event)

reset(event)
append_reset!(list, event)

timed_pool = ZeEventPool(drv, 1; flags=oneL0.ZE_EVENT_POOL_FLAG_TIMESTAMP)
timed_event = timed_pool[1]
@test global_time(timed_event).start == nothing
@test context_time(timed_event).start == nothing
signal(timed_event)
@test global_time(timed_event).start != nothing
@test context_time(timed_event).start != nothing


## barrier

append_barrier!(list)
append_barrier!(list, event)
append_barrier!(list, event, event)

#device_barrier(dev)    # unsupported


## module

data = read(joinpath(@__DIR__, "dummy.spv"))
mod = ZeModule(dev, data)

@test length(kernels(mod)) == 2
@test haskey(kernels(mod), "foo")
@test !haskey(kernels(mod), "baz")
kernel = kernels(mod)["foo"]

suggest_groupsize(kernel, 1024)
groupsize!(kernel, 1)
groupsize!(kernel, (1,))
groupsize!(kernel, (1, 1))
groupsize!(kernel, (1, 1, 1))

kernel = kernels(mod)["bar"]
arguments(kernel)[1] = Int32(42)

attrs = attributes(kernel)
@test !attrs[oneL0.ZE_KERNEL_ATTR_INDIRECT_HOST_ACCESS]
@test !attrs[oneL0.ZE_KERNEL_ATTR_INDIRECT_SHARED_ACCESS]
@test isempty(attrs[oneL0.ZE_KERNEL_ATTR_SOURCE_ATTRIBUTE])

props = properties(kernel)
@test props.numKernelArgs == 1
@test props.name == "bar"
@test props.requiredGroupSize isa oneL0.ZeDim3

# kernel execution

append_launch!(list, kernel, 1)

queue = ZeCommandQueue(dev)
list = ZeCommandList(dev)

pool = ZeEventPool(drv, 2)
signal_event = pool[1]
wait_event = pool[2]

append_launch!(list, kernel, 1, signal_event, wait_event)
close(list)
execute!(queue, [list])
@test !query(signal_event)

signal(wait_event)
synchronize(queue)
@test query(signal_event)


## memory

buf = device_alloc(dev, 1024)
props = properties(buf)
@test props.type == oneL0.ZE_MEMORY_TYPE_DEVICE
@test_throws ArgumentError convert(Ptr{Cvoid}, buf)
ptr = convert(ZePtr{Cvoid}, buf)
@test lookup_alloc(drv, ptr) isa typeof(buf)
free(buf)

buf = host_alloc(drv, 1024)
props = properties(buf)
@test props.type == oneL0.ZE_MEMORY_TYPE_HOST
ptr = convert(ZePtr{Cvoid}, buf)
@test lookup_alloc(drv, ptr) isa typeof(buf)
ptr = convert(Ptr{Cvoid}, buf)
@test lookup_alloc(drv, ptr) isa typeof(buf)
free(buf)

buf = shared_alloc(drv, dev, 1024)
props = properties(buf)
@test props.type == oneL0.ZE_MEMORY_TYPE_SHARED
ptr = convert(ZePtr{Cvoid}, buf)
@test lookup_alloc(drv, ptr) isa typeof(buf)
ptr = convert(Ptr{Cvoid}, buf)
@test lookup_alloc(drv, ptr) isa typeof(buf)
free(buf)

end
