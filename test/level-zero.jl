@testset "oneL0" begin

using oneAPI.oneL0


@testset "driver" begin

drvs = drivers()
@assert !isempty(drvs)
drv = first(drvs)

api_version(drv)

end

drv = first(drivers())


@testset "device" begin

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

end

dev = first(devices(drv))


@testset "command" begin

queue = ZeCommandQueue(dev)

list = ZeCommandList(dev)
close(list)
execute!(queue, [list])
synchronize(queue)
reset(list)

list = ZeCommandList(dev) do list
    @test list isa ZeCommandList
end

execute!(queue) do list
    @test list isa ZeCommandList
end

end

queue = ZeCommandQueue(dev)


@testset "event" begin

ZeEventPool(drv, 1)
ZeEventPool(drv, 1, dev)

pool = ZeEventPool(drv, 1)

event = pool[1]
@test !query(event)

signal(event)
ZeCommandList(dev) do list
    append_signal!(list, event)
end
@test query(event)

wait(event, 1)
ZeCommandList(dev) do list
    append_wait!(list, event)
end

reset(event)
ZeCommandList(dev) do list
    append_reset!(list, event)
end

timed_pool = ZeEventPool(drv, 1; flags=oneL0.ZE_EVENT_POOL_FLAG_TIMESTAMP)
timed_event = timed_pool[1]
@test global_time(timed_event).start == nothing
@test context_time(timed_event).start == nothing
signal(timed_event)
@test global_time(timed_event).start != nothing
@test context_time(timed_event).start != nothing

end


@testset "barrier" begin

pool = ZeEventPool(drv, 1)
event = pool[1]

ZeCommandList(dev) do list
    append_barrier!(list)
    append_barrier!(list, event)
    append_barrier!(list, event, event)
end

#device_barrier(dev)    # unsupported

end


@testset "module" begin

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
attrs[oneL0.ZE_KERNEL_ATTR_INDIRECT_HOST_ACCESS] = true
@test attrs[oneL0.ZE_KERNEL_ATTR_INDIRECT_HOST_ACCESS]

props = properties(kernel)
@test props.numKernelArgs == 1
@test props.name == "bar"
@test props.requiredGroupSize isa oneL0.ZeDim3


@testset "kernel execution" begin

ZeCommandList(dev) do list
    append_launch!(list, kernel, 1)
end

pool = ZeEventPool(drv, 2)
signal_event = pool[1]
wait_event = pool[2]

execute!(queue) do list
    append_launch!(list, kernel, 1, signal_event, wait_event)
end
@test !query(signal_event)

signal(wait_event)
synchronize(queue)
@test query(signal_event)

end

end


@testset "memory" begin

buf = device_alloc(dev, 1024)
props = properties(buf)
@test props.device == dev
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


@testset "copy" begin

let src = rand(Int, 1024)
    chk = ones(Int, length(src))

    dst = device_alloc(dev, sizeof(src))

    execute!(queue) do list
        append_copy!(list, pointer(dst), pointer(src), sizeof(src))
        append_barrier!(list)
        append_copy!(list, pointer(chk), pointer(dst), sizeof(src))
    end
    synchronize(queue)
    @test chk == src

    execute!(queue) do list
        pattern = [42]
        append_fill!(list, pointer(dst), pointer(pattern), sizeof(pattern), sizeof(src))
        append_barrier!(list)
        append_copy!(list, pointer(chk), pointer(dst), sizeof(src))
    end
    synchronize(queue)
    @test all(isequal(42), chk)

    free(dst)
end

let buf = shared_alloc(drv, dev, 1024)

    execute!(queue) do list
        append_prefetch!(list, pointer(buf), sizeof(buf))

        append_advise!(list, dev, pointer(buf), sizeof(buf),
                       oneL0.ZE_MEMORY_ADVICE_SET_READ_MOSTLY)
    end

    free(buf)
end

end


end
