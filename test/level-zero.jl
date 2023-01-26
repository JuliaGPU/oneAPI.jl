using oneAPI.oneL0


@testset "driver" begin

drvs = drivers()
@assert !isempty(drvs)
drv = first(drvs)
@test drv == drvs[1]
show(devnull, drv)
show(devnull, MIME("text/plain"), drv)

api_version(drv)

properties(drv)
ipc_properties(drv)
#extension_properties(drv)

end

drv = first(drivers())


@testset "device" begin

devs = devices(drv)
@assert !isempty(devs)
dev = first(devs)
@test dev == devs[1]
show(devnull, dev)
show(devnull, MIME("text/plain"), dev)

@test collect(devices()) == collect(devices(drv))
@test device!(dev) == dev

properties(dev)
compute_properties(dev)
module_properties(dev)
memory_properties(dev)
memory_access_properties(dev)
cache_properties(dev)
image_properties(dev)
p2p_properties(dev, dev)

end

dev = first(devices(drv))


@testset "context" begin

ctx = ZeContext(drv)
show(devnull, ctx)

#status(ctx)

end

ctx = ZeContext(drv)


@testset "command" begin

groups = command_queue_groups(dev)
@test !isempty(groups)

groups = compute_groups(dev)
group = first(groups)

queue = ZeCommandQueue(ctx, dev, group.ordinal)

list = ZeCommandList(ctx, dev, group.ordinal)
close(list)
execute!(queue, [list])
synchronize(queue)
reset(list)

list = ZeCommandList(ctx, dev, group.ordinal) do list
    @test list isa ZeCommandList
end

execute!(queue) do list
    @test list isa ZeCommandList
end

end

group = first(compute_groups(dev))
queue = ZeCommandQueue(ctx, dev, group.ordinal)


@testset "fence" begin

fence = ZeFence(queue)

@test !Base.isdone(fence)

execute!(queue, fence) do list
    # do nothing, but signal the fence on completion
end

wait(fence)
@test Base.isdone(fence)

reset(fence)
@test !Base.isdone(fence)

end


@testset "event" begin

ZeEventPool(ctx, 1)
ZeEventPool(ctx, 1, dev)

pool = ZeEventPool(ctx, 1)

event = pool[1]
@test !Base.isdone(event)

signal(event)
ZeCommandList(ctx, dev, group.ordinal) do list
    append_signal!(list, event)
end
@test Base.isdone(event)

wait(event, 1)
ZeCommandList(ctx, dev, group.ordinal) do list
    append_wait!(list, event)
end

reset(event)
ZeCommandList(ctx, dev, group.ordinal) do list
    append_reset!(list, event)
end

# timed_pool = ZeEventPool(ctx, 1; flags=oneL0.ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP)
# timed_event = timed_pool[1]
# @test kernel_timestamp(timed_event).global.start == nothing
# @test kernel_timestamp(timed_event).context.start == nothing
# signal(timed_event) # FIXME: A kernel timestamp event can only be signaled from zeCommandListAppendLaunchKernel et al. functions
# @test kernel_timestamp(timed_event).global.start != nothing
# @test kernel_timestamp(timed_event).context.start != nothing

end


@testset "barrier" begin

pool = ZeEventPool(ctx, 1)
event = pool[1]

ZeCommandList(ctx, dev, group.ordinal) do list
    append_barrier!(list)
    append_barrier!(list, event)
    append_barrier!(list, event, event)
end

#device_barrier(dev)    # unsupported

end


@testset "module" begin

data = read(joinpath(@__DIR__, "dummy.spv"))
mod = ZeModule(ctx, dev, data)

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

@test indirect_access(kernel) == 0
indirect_access!(kernel, oneL0.ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE)
@test indirect_access(kernel) == oneL0.ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE

# oneapi-src/level-zero#55
if !parse(Bool, get(ENV, "ZE_ENABLE_PARAMETER_VALIDATION", "false"))
    attrs = source_attributes(kernel)
    @test isempty(attrs)
end

props = properties(kernel)
@test props.numKernelArgs == 1
@test props.requiredGroupSize isa oneL0.ZeDim3


@testset "kernel execution" begin

ZeCommandList(ctx, dev, group.ordinal) do list
    append_launch!(list, kernel, 1)
end

pool = ZeEventPool(ctx, 2)
signal_event = pool[1]
wait_event = pool[2]

execute!(queue) do list
    append_launch!(list, kernel, 1, signal_event, wait_event)
end
@test !Base.isdone(signal_event)

signal(wait_event)
synchronize(queue)
@test Base.isdone(signal_event)

end

end


@testset "memory" begin

buf = device_alloc(ctx, dev, 1024)
props = properties(buf)
@test props.device == dev
@test props.type == oneL0.ZE_MEMORY_TYPE_DEVICE
@test_throws ArgumentError convert(Ptr{Cvoid}, buf)
ptr = convert(ZePtr{Cvoid}, buf)
@test lookup_alloc(ctx, ptr) isa typeof(buf)
free(buf)

buf = host_alloc(ctx, 1024)
props = properties(buf)
@test props.type == oneL0.ZE_MEMORY_TYPE_HOST
ptr = convert(ZePtr{Cvoid}, buf)
@test lookup_alloc(ctx, ptr) isa typeof(buf)
ptr = convert(Ptr{Cvoid}, buf)
@test lookup_alloc(ctx, ptr) isa typeof(buf)
free(buf)

buf = shared_alloc(ctx, dev, 1024)
props = properties(buf)
@test props.type == oneL0.ZE_MEMORY_TYPE_SHARED
ptr = convert(ZePtr{Cvoid}, buf)
@test lookup_alloc(ctx, ptr) isa typeof(buf)
ptr = convert(Ptr{Cvoid}, buf)
@test lookup_alloc(ctx, ptr) isa typeof(buf)
free(buf)

end


@testset "copy" begin

let src = rand(Int, 1024)
    chk = ones(Int, length(src))

    dst = device_alloc(ctx, dev, sizeof(src))

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

for buf in [device_alloc(ctx, dev, 1024),
            host_alloc(ctx, 1024),
            shared_alloc(ctx, dev, 1024)]
    execute!(queue) do list
        append_prefetch!(list, pointer(buf), sizeof(buf))

        append_advise!(list, dev, pointer(buf), sizeof(buf),
                       oneL0.ZE_MEMORY_ADVICE_SET_READ_MOSTLY)
    end

    free(buf)
end

end



@testset "residency" begin

for buf in [device_alloc(ctx, dev, 1024),
            host_alloc(ctx, 1024),
            shared_alloc(ctx, dev, 1024)]
    make_resident(ctx, dev, buf)
    evict(ctx, dev, buf)
    make_resident(ctx, dev, buf, 1024)
    evict(ctx, dev, buf, 1024)
    free(buf)
end

end
