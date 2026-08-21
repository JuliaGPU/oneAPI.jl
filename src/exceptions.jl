# device exceptions
#
# Every kernel receives a `KernelState` (src/device/runtime.jl) that names a host-visible
# flag owned by the (context, device) it was linked for. The device runtime sets the flag
# when a kernel throws; the host reads it whenever it synchronizes a stream and raises a
# `KernelException`.

export KernelException


## exception type

"""
    KernelException

Thrown on the host, at the next synchronization, after a kernel threw an exception on the
device. The device runtime prints the reason (e.g. `ERROR: Out-of-bounds array access.`)
to the standard output of the process at the time of the throw; this exception only carries
the device and whether the work-item ran out of private heap memory along the way.
"""
struct KernelException <: Exception
    dev::ZeDevice
    oom::Bool
end

function Base.showerror(io::IO, err::KernelException)
    name = oneL0.properties(err.dev).name
    print(io, "KernelException: exception thrown during kernel execution on device $name")
    if err.oom
        print(
            io, " (a work-item allocated more than the $(PRIVATE_HEAP_SIZE) bytes of ",
            "private heap memory available to it)"
        )
    end
    return
end


## exception flags

# one 16-byte host USM buffer per (context, device): [1] exception, [2] oom (Int32 each)
const exception_flags = Dict{Tuple{ZeContext, ZeDevice}, oneL0.HostBuffer}()
const exception_flags_lock = ReentrantLock()

function exception_flag(ctx::ZeContext, dev::ZeDevice)
    return Base.@lock exception_flags_lock get!(exception_flags, (ctx, dev)) do
        flag = oneL0.host_alloc(ctx, 16, 16)
        # a pointer embedded in the kernel state is an indirect access as far as Level Zero
        # is concerned, so the buffer needs explicit residency
        oneL0.make_resident(ctx, dev, flag)
        p = convert(Ptr{Int32}, flag)
        unsafe_store!(p, Int32(0), 1)
        unsafe_store!(p, Int32(0), 2)
        flag
    end
end

# the kernel state for kernels linked against `ctx`/`dev`; built once per `HostKernel`.
# The private heap pointer is filled in on the device (`add_private_heap!`).
function kernel_state(ctx::ZeContext, dev::ZeDevice)
    flag = exception_flag(ctx, dev)
    return KernelState(
        reinterpret(LLVMPtr{Int32, AS.CrossWorkgroup}, pointer(flag)),
        C_NULL
    )
end


## host-side check

# Called after a stream has been synchronized. Clears the flag words so the exception is
# reported once; the clear is an atomic swap so two tasks synchronizing the same device
# cannot both report one throw.
function check_exceptions(ctx::ZeContext, dev::ZeDevice)
    flag = Base.@lock exception_flags_lock get(exception_flags, (ctx, dev), nothing)
    flag === nothing && return
    p = convert(Ptr{Int32}, flag)
    unsafe_load(p, 1) == 0 && return
    thrown = Core.Intrinsics.atomic_pointerswap(p, Int32(0), :sequentially_consistent)
    oom = Core.Intrinsics.atomic_pointerswap(p + sizeof(Int32), Int32(0), :sequentially_consistent)
    thrown == 0 && return
    throw(KernelException(dev, oom != 0))
end
