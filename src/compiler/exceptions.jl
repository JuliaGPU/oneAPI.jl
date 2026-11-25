# reporting of exceptions thrown by kernels

export KernelException

"""
    KernelException

An exception thrown by a kernel on device `dev`, reported when synchronizing the stream it
was launched on (e.g. by `synchronize()`, or a copy to the host such as `Array(x)`).

The position of the work-item that threw is always known (`work_item` is its local id,
`work_group` the id of its work-group). What else is known depends on the debug level the
kernel was compiled with (Julia's `-g` level, or `@oneapi debug_level=...`):

- `0`: only that an exception was thrown;
- `1`: also the `name` of its type and the `reason`, for exceptions thrown by Julia's
  runtime (e.g. bounds errors, or domain errors in math functions);
- `2`: also the `backtrace` of the throw, as `(function, file, line)` tuples.

Fields that were not recorded are empty, or zero. Only one exception is reported per
synchronization, and the other work-items of the kernel keep running.
"""
struct KernelException <: Exception
    dev::ZeDevice
    name::String
    reason::String
    work_item::NTuple{3, Int}
    work_group::NTuple{3, Int}
    backtrace::Vector{Tuple{String, String, Int}}
end

function Base.showerror(io::IO, err::KernelException)
    name = isempty(err.name) ? "exception" : err.name
    article = first(uppercase(name)) in ('A', 'E', 'I', 'O', 'U') ? "An" : "A"
    print(io, "KernelException: $article $name was thrown")
    # positions are 1-based, so zero coordinates mean they were not recorded
    if err.work_item != (0, 0, 0)
        print(io, " by work-item ", join(err.work_item, '×'),
              " in work-group ", join(err.work_group, '×'))
    end
    print(io, " on ", oneL0.properties(err.dev).name)
    isempty(err.reason) || print(io, ": ", err.reason)
    if isempty(err.backtrace)
        print(io, "\nFor more details, run Julia with `-g2`, or launch the kernel with `@oneapi debug_level=2`.")
    else
        print(io, "\nStacktrace:")
        for (i, (func, file, line)) in enumerate(err.backtrace)
            print(io, "\n [", i, "] ", func)
            isempty(file) || print(io, " at ", file, ":", line)
        end
    end
end

## per-stream report

# Reports are carved out of slabs shared by a device's streams, as kernels access memory
# indirectly, which makes every allocation add to the cost of every launch (see `pool.jl`).
const REPORTS_PER_SLAB = 64

const free_report_slots = Dict{Tuple{ZeContext, ZeDevice}, Vector{ReportSlot}}()

KernelState(s::oneStream, kernel::ZeKernel) =
    KernelState(UInt(s.exceptions.report) % UInt64, UInt(s.exceptions.claim) % UInt64,
                kernel_id(kernel))
# explicitly created queues have no report
KernelState(::ZeCommandQueue, ::ZeKernel) = KernelState()

# how a launch identifies its kernel in the report. the stream keeps the kernels it launched
# alive until it has checked the report (see `note_launch!`), so the id is not reused.
kernel_id(kernel::ZeKernel) = UInt(pointer_from_objref(kernel)) % UInt64

# Remember that `kernel` was launched on `s`, to look up the exceptions it reports. The
# caller must hold `s.lock`.
@inline function note_launch!(s::oneStream, kernel::ZeKernel)
    s.last_launched === kernel && return
    push!(s.launched, kernel)
    s.last_launched = kernel
    return
end

# the pattern resetting a claim; a global, as Level Zero does not specify how long a fill's
# pattern has to remain valid
const unclaimed = Ref{UInt32}(0)

reset_claim!(list, slot::ReportSlot) =
    GC.@preserve unclaimed append_fill!(list, slot.claim,
                                        Base.unsafe_convert(Ptr{UInt32}, unclaimed),
                                        sizeof(UInt32), sizeof(UInt32))

clear_report!(slot::ReportSlot) =
    fill!(unsafe_wrap(Array, convert(Ptr{UInt8}, slot.report), sizeof(ExceptionReport)), 0x00)

function new_report_slab(ctx::ZeContext, dev::ZeDevice)
    reports = allocate(oneL0.HostBuffer, ctx, dev,
                       REPORTS_PER_SLAB * sizeof(ExceptionReport), sizeof(UInt64))
    claims = try
        allocate(oneL0.DeviceBuffer, ctx, dev,
                 REPORTS_PER_SLAB * sizeof(UInt32), sizeof(UInt32))
    catch
        release(reports)
        rethrow()
    end
    return [ReportSlot(convert(Ptr{ExceptionReport}, reports) +
                           (i - 1) * sizeof(ExceptionReport),
                       convert(ZePtr{UInt32}, pointer(claims)) + (i - 1) * sizeof(UInt32))
            for i in 1:REPORTS_PER_SLAB]
end

# The claim is reset on `list`, in order before any kernel uses it. Slabs are never freed.
function allocate_report_slot(ctx::ZeContext, dev::ZeDevice, list)
    slot = with_pool_lock() do
        free = get(free_report_slots, (ctx, dev), nothing)
        (free === nothing || isempty(free)) ? nothing : pop!(free)
    end
    if slot === nothing
        # outside of the lock, as allocating memory may collect garbage, freeing slots
        slots = new_report_slab(ctx, dev)
        slot = pop!(slots)
        with_pool_lock() do
            append!(get!(Vector{ReportSlot}, free_report_slots, (ctx, dev)), slots)
        end
    end
    try
        clear_report!(slot)
        reset_claim!(list, slot)
    catch
        release_report_slot(ctx, dev, slot)
        rethrow()
    end
    return slot
end

release_report_slot(ctx::ZeContext, dev::ZeDevice, slot::ReportSlot) =
    with_pool_lock() do
        # the slabs are forgotten when the precompilation workload resets its state, and
        # streams it created can be finalized later
        free = get(free_report_slots, (ctx, dev), nothing)
        free === nothing || push!(free, slot)
        return
    end

# Called when a stream is collected. Its kernels may still be executing, and writing the
# report, in which case the slot is leaked, as is the command list (see its finalizer).
function release_report_slot(s::oneStream)
    list = s.immediate
    # the list's finalizer may have run already, destroying it once it was idle
    if list.handle == C_NULL ||
       oneL0.unchecked_zeCommandListHostSynchronize(list, 0) == oneL0.RESULT_SUCCESS
        release_report_slot(s.ctx, s.dev, s.exceptions)
    end
    return
end

# Check whether a kernel on `s` threw an exception, returning it after resetting the report.
# The caller must hold `s.lock`, and have checked that the stream is idle.
function check_exception!(s::oneStream)
    slot = s.exceptions
    if unsafe_load(convert(Ptr{Int32}, slot.report)) == 0    # `status`
        forget_launches!(s)
        return nothing
    end
    report = unsafe_load(slot.report)

    # reset the claim first: if that fails, the exception is still reported later
    reset_claim!(s.immediate, slot)
    clear_report!(slot)

    # the report was written by the device, so it is only trusted to identify a kernel
    # launched since the last check, and a site in that kernel's table
    kernel = nothing
    for k in s.launched
        if kernel_id(k) == report.kernel
            kernel = k
            break
        end
    end
    forget_launches!(s)
    sites = kernel === nothing ? GPUCompiler.ExceptionSite[] : kernel.exception_sites
    site = get(sites, Int(report.site), nothing)
    name, reason, backtrace = if site === nothing
        "", "", Tuple{String, String, Int}[]
    else
        site.name, site.reason,
        Tuple{String, String, Int}[(frame.func, frame.file, frame.line)
                                   for frame in site.frames]
    end
    return KernelException(s.dev, name, reason,
                           Int.(report.work_item[1:3]), Int.(report.work_group[1:3]),
                           backtrace)
end

# Forget the kernels launched on `s`, once the report was checked, unless some may still
# execute: those recorded in a batch, or in batches that failed to submit. The caller must
# hold `s.lock`, and have checked that the stream is idle.
function forget_launches!(s::oneStream)
    if s.batch === nothing && isempty(s.failed)
        empty!(s.launched)
        s.last_launched = nothing
    end
    return
end
