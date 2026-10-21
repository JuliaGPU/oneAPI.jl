# reporting of exceptions thrown by kernels

export KernelException

"""
    KernelException

An exception thrown by a kernel on device `dev`, reported when synchronizing the stream it
was launched on (e.g. by `synchronize()`, or a copy to the host such as `Array(x)`).

What is known about the exception depends on the debug level the kernel was compiled with
(Julia's `-g` level, or `@oneapi debug_level=...`):

- `0`: only that an exception was thrown;
- `1`: also the `name` of its type and the `reason`, for exceptions thrown by Julia's
  runtime (e.g. bounds errors, or domain errors in math functions);
- `2`: also the position of the work-item that threw (`work_item` is its local id,
  `work_group` the id of its work-group), and its `backtrace` as `(function, file, line)`
  tuples.

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
    if err.work_item == (0, 0, 0)
        print(io, "\nFor more details, run Julia with `-g2`, or launch the kernel with `@oneapi debug_level=2`.")
    else
        print(io, "\nStacktrace:")
        for (i, (func, file, line)) in enumerate(err.backtrace)
            print(io, "\n [", i, "] ", func)
            isempty(file) || print(io, " at ", file, ":", line)
        end
    end
end

const EXCEPTION_STRING_MAX = 4096

# buffers that copies may still be writing to, as their list could not be synchronized
const unsynchronized_string_buffers = Any[]
const unsynchronized_string_buffers_lock = ReentrantLock()

# Read the strings at `addresses`, as reported by a kernel, returning a dictionary that maps
# the addresses that could be read to their contents. The addresses come from memory the
# kernel could have corrupted, so they are only read if they fall within an allocation,
# such as a module's constants, and only up to the end of it. Freed modules are not read.
function read_device_strings(list, ctx::ZeContext, addresses)
    buffers = Dict{UInt64, Vector{UInt8}}()
    for address in addresses
        (address == 0 || haskey(buffers, address)) && continue
        base = Ref{Ptr{Cvoid}}()
        size = Ref{Csize_t}()
        res = oneL0.unchecked_zeMemGetAddressRange(ctx, Ptr{Cvoid}(address), base, size)
        res == oneL0.RESULT_SUCCESS || continue
        available = UInt64(UInt(base[]) + size[]) - address
        available > 0 || continue
        buffers[address] = Vector{UInt8}(undef, min(available, EXCEPTION_STRING_MAX))
    end
    isempty(buffers) && return Dict{UInt64, String}()
    copied = false
    GC.@preserve buffers try
        for (address, buffer) in buffers
            append_copy!(list, pointer(buffer), ZePtr{UInt8}(address), length(buffer))
        end
        copied = true
    catch err
        bt = catch_backtrace()
        @error "Failed to read the strings of a kernel exception" exception=(err, bt)
    finally
        # copies appended before a failure still write to the buffers
        if oneL0.unchecked_zeCommandListHostSynchronize(list, typemax(UInt64)) !=
           oneL0.RESULT_SUCCESS
            @lock unsynchronized_string_buffers_lock begin
                push!(unsynchronized_string_buffers, buffers)
            end
            copied = false
        end
    end
    copied || return Dict{UInt64, String}()
    return Dict(address => String(buffer[1:something(findfirst(iszero, buffer),
                                                      length(buffer) + 1) - 1])
                for (address, buffer) in buffers)
end


## per-stream report

# Reports are carved out of slabs shared by a device's streams, as kernels access memory
# indirectly, which makes every allocation add to the cost of every launch (see `pool.jl`).
const REPORTS_PER_SLAB = 64

const free_report_slots = Dict{Tuple{ZeContext, ZeDevice}, Vector{ReportSlot}}()

KernelState(s::oneStream) = KernelState(UInt(s.exceptions.report) % UInt64,
                                        UInt(s.exceptions.claim) % UInt64)
# explicitly created queues have no report
KernelState(::ZeCommandQueue) = KernelState()

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
    with_pool_lock(() -> push!(free_report_slots[(ctx, dev)], slot))

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
# The caller must hold `s.lock`.
function check_exception!(s::oneStream)
    # Kernels may still write the report while the stream is busy. That is only the case
    # after waiting for it if another task submitted work in the meantime, in which case the
    # next synchronization reports the exception.
    oneL0.unchecked_zeCommandListHostSynchronize(s.immediate, 0) == oneL0.RESULT_SUCCESS ||
        return nothing

    slot = s.exceptions
    unsafe_load(convert(Ptr{Int32}, slot.report)) == 0 && return nothing    # `status`
    report = unsafe_load(slot.report)

    # reset the claim first: if that fails, the exception is still reported later
    reset_claim!(s.immediate, slot)
    clear_report!(slot)

    nframes = clamp(Int(report.num_frames), 0, EXCEPTION_MAX_FRAMES)
    frames = report.frames[1:nframes]
    strings = read_device_strings(s.immediate, s.ctx,
                                  [report.name; report.reason;
                                   [frame.func for frame in frames];
                                   [frame.file for frame in frames]])
    str(address) = get(strings, address, "")
    backtrace = Tuple{String, String, Int}[(str(frame.func), str(frame.file),
                                            Int(frame.line)) for frame in frames]
    return KernelException(s.dev, str(report.name), str(report.reason),
                           Int.(report.work_item[1:3]), Int.(report.work_group[1:3]),
                           backtrace)
end
