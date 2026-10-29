# device runtime libraries


## kernel state

# GPUCompiler passes this as a hidden kernel argument and forwards it to device callees.
struct KernelState
    # Initialized on the device by add_heap!; the host passes a null pointer.
    heap::LLVMPtr{UInt8, AS.Function}

    # Addresses of the launching stream's exception report and claim word, or zero when
    # exceptions are not reported. Integers, as SPIR-V validation rejects pointer fields in
    # by-value kernel arguments.
    exception_report::UInt64
    exception_claim::UInt64
end

KernelState(exception_report::UInt64=UInt64(0), exception_claim::UInt64=UInt64(0)) =
    KernelState(reinterpret(LLVMPtr{UInt8, AS.Function}, C_NULL),
                exception_report, exception_claim)

@device_function @inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)


## dynamic memory allocation

# Julia's boxed objects use address-space-0 pointers, which SPIR-V maps to private
# memory. A global (USM) allocation cannot back those pointers on Intel GPUs.
# Use a per-work-item bump allocator: objects remain valid until the work-item exits,
# and must not be shared with other work-items or retained across launches.
#
# add_heap! reserves the arena in the kernel entry block. Its header contains the cursor
# and capacity in bytes, keeping the runtime independent of the compiler's chosen size.

# bytes of private memory reserved per work-item for dynamic allocations
const HEAP_SIZE = 1024

# alignment of every allocation; the largest Julia's codegen assumes for heap objects
const HEAP_ALIGNMENT = 16

# Two 64-bit words keep the payload aligned to HEAP_ALIGNMENT.
const HEAP_HEADER = 2 * sizeof(Csize_t)

@device_function function malloc(sz::Csize_t)
    heap = kernel_state().heap
    heap == reinterpret(LLVMPtr{UInt8, AS.Function}, C_NULL) && return C_NULL

    header = reinterpret(LLVMPtr{Csize_t, AS.Function}, heap)
    cursor = unsafe_load(header, 1, Val(sizeof(Csize_t)))
    capacity = unsafe_load(header, 2, Val(sizeof(Csize_t)))

    bytes = (sz + Csize_t(HEAP_ALIGNMENT - 1)) & ~Csize_t(HEAP_ALIGNMENT - 1)
    bytes < sz && return C_NULL                     # alignment rounding overflowed
    bytes > capacity - cursor && return C_NULL      # gc_pool_alloc reports exhaustion

    unsafe_store!(header, cursor + bytes, 1, Val(sizeof(Csize_t)))
    # `cursor` fits an `Int` (it is below the capacity); reinterpret rather than convert,
    # as the checked conversion would drag an `InexactError` throw path into every kernel
    return reinterpret(Ptr{Cvoid}, heap + HEAP_HEADER + (cursor % Int))
end

function report_oom(sz)
    @println("ERROR: Out of dynamic GPU memory (trying to allocate ", sz, " bytes)")
    return
end


## exceptions

# A work-item that throws reports to its stream's `ExceptionReport`, which the host checks
# when synchronizing the stream (see `check_exception!`). SPIR-V cannot abort a kernel, so
# the work-item then exits (see `lower_unreachable_control_flow!` in GPUCompiler) while the
# others keep running.
#
# The report lives in host memory, which the host can read without migrating pages, but on
# which devices need not support atomics. So the first work-item to claim the stream's claim
# word, in device memory, becomes the report's only writer. The claim is held until the
# host has read the report, so a stream reports one exception per synchronization.
#
# No per-work-item state is kept, as reserving private memory slows down every kernel.
#
# What is reported depends on the debug level: at 0 only that an exception was thrown, at 1
# also the exception's name and reason, and at 2 also the position of the work-item and a
# backtrace.
#
# Strings are reported by the address of the constant holding them, and read by the host.
# Copying them here would put a loop in every reporting function, which GPUCompiler calls
# for every frame of every throw site, and makes IGC generate many times more code.

const EXCEPTION_MAX_FRAMES = 16

# String fields are addresses of NUL-terminated strings in device memory, or zero.
struct ExceptionFrame
    func::UInt64
    file::UInt64
    line::Int32
end

struct ExceptionReport
    status::Int32
    num_frames::Int32
    # Four components, as narrowing or merging adjacent three-component accesses can
    # produce six-wide vectors, which SPIR-V does not support.
    work_item::NTuple{4, UInt64}
    work_group::NTuple{4, UInt64}
    name::UInt64
    reason::UInt64
    frames::NTuple{EXCEPTION_MAX_FRAMES, ExceptionFrame}
end

# Offsets are computed when generating the code, as inference does not always constant-fold
# `fieldoffset` in the code `@gputhrow` reaches, and are added as `UInt64`, as mixing in an
# `Int` would be a checked conversion.
@generated function report_field(::Val{field}) where {field}
    T = fieldtype(ExceptionReport, field)
    offset = fieldoffset(ExceptionReport, Base.fieldindex(ExceptionReport, field)) % UInt64
    :(reinterpret(LLVMPtr{$T, AS.CrossWorkgroup}, kernel_state().exception_report + $offset))
end

@generated function frame_field(idx, ::Val{field}) where {field}
    T = fieldtype(ExceptionFrame, field)
    offset = (fieldoffset(ExceptionReport, Base.fieldindex(ExceptionReport, :frames)) +
              fieldoffset(ExceptionFrame, Base.fieldindex(ExceptionFrame, field))) % UInt64
    stride = sizeof(ExceptionFrame) % UInt64
    quote
        frame = (idx - 1) % UInt64 * $stride
        reinterpret(LLVMPtr{$T, AS.CrossWorkgroup},
                    kernel_state().exception_report + frame + $offset)
    end
end

const CString = LLVMPtr{Cchar, AS.CrossWorkgroup}

@llvmgenerated builder function string_pointer(::Val{str})::CString where {str}
    globalstring_ptr!(builder, String(str); addrspace=AS.CrossWorkgroup)
end

# throw a device-side exception of type `name`, which is reported together with `reason`
macro gputhrow(name::String, reason::String)
    # without line numbers, so that backtraces start at the caller
    Base.remove_linenums!(quote
        if kernel_debug_level() >= 1
            reporting_call(record_exception!, string_pointer($(Val(Symbol(name)))),
                           string_pointer($(Val(Symbol(reason)))))
        end
        throw(nothing)
    end)
end

# The work-item's id in the claim word, where zero means unclaimed. Ids repeat beyond 2^32-1
# work-items, in which case two throwing work-items could both write the report.
@inline function work_item_id()
    id = get_global_linear_id() % UInt32
    return id == 0 ? 1 % UInt32 : id
end

# Claim the report, returning whether this work-item owns it. Idempotent, so that every
# reporting function can check ownership with it. The owner sets the report's status before
# exiting, which keeps work-items with the same id in later kernels from claiming it too.
@device_function @inline function claim_report!()
    claim = kernel_state().exception_claim
    claim == 0 && return false
    # A relaxed claim suffices: the host only reads the report once the kernel has completed.
    claim_ptr = reinterpret(LLVMPtr{UInt32, AS.CrossWorkgroup}, claim)
    id = work_item_id()
    old = SPIRVIntrinsics.atomic_cmpxchg!(claim_ptr, 0 % UInt32, id)
    old == 0 && return true
    return old == id && unsafe_load(report_field(Val(:status))) == 0
end

# GPUCompiler inlines throwing functions into every call site, where reporting is kept out
# of line, as it slows down kernels otherwise. Except at -g2, where every frame of every
# throw site is reported too: IGC compiles that faster inlined than as calls.
@inline reporting_call(f, args...) =
    kernel_debug_level() >= 2 ? f(args...) : out_of_line(f, args...)
@noinline out_of_line(f, args...) = f(args...)

string_address(str::CString) = reinterpret(UInt64, str)

# Record the exception's name and reason, unless one was recorded already: `@gputhrow` comes
# first, and knows a more specific name than GPUCompiler derives from the throw.
@device_function @inline function record_exception!(name::CString, reason::Union{CString, Nothing}=nothing)
    reporting_call(claim_report!) || return false
    name_ptr = report_field(Val(:name))
    if unsafe_load(name_ptr) == 0
        unsafe_store!(name_ptr, string_address(name))
        reason === nothing ||
            unsafe_store!(report_field(Val(:reason)), string_address(reason))
    end
    return true
end

report_exception(ex) = (reporting_call(record_exception!, ex); return)

function report_exception_name(ex)
    record_exception!(ex) || return
    unsafe_store!(report_field(Val(:work_item)),
                  (get_local_id(1) % UInt64, get_local_id(2) % UInt64,
                   get_local_id(3) % UInt64, 0 % UInt64))
    unsafe_store!(report_field(Val(:work_group)),
                  (get_group_id(1) % UInt64, get_group_id(2) % UInt64,
                   get_group_id(3) % UInt64, 0 % UInt64))
    return
end

function report_exception_frame(idx, func, file, line)
    claim_report!() || return
    1 <= idx <= EXCEPTION_MAX_FRAMES || return
    unsafe_store!(frame_field(idx, Val(:func)), string_address(func))
    unsafe_store!(frame_field(idx, Val(:file)), string_address(file))
    unsafe_store!(frame_field(idx, Val(:line)), line % Int32)
    unsafe_store!(report_field(Val(:num_frames)), idx % Int32)
    return
end

@device_function function signal_exception()
    # At debug level 0 this is the only reporting function, and claims the report.
    reporting_call(claim_report!) && unsafe_store!(report_field(Val(:status)), 1 % Int32)
    return
end
