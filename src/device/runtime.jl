# device runtime libraries


## kernel state

# GPUCompiler passes this as a hidden kernel argument and forwards it to device callees.
struct KernelState
    # Initialized on the device by add_heap!; the host passes a null pointer.
    heap::LLVMPtr{UInt8, AS.Function}

    # Addresses of the launching stream's exception report and claim word, or zero when
    # exceptions are not reported, and how the host identifies the launched kernel in the
    # report (see `check_exception!`). Integers, as SPIR-V validation rejects pointer fields
    # in by-value kernel arguments.
    exception_report::UInt64
    exception_claim::UInt64
    exception_kernel::UInt64
end

KernelState(exception_report::UInt64=UInt64(0), exception_claim::UInt64=UInt64(0),
            exception_kernel::UInt64=UInt64(0)) =
    KernelState(reinterpret(LLVMPtr{UInt8, AS.Function}, C_NULL),
                exception_report, exception_claim, exception_kernel)

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
# A throw only reports the id of its site (see `GPUCompiler.uses_exception_sites`), which
# the host looks up in the table of the kernel that threw. So the report also identifies the
# kernel, as passed by the host in the `KernelState`.

using GPUCompiler: @gputhrow

struct ExceptionReport
    status::Int32
    site::Int32
    kernel::UInt64
    # Four components, as narrowing or merging adjacent three-component accesses can
    # produce six-wide vectors, which SPIR-V does not support.
    work_item::NTuple{4, UInt64}
    work_group::NTuple{4, UInt64}
end

# Offsets are computed when generating the code, as inference does not always constant-fold
# `fieldoffset` here, and are added as `UInt64`, as mixing in an `Int` would be a checked
# conversion.
@generated function report_field(::Val{field}) where {field}
    T = fieldtype(ExceptionReport, field)
    offset = fieldoffset(ExceptionReport, Base.fieldindex(ExceptionReport, field)) % UInt64
    :(reinterpret(LLVMPtr{$T, AS.CrossWorkgroup}, kernel_state().exception_report + $offset))
end

# Nothing reachable from here may throw, not even a checked conversion.
@device_function function exception_site(site::Int32)
    claim = kernel_state().exception_claim
    claim == 0 && return
    # A relaxed claim suffices: the host only reads the report once the kernel has completed.
    claim_ptr = reinterpret(LLVMPtr{UInt32, AS.CrossWorkgroup}, claim)
    if SPIRVIntrinsics.atomic_xchg!(claim_ptr, 1 % UInt32) == 0
        unsafe_store!(report_field(Val(:site)), site)
        unsafe_store!(report_field(Val(:kernel)), kernel_state().exception_kernel)
        unsafe_store!(report_field(Val(:work_item)),
                      (get_local_id(1) % UInt64, get_local_id(2) % UInt64,
                       get_local_id(3) % UInt64, 0 % UInt64))
        unsafe_store!(report_field(Val(:work_group)),
                      (get_group_id(1) % UInt64, get_group_id(2) % UInt64,
                       get_group_id(3) % UInt64, 0 % UInt64))
        unsafe_store!(report_field(Val(:status)), 1 % Int32)
    end
    return
end
