# device runtime libraries


## kernel state

# GPUCompiler passes this as a hidden kernel argument and forwards it to device callees.
struct KernelState
    # Initialized on the device by add_heap!; the host passes a null pointer.
    heap::LLVMPtr{UInt8, AS.Function}
end

KernelState() = KernelState(reinterpret(LLVMPtr{UInt8, AS.Function}, C_NULL))

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)


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

function malloc(sz::Csize_t)
    heap = kernel_state().heap
    heap == reinterpret(LLVMPtr{UInt8, AS.Function}, C_NULL) && return C_NULL

    header = reinterpret(LLVMPtr{Csize_t, AS.Function}, heap)
    cursor = unsafe_load(header, 1, Val(sizeof(Csize_t)))
    capacity = unsafe_load(header, 2, Val(sizeof(Csize_t)))

    bytes = (sz + Csize_t(HEAP_ALIGNMENT - 1)) & ~Csize_t(HEAP_ALIGNMENT - 1)
    bytes < sz && return C_NULL                     # alignment rounding overflowed
    bytes > capacity - cursor && return C_NULL      # gc_pool_alloc reports exhaustion

    unsafe_store!(header, cursor + bytes, 1, Val(sizeof(Csize_t)))
    return reinterpret(Ptr{Cvoid}, heap + HEAP_HEADER + cursor)
end

function report_oom(sz)
    @println("ERROR: Out of dynamic GPU memory (trying to allocate ", sz, " bytes)")
    return
end


## exceptions

# SPIR-V has no way to abort a kernel, and the exception is not reported to the host: the
# work-item that threw simply exits (see `lower_unreachable_control_flow!` in GPUCompiler).
function signal_exception()
    return
end

function report_exception(ex)
    return
end

function report_exception_name(ex)
    return
end

function report_exception_frame(idx, func, file, line)
    return
end
