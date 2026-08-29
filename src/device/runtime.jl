# device runtime libraries
#
# GPUCompiler resolves the back-end runtime by name in this module (`runtime_module`):
# `signal_exception`, `report_*` and `malloc` below are compiled into the runtime library
# that is linked into every kernel, with `malloc` becoming the `gpu_malloc` symbol that
# `gc_pool_alloc` — and so every heap allocation that survives optimization — calls.


## kernel state

# Passed by value as the hidden first argument of every kernel and threaded by GPUCompiler
# to every device function that calls `kernel_state()`. The exception flag is filled in on
# the host (`kernel_state` in src/exceptions.jl, once per linked kernel); the heap pointer
# is patched in on the device, in the kernel's entry block (`add_private_heap!` in
# src/compiler/compilation.jl), because it points at private memory.
struct KernelState
    # host USM, 16 bytes, read by the host after synchronization and cleared by it:
    #   [1] nonzero after `signal_exception`, [2] nonzero after `report_oom`
    exception_flag::LLVMPtr{Int32, AS.CrossWorkgroup}
    # per-work-item private memory: a `HEAP_HEADER`-byte header whose first word is the
    # bump cursor, followed by `PRIVATE_HEAP_SIZE` allocatable bytes; null in kernels that
    # do not allocate
    heap::Ptr{UInt8}
end

# Bytes of private memory set aside for dynamic allocations, per work-item. Julia objects
# allocated in a kernel never outlive the work-item that created them — exception objects
# on a throw path, boxes handed to a `@noinline` callee — so private memory is the right
# place for them, and the only one: address space 0, which Julia's boxed objects live in
# after GPUCompiler strips its address spaces, is private memory to SPIR-V and Intel's
# compiler, and a store through an address-space-0 pointer into global memory is silently
# lost. The arena is only materialized in kernels whose code calls `malloc`.
const PRIVATE_HEAP_SIZE = 1024
const HEAP_HEADER = 16     # keeps the first allocation 16-byte aligned

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)


## exceptions

function signal_exception()
    unsafe_store!(kernel_state().exception_flag, Int32(1), 1)
    return
end

function report_oom(sz)
    unsafe_store!(kernel_state().exception_flag, Int32(1), 2)
    return
end

function report_exception(ex)
    # @cuprintf("""
    #     ERROR: a %s was thrown during kernel execution.
    #            Run Julia on debug level 2 for device stack traces.
    #     """, ex)
    return
end

function report_exception_name(ex)
    # @cuprintf("""
    #     ERROR: a %s was thrown during kernel execution.
    #     Stacktrace:
    #     """, ex)
    return
end

function report_exception_frame(idx, func, file, line)
    # @cuprintf(" [%i] %s at %s:%i\n", idx, func, file, line)
    return
end


## dynamic memory allocation

# Bump allocator over the work-item's private arena: nothing is ever freed (the arena dies
# with the work-item), and exhaustion returns null, which `gc_pool_alloc` turns into
# `report_oom` + `OutOfMemoryError`, i.e. a loud `KernelException` on the host rather than
# a silent failure. The cursor is private to the work-item, so no atomics are needed.
function malloc(sz::Csize_t)
    heap = kernel_state().heap
    heap == C_NULL && return C_NULL
    sz > PRIVATE_HEAP_SIZE && return C_NULL
    bytes = (UInt32(sz) + UInt32(15)) & ~UInt32(15)
    cursor = convert(Ptr{UInt32}, heap)
    old = unsafe_load(cursor)
    new = old + bytes
    new > PRIVATE_HEAP_SIZE && return C_NULL
    unsafe_store!(cursor, new)
    return Ptr{Cvoid}(heap + HEAP_HEADER + old)
end
