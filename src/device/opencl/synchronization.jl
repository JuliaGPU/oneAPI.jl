# Synchronization Functions

export barrier

const cl_mem_fence_flags = Cuint
const CLK_LOCAL_MEM_FENCE = cl_mem_fence_flags(1)
const CLK_GLOBAL_MEM_FENCE = cl_mem_fence_flags(2)

barrier(flags=0) = @builtin_ccall("barrier", Cvoid, (Cuint,), flags)
