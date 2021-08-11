module oneAPI

using GPUArrays
using Adapt

using GPUCompiler

import ExprTools

using SpecialFunctions

using LLVM
using LLVM.Interop
using Core: LLVMPtr

using SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll

export oneL0, SYCL

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/level-zero/oneL0.jl")
include("../lib/sycl/SYCL.jl")
using .oneL0
functional() = oneL0.functional[]

# device functionality (needs to be loaded first, because of generated functions)
include("device/utils.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/runtime.jl")
include("device/opencl/work_item.jl")
include("device/opencl/synchronization.jl")
include("device/opencl/memory.jl")
include("device/opencl/printf.jl")
include("device/opencl/math.jl")
include("device/opencl/integer.jl")
include("device/opencl/atomic.jl")
include("device/quirks.jl")

# essential stuff
include("context.jl")

# compiler implementation
include("compiler/gpucompiler.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

# array abstraction
include("memory.jl")
include("pool.jl")
include("array.jl")

# array libraries
include("../lib/mkl/oneMKL.jl")
export oneMKL

# integrations and specialized functionality
include("broadcast.jl")
include("mapreduce.jl")
include("gpuarrays.jl")
include("random.jl")
include("utils.jl")

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    if !precompiling
        eval(overrides)
    end
end

end
