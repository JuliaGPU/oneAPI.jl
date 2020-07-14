module oneAPI

using GPUArrays
using Adapt

using GPUCompiler

using LLVM
using LLVM.Interop
using Core: LLVMPtr

using SPIRV_LLVM_Translator_jll, SPIRV_Tools_jll

export oneL0

# libraries
include("../lib/utils/APIUtils.jl")
include("../lib/level-zero/oneL0.jl")
using .oneL0

# device functionality
include("device/pointer.jl")
include("device/array.jl")
include("device/runtime.jl")
include("device/opencl/utils.jl")
include("device/opencl/work_item.jl")
include("device/opencl/printf.jl")
include("device/opencl/math.jl")

# host functionality
include("context.jl")
include("compiler.jl")
include("execution.jl")
include("memory.jl")
include("array.jl")
include("gpuarrays.jl")
include("reflection.jl")

end
