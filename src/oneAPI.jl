module oneAPI

using GPUArrays
using Adapt

using GPUCompiler

import ExprTools

using SpecialFunctions

import Preferences

using LLVM
using LLVM.Interop
using Core: LLVMPtr

using SPIRV_LLVM_Translator_unified_jll, SPIRV_Tools_jll

import KernelAbstractions as KA

export oneL0

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/level-zero/oneL0.jl")
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

# array abstraction
include("memory.jl")
include("pool.jl")
include("array.jl")

# compiler implementation
include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

if Sys.islinux()
# library interop
include("../lib/support/Support.jl")
include("../lib/sycl/SYCL.jl")
using .SYCL
export SYCL

# array libraries
include("../lib/mkl/oneMKL.jl")
export oneMKL
end

# integrations and specialized functionality
include("broadcast.jl")
include("mapreduce.jl")
include("oneAPIKernels.jl")
import .oneAPIKernels: oneAPIBackend, KA.launch_config
export oneAPIBackend

include("gpuarrays.jl")
include("random.jl")
include("utils.jl")

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return

    if Sys.iswindows()
        @warn """oneAPI.jl support for native Windows is experimental and incomplete.
                 For the time being, it is recommended to use WSL or Linux instead."""
    else
        # ensure that the OpenCL loader finds the ICD files from our artifacts
        ENV["OCL_ICD_FILENAMES"] = oneL0.NEO_jll.libigdrcl
    end

    # XXX: work around an issue with SYCL/Level Zero interoperability
    #      (see JuliaGPU/oneAPI.jl#417)
    ENV["SYCL_PI_LEVEL_ZERO_BATCH_SIZE"] = "1"
end

function set_debug!(debug::Bool)
    for jll in [oneL0.NEO_jll, oneL0.NEO_jll.libigc_jll]
        Preferences.set_preferences!(jll, "debug" => string(debug); force=true)
    end
    @info "oneAPI debug mode $(debug ? "enabled" : "disabled"); please re-start Julia."
end

end
