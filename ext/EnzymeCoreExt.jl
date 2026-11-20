# compatibility with EnzymeCore
module EnzymeCoreExt

using oneAPI
import oneAPI: GPUCompiler, oneAPIBackend, oneL0
import KernelInterface as KI

import EnzymeCore
using EnzymeCore.EnzymeRules

# the compiler job that kernels for `backend` are compiled with (see `KI.kernel_function`),
# for Enzyme to compile the derivatives of kernels (e.g. by KernelAbstractions' Enzyme
# extension) and their tapes
function EnzymeCore.compiler_job_from_backend(
        backend::oneAPIBackend, @nospecialize(F::Type), @nospecialize(TT::Type)
    )
    mi = GPUCompiler.methodinstance(F, TT)
    sub_group_size = KI.sub_group_size(backend)
    config = if sub_group_size > 0
        oneAPI.compiler_config(oneAPI.device(); backend.always_inline, sub_group_size)
    else
        oneAPI.compiler_config(oneAPI.device(); backend.always_inline)
    end
    return GPUCompiler.CompilerJob(mi, config)
end

# waiting for the GPU has no derivative
EnzymeRules.inactive(::typeof(oneL0.synchronize), args...; kwargs...) = nothing

end
