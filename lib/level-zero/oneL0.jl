module oneL0

using ..APIUtils

using CEnum

using Printf

using NEO_jll
using oneAPI_Level_Zero_Loader_jll

include("utils.jl")
include("pointer.jl")

# core API
macro check(ex)
    quote
        res = $(esc(ex))
        if res != RESULT_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
include("libze.jl")

# core wrappers
include("error.jl")
include("common.jl")
include("driver.jl")
include("device.jl")
include("context.jl")
include("cmdqueue.jl")
include("cmdlist.jl")
include("event.jl")
include("barrier.jl")
include("module.jl")
include("memory.jl")
include("copy.jl")
include("residency.jl")

const functional = Ref{Bool}(false)

function __init__()
    res = unsafe_zeInit(0)
    if res == RESULT_ERROR_UNINITIALIZED
        @error """No compatible oneAPI driver implementation found.
                  Your hardware probably is not supported by any oneAPI driver.

                  oneAPI.jl currently only supports the Intel Compute runtime,
                  consult their README for a list of compatible hardware:
                  https://github.com/intel/compute-runtime#supported-platforms"""
    elseif res !== RESULT_SUCCESS
        throw_api_error(res)
    else
        functional[] = true
    end
end

end
