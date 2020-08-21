module oneL0

using ..APIUtils

using CEnum

using Printf

using NEO_jll
using oneAPI_Level_Zero_Loader_jll

include("utils.jl")
include("pointer.jl")

# core API
include("libze_common.jl")
include("error.jl")
include("libze.jl")
include("libze_aliases.jl")

# core wrappers
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

function __init__()
    res = unsafe_zeInit(0)
    if res == RESULT_ERROR_UNINITIALIZED
        # https://github.com/oneapi-src/level-zero/issues/7#issuecomment-606701224
        error("No oneAPI driver implementation found.")
    elseif res !== RESULT_SUCCESS
        throw_api_error(res)
    end
end

end
