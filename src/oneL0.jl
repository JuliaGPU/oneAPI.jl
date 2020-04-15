module oneL0

using CEnum
using CUDAapi

using oneAPI_Level_Zero_jll

# core API
include("libze_common.jl")
include("error.jl")
include("libze.jl")
include("libze_aliases.jl")

include("utils.jl")

# wrappers
include("common.jl")
include("drivers.jl")
include("devices.jl")
include("commands.jl")
include("events.jl")
include("barriers.jl")

# wrappers

function __init__()
    res = unsafe_zeInit(ZE_INIT_FLAG_NONE)
    if res == RESULT_ERROR_UNINITIALIZED
        # https://github.com/oneapi-src/level-zero/issues/7#issuecomment-606701224
        error("No oneAPI driver implementation found.")
    elseif res !== RESULT_SUCCESS
        throw_api_error(res)
    end
end

end
