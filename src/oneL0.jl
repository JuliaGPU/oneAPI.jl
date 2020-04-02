module oneL0

using CEnum
using CUDAapi

const libze = "libze_loader.so.0.91"

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
    zeInit(ZE_INIT_FLAG_NONE)
end

end
