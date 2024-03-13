module Support

using ..oneAPI

using ..oneL0

using ..oneL0:
  ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t,
  ze_command_queue_handle_t, ze_event_handle_t

using oneAPI_Support_jll

include("liboneapi_support.jl")

# export everything
for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return
    
    if !oneAPI_Support_jll.is_available()
        @error """oneAPI support wrapper not available for your platform."""
        return
    end
end

end
