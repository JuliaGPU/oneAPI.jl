module Support

using ..oneAPI

using ..oneL0

using ..oneL0:
  ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t,
  ze_command_queue_handle_t, ze_event_handle_t

using oneAPI_Support_jll

import Libdl

include("liboneapi_support.jl")

# export everything
for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

# A CSC symbol alone does not establish exception containment: older wide-ABI artifacts
# expose it but can abort on an unsupported native operation. Version 2 also promises
# the catch-all boundary. Probe the loaded library, without disabling unrelated dense APIs.
const _sparse_wrapper_abi_version = UInt32(2)
const _sparse_abi_ok = Ref(false)
const _sparse_abi_error = Ref("support library has not been checked")

function _sparse_wrapper_error(lookup)
    marker = lookup(:onemkl_sparse_wrapper_abi_version)
    marker === nothing && return "missing sparse wrapper version marker"
    version = ccall(marker, UInt32, ())
    version == _sparse_wrapper_abi_version ||
        return "expected sparse wrapper version $(_sparse_wrapper_abi_version), found $version"
    lookup(:onemklSsparse_set_csc_data) === nothing && return "missing wide CSC entrypoint"
    return nothing
end

function _check_sparse_abi()
    _sparse_abi_ok[] && return
    error(
        "The loaded oneAPI support library $(liboneapi_support) is not safe for sparse calls: " *
        "$(_sparse_abi_error[]). Required wrapper version 2 includes wide signatures and " *
        "native exception containment. " *
        "Rebuild the support library from source with `julia --project deps/build_local.jl` " *
        "before using sparse operations."
    )
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return

    if !oneAPI_Support_jll.is_available()
        @error """oneAPI support wrapper not available for your platform."""
        return
    end

    _sparse_abi_ok[] = false
    _sparse_abi_error[] = "support library could not be loaded"
    handle = Libdl.dlopen(oneAPI_Support_jll.liboneapi_support; throw_error = false)
    if handle !== nothing
        try
            reason = _sparse_wrapper_error(name -> Libdl.dlsym(handle, name; throw_error=false))
            _sparse_abi_ok[] = reason === nothing
            _sparse_abi_error[] = something(reason, "")
        finally
            Libdl.dlclose(handle)
        end
    end
end

end
