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

# The sparse wrappers were regenerated with a wider ABI: the onemklX sparse_set_{csr,csc,coo}
# setters gained a 64-bit `nnz` argument and 64-bit dims, and new set_csc_data / set_bsr_data
# entry points were added. These only match a liboneapi_support built from the updated
# onemkl.cpp/onemkl.h. The oneAPI_Support_jll 0.9.2 binary shipped in the registry predates
# them, so calling a sparse setter against it shifts every pointer one slot and MKL
# dereferences garbage -> segfault or silently corrupted sparse data. `deps/build_local.jl`
# rebuilds a matching binary; a plain `Pkg.add` / CI `julia-buildpkg` does not. Probe once at
# load for a symbol that only the rebuilt binary exports and turn the crash into a clear error.
const _sparse_abi_ok = Ref(false)

function _check_sparse_abi()
    _sparse_abi_ok[] && return
    error(
        "The loaded oneAPI_Support_jll binary predates the current sparse wrappers and has an " *
        "incompatible ABI; calling sparse routines against it would corrupt memory or crash. " *
        "Rebuild the support library from source with `julia --project deps/build_local.jl` " *
        "(or install a oneAPI_Support_jll built from the updated onemkl.cpp)."
    )
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return

    if !oneAPI_Support_jll.is_available()
        @error """oneAPI support wrapper not available for your platform."""
        return
    end

    # Probe the loaded support binary for a symbol that only the regenerated (wide-ABI)
    # sparse wrappers export; `_check_sparse_abi` gates the affected calls on the result.
    handle = Libdl.dlopen(oneAPI_Support_jll.liboneapi_support; throw_error = false)
    if handle !== nothing
        _sparse_abi_ok[] =
            Libdl.dlsym(handle, :onemklSsparse_set_csc_data; throw_error = false) !== nothing
    end
end

end
