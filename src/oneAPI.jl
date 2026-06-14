module oneAPI

using GPUArrays
using Adapt

using GPUCompiler

import ExprTools

using SpecialFunctions

import Preferences

import KernelAbstractions: KernelAbstractions

using LLVM
using LLVM.Interop
using Core: LLVMPtr

using SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll
using oneAPI_Support_jll

export oneL0

# core library
include("../lib/utils/APIUtils.jl")
include("../lib/level-zero/oneL0.jl")
using .oneL0
functional() = oneL0.functional[]

# device functionality
import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public
Base.Experimental.@MethodTable(method_table)
include("device/runtime.jl")
include("device/array.jl")
include("device/quirks.jl")
include("device/atomics.jl")

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
include("gpuarrays.jl")
include("random.jl")
include("utils.jl")

include("oneAPIKernels.jl")
import .oneAPIKernels: oneAPIBackend
include("accumulate.jl")
include("sorting.jl")
include("indexing.jl")
export oneAPIBackend

# Work around a deadlock in Pkg's parallel precompilation on Julia 1.10, where it does
# not pass `loadable_exts` to `Base.compilecache` (the kwarg is accidentally commented
# out in Pkg's precompilation.jl), so a worker precompiling an extension freely loads
# other extensions. If such an extension is being precompiled concurrently by another
# worker, loading it blocks on a pidfile lock held by the Pkg driver, which in turn
# waits for our worker: a deadlock. This bites oneAPI in particular, as it triggers
# extensions of its own dependencies (AtomixoneAPIExt, AcceleratedKernelsoneAPIExt)
# that have identical trigger sets and are thus precompiled concurrently.
#
# Mimic what Pkg on 1.11+ does by disallowing extension loading when precompiling an
# extension. This needs to happen from `__init__`, which runs in the worker process
# right before loading of oneAPI completes and extension callbacks are processed.
function prevent_extension_deadlock()
    isdefined(Base, :loadable_extensions) || return
    isdefined(Base, :precompilation_target) || return
    Base.loadable_extensions === nothing || return  # Pkg already restricted loading
    target = Base.precompilation_target
    (target === nothing || target.uuid === nothing) && return

    # determine whether the precompilation target is an extension, i.e., whether its
    # entry-point lives in the `ext` directory of a parent package whose UUID also
    # generates the extension's UUID (the scheme used by `Base.insert_extension_triggers`)
    path = Base.locate_package(target)
    path === nothing && return
    dir = dirname(path)
    if basename(dir) != "ext"
        dir = dirname(dir)
        basename(dir) == "ext" || return
    end
    parent_uuid = nothing
    for proj in Base.project_names
        project_file = joinpath(dirname(dir), proj)
        if isfile(project_file)
            d = Base.parsed_toml(project_file)
            parent_uuid = get(d, "uuid", nothing)
            break
        end
    end
    parent_uuid isa String || return
    target.uuid == Base.uuid5(Base.UUID(parent_uuid), target.name) || return

    Base.loadable_extensions = Base.PkgId[]
    return
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    if precompiling
        @static if VERSION < v"1.11"
            prevent_extension_deadlock()
        end
        return
    end

    if oneL0.functional[]
        @static if Sys.iswindows()
            @warn """oneAPI.jl support for native Windows is experimental and incomplete.
                 For the time being, it is recommended to use WSL or Linux instead."""
        else
            if oneL0.NEO_jll.is_available()
                # ensure that the OpenCL loader finds the ICD files from our artifacts
                ENV["OCL_ICD_FILENAMES"] = oneL0.NEO_jll.libigdrcl
            end
        end

        # XXX: work around an issue with SYCL/Level Zero interoperability
        #      (see JuliaGPU/oneAPI.jl#417)
        ENV["SYCL_PI_LEVEL_ZERO_BATCH_SIZE"] = "1"
    end
    return nothing
end

function set_debug!(debug::Bool)
    for jll in [oneL0.NEO_jll, oneL0.NEO_jll.libigc_jll]
        Preferences.set_preferences!(jll, "debug" => string(debug); force=true)
    end
    @info "oneAPI debug mode $(debug ? "enabled" : "disabled"); please re-start Julia."
end

end
