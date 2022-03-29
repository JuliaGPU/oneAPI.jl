## generate preferences for loading a local copy of the oneAPI toolchain

#
# discovery
#

import Libdl

function scan_library!(output, lib, locations=String[])
    name = Libdl.find_library(lib, locations)
    if name != ""
        path = Libdl.dlopen(name) do handle
            Libdl.dlpath(handle)
        end
        println("- found $lib at $path")
        output[lib] = path
    else
        println("- did not find $lib")
    end
end

# NOTE: some JLLs also provide binaries (e.g. ocloc, iga64, etc),
#       but we don't scan for them if our toolchain does not use them

igc = Dict()
println("Trying to find local IGC...")
for lib = ["libigc", "libiga64", "libigdfcl", "libopencl-clang"]
    scan_library!(igc, lib)
end

gmmlib = Dict()
println("\nTrying to find local gmmlib...")
scan_library!(gmmlib, "libigdgmm")

neo = Dict()
println("\nTrying to find local NEO...")
## version suffixed
scan_library!(neo, "libze_intel_gpu.so.1")
## in intel-opencl subdirectory
locations = String[]
if haskey(igc, "libigc")
    push!(locations, joinpath(dirname(igc["libigc"]), "intel-opencl"))
end
scan_library!(neo, "libigdrcl", locations)

loader = Dict()
println("\nTrying to find local oneAPI loader...")
scan_library!(loader, "libze_loader")
scan_library!(loader, "libze_validation_layer")


#
# setting preferences
#

println("\nWriting preferences:\n")

using Pkg

# use a temporary environment to install packages we need
Pkg.activate(; temp=true)
Pkg.add(["Preferences", "NEO_jll", "oneAPI_Level_Zero_Loader_jll"])
using Preferences
using NEO_jll, oneAPI_Level_Zero_Loader_jll

# activate the global environment, where we'll set the preferences
Pkg.activate()

function set_preferences(mod, entries)
    for (lib, path) in entries
        binding = replace(split(lib, '.')[1], "-" => "_")
        if binding == "libiga64"
            binding = "libiga"  # sigh
        end
        set_preferences!(mod, binding * "_path" => path)
    end
end

set_preferences(NEO_jll, neo)
set_preferences(NEO_jll.libigc_jll, igc)
set_preferences(NEO_jll.gmmlib_jll, gmmlib)
set_preferences(oneAPI_Level_Zero_Loader_jll, loader)

println("""

Prefences have been written to `$(joinpath(dirname(Base.active_project()), "LocalPreferences.toml"))`.
Please modify the file to your liking, and remove the oneAPI-related preferences (or the entire file) to revert to the original binaries.""")
