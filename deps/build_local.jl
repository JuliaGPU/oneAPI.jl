# build liboneapi_support with C wrappers for C++ APIs

using Pkg

# Conda.jl fails to precompile when its root environment directory has been removed
# (e.g., by depot clean-up) while its deps.jl still points to it. Pre-create the
# directory so that precompilation succeeds; Conda lazily re-installs itself on use.
let deps_jl = joinpath(first(DEPOT_PATH), "conda", "deps.jl")
    if isfile(deps_jl)
        mod = Module()
        Base.include(mod, deps_jl)
        isdefined(mod, :ROOTENV) && mkpath(mod.ROOTENV)
    end
end

Pkg.activate(@__DIR__)
Pkg.instantiate()

if haskey(ENV, "BUILDKITE")
    run(`buildkite-agent annotate 'Using a locally-built support library; A bump of oneAPI_Support_jll is required before releasing this packages.' --style 'warning' --context 'ctx-deps'`)
end

using Scratch, Preferences, CMake_jll, Ninja_jll
import oneAPI_Level_Zero_Headers_LTS_jll as oneAPI_Level_Zero_Headers_jll

oneAPI = Base.UUID("8f75cd03-7ff8-4ecb-9b8f-daf728133b1b")

# get scratch directories
conda_dir = get_scratch!(oneAPI, "conda")
install_dir = get_scratch!(oneAPI, "deps")
rm(install_dir; recursive=true)

# get build directory
build_dir = if isempty(ARGS)
    mktempdir()
else
    ARGS[1]
end
mkpath(build_dir)

# install the toolchain
try
    using Conda
catch err
    # Sometimes, Conda fails to import because its environment is missing.
    # That's probably caused by a missing build, but Pkg should do that...
    Pkg.build("Conda")
    using Conda
end
if !isdir(Conda.ROOTENV)
    # Same as above
    Pkg.build("Conda")
end

# make sure the CA roots used by conda's Python are up-to-date, as outdated certificates
# otherwise result in SSL verification errors when accessing Intel's package repository
Conda.add(["ca-certificates", "certifi"], Conda.ROOTENV)

if !isfile(joinpath(conda_dir, "condarc-julia.yml"))
    Conda.create(conda_dir)
    # conda#8850
    mkpath(joinpath(conda_dir, "conda-meta"))
    touch(joinpath(conda_dir, "conda-meta", "history"))
end
Conda.add_channel("https://software.repos.intel.com/python/conda/", conda_dir)
Conda.add(["dpcpp_linux-64=2025.3.1", "mkl-devel-dpcpp=2025.3.1"], conda_dir)

Conda.list(conda_dir)

# XXX: isn't there a Conda package providing ze_api.hpp?
include_dir = joinpath(oneAPI_Level_Zero_Headers_jll.artifact_dir, "include")

# build and install
withenv("PATH"=>"$(ENV["PATH"]):$(Conda.bin_dir(conda_dir))",
        "LD_LIBRARY_PATH"=>Conda.lib_dir(conda_dir)) do
    cmake() do cmake_path
    ninja() do ninja_path
        run(```$cmake_path -DCMAKE_CXX_COMPILER="icpx"
                           -DCMAKE_CXX_FLAGS="-fsycl -isystem $(conda_dir)/include -isystem $include_dir -fdiagnostics-color=always"
                           -DCMAKE_INSTALL_RPATH=$(Conda.lib_dir(conda_dir))
                           -DCMAKE_INSTALL_PREFIX=$install_dir
                           -GNinja -S $(@__DIR__) -B $build_dir```)
        run(`$cmake_path --build $(build_dir) --target install`)
    end
    end
end

# TODO: adapt when we support more platforms
lib_path = joinpath(install_dir, "lib", "liboneapi_support.so")
@assert ispath(lib_path)

# tell oneAPI_Support_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "oneAPI_Support_jll",
    "liboneapi_support_path" => lib_path;
    force=true,
)

# copy the preferences to `test/` as well to work around Pkg.jl#2500
cp(joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
   joinpath(dirname(@__DIR__), "test", "LocalPreferences.toml"); force=true)
