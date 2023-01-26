# build liboneapi_support with C wrappers for C++ APIs

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

if haskey(ENV, "BUILDKITE")
    run(`buildkite-agent annotate 'Using a locally-built support library; A bump of oneAPI_Support_jll is required before releasing this packages.' --style 'warning' --context 'ctx-deps'`)
end

using Scratch, Preferences, CMake_jll, Ninja_jll, oneAPI_Level_Zero_Headers_jll

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
if !isfile(joinpath(conda_dir, "condarc-julia.yml"))
    Conda.create(conda_dir)
    # conda#8850
    mkpath(joinpath(conda_dir, "conda-meta"))
    touch(joinpath(conda_dir, "conda-meta", "history"))
end
Conda.add(["dpcpp_linux-64", "mkl-devel-dpcpp"], conda_dir; channel="intel")

Conda.list(conda_dir)

# XXX: isn't there a Conda package providing ze_api.hpp?
include_dir = joinpath(oneAPI_Level_Zero_Headers_jll.artifact_dir, "include")

# build and install
withenv("PATH"=>"$(ENV["PATH"]):$(Conda.bin_dir(conda_dir))",
        "LD_LIBRARY_PATH"=>Conda.lib_dir(conda_dir)) do
    cmake() do cmake_path
    ninja() do ninja_path
        run(```$cmake_path -DCMAKE_CXX_COMPILER="icpx"
                           -DCMAKE_CXX_FLAGS="-fsycl -isystem $(conda_dir)/include -isystem $include_dir"
                           -DCMAKE_INSTALL_RPATH=$(Conda.lib_dir(conda_dir))
                           -DCMAKE_INSTALL_PREFIX=$install_dir
                           -GNinja -S $(@__DIR__) -B $build_dir```)
        run(`$ninja_path -C $(build_dir) install`)
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
