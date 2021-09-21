# build liboneapilib with C wrappers for C++ APIs

# XXX: build this library on Yggdrasil

using Pkg
Pkg.instantiate()


# XXX: this still assumes build essentials are installed
if haskey(ENV, "CI")
    run(`apt update`)
    run(`apt install -y build-essential cmake`)
end


using Conda

# install the toolchain
Conda.add("dpcpp_linux-64", :oneapi; channel="intel")
Conda.add("mkl-devel-dpcpp", :oneapi; channel="intel")


using oneAPI_Level_Zero_Headers_jll

# XXX: isn't there a Conda package providing ze_api.hpp?
include_dir =  joinpath(oneAPI_Level_Zero_Headers_jll.artifact_dir, "include")

rpath = String[]
push!(rpath, Conda.lib_dir(:oneapi))

withenv("PATH"=>"$(ENV["PATH"]):$(Conda.bin_dir(:oneapi))",
        "LD_LIBRARY_PATH"=>Conda.lib_dir(:oneapi)) do
mktempdir() do build_dir
mktempdir() do install_dir
    run(```cmake -DCMAKE_CXX_FLAGS="-isystem $include_dir"
                 -DCMAKE_INSTALL_RPATH=$(join(rpath, ';'))
                 -DCMAKE_INSTALL_PREFIX=$install_dir
                 -S $(@__DIR__) -B $build_dir```)
    run(`make -C $build_dir -j $(Sys.CPU_THREADS) VERBOSE=true install`)
    mv(joinpath(install_dir, "lib", "liboneapilib.so"),
       joinpath(@__DIR__, "liboneapilib.so"); force=true)
end
end
end
