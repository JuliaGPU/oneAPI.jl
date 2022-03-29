
function versioninfo(io::IO=stdout)
    # get a hold of Pkg without adding a dependency on the package
    Pkg = let
        id = Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg")
        Base.loaded_modules[id]
    end
    deps = Pkg.dependencies()
    versions = Dict(map(uuid->deps[uuid].name => deps[uuid].version, collect(keys(deps))))

    println(io, "Binary dependencies:")
    for pkg in ["NEO_jll", "libigc_jll", "gmmlib_jll", "SPIRV_LLVM_Translator_unified_jll", "SPIRV_Tools_jll"]
        println(io, "- $pkg: $(versions[pkg])")
    end
    println(io)

    println(io, "Toolchain:")
    println(io, "- Julia: $VERSION")
    println(io, "- LLVM: $(LLVM.version())")
    println(io)

    env = filter(var->startswith(var, "JULIA_ONEAPI"), keys(ENV))
    if !isempty(env)
        println(io, "Environment:")
        for var in env
            println(io, "- $var: $(ENV[var])")
        end
        println(io)
    end

    drvs = drivers()
    if isempty(drvs)
        println(io, "No oneAPI-capable drivers.")
    elseif length(drvs) == 1
        println(io, "1 driver:")
    else
        println(io, length(drvs), " drivers:")
    end
    for drv in drivers()
        props = properties(drv)
        println(io, "- $(props.uuid) (v$(props.driverVersion), API v$(api_version(drv)))")
    end
    println(io)

    devs = [dev for drv in drivers() for dev in devices(drv)]
    if isempty(devs)
        println(io, "No oneAPI-capable devices.")
    elseif length(devs) == 1
        println(io, "1 device:")
    else
        println(io, length(devs), " devices:")
    end
    for dev in devs
        props = properties(dev)
        println(io, "- $(props.name)")
    end
end

"""
    @sync ex

Run expression `ex` and synchronize the GPU afterwards.

See also: [`synchronize`](@ref).
"""
macro sync(ex)
    quote
        local ret = $(esc(ex))
        synchronize()
        ret
    end
end
