
function versioninfo(io::IO=stdout)
    if Sys.islinux()
        println(io, "Binary dependencies:")
        for jll in [oneL0.NEO_jll, oneL0.NEO_jll.libigc_jll, oneL0.NEO_jll.gmmlib_jll,
                    SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll]
            name = string(jll)
            print(io, "- $(name[1:end-4]): $(Base.pkgversion(jll))")
            if jll.host_platform !== nothing
                debug = tryparse(Bool, get(jll.host_platform.tags, "debug", "false"))
                if debug === true
                    print(io, " (debug)")
                end
            end
            println(io)
        end
        println(io)
    end

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
