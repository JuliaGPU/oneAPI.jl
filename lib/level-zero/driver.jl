export ZeDriver, api_version, properties, ipc_properties, extension_properties

struct ZeDriver
    handle::ze_driver_handle_t
end

Base.unsafe_convert(::Type{ze_driver_handle_t}, drv::ZeDriver) = drv.handle

function api_version(drv::ZeDriver)
    version_ref = Ref{ze_api_version_t}()
    zeDriverGetApiVersion(drv, version_ref)
    unmake_version(version_ref[])
end

function Base.show(io::IO, ::MIME"text/plain", drv::ZeDriver)
    props = properties(drv)
    print(io, "ZeDriver(")
    @printf(io, "%p", drv.handle)   # FIXME: print UUID once available
    print(io, ", version $(props.driverVersion)")
    print(io, ")")
end


## driver iteration

export drivers

struct ZeDrivers
    handles::Vector{ze_driver_handle_t}

    function ZeDrivers()
        count_ref = Ref{UInt32}(0)
        zeDriverGet(count_ref, C_NULL)

        handles = Vector{ze_driver_handle_t}(undef, count_ref[])
        zeDriverGet(count_ref, handles)

        new(handles)
    end
end

drivers() = ZeDrivers()

Base.eltype(::ZeDrivers) = ZeDriver

function Base.iterate(iter::ZeDrivers, i=1)
    i >= length(iter) + 1 ? nothing : (ZeDriver(iter.handles[i]), i+1)
end

Base.length(iter::ZeDrivers) = length(iter.handles)

Base.IteratorSize(::ZeDrivers) = Base.HasLength()


## properties

function properties(drv::ZeDriver)
    props_ref = Ref{ze_driver_properties_t}()
    zeDriverGetProperties(drv, props_ref)

    props = props_ref[]
    return (
        # https://github.com/intel/compute-runtime/issues/321
        # uuid=Base.UUID(reinterpret(UInt128, [props.uuid.id...])[1]),
        driverVersion=VersionNumber((props.driverVersion & 0xFF000000) >> 24,
                                    (props.driverVersion & 0x00FF0000) >> 16,
                                    props.driverVersion & 0x0000FFFF),
    )
end

function ipc_properties(drv::ZeDriver)
    props_ref = Ref{ze_driver_ipc_properties_t}()
    zeDriverGetIpcProperties(drv, props_ref)

    props = props_ref[]
    return (
        flags=props.flags,
    )
end

# FIXME: throws ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
function extension_properties(drv::ZeDriver)
    count_ref = Ref{UInt32}(0)
    zeDriverGetExtensionProperties(drv, count_ref, C_NULL)

    all_props = Vector{ze_driver_extension_properties_t}(undef, count_ref[])
    zeDriverGetExtensionProperties(drv, count_ref, all_props)

    return [(name=String([props.name[1:findfirst(isequal(0), props.name)-1]...]),
             version=Int(props.version),
            ) for props in all_props[1:count_ref[]]]
end
