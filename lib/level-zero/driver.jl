export ZeDriver, api_version, properties, ipc_properties, extension_properties

struct ZeDriver
    handle::ze_driver_handle_t

    # only accept handles, don't convert
    ZeDriver(handle::ze_driver_handle_t) = new(handle)
end

Base.unsafe_convert(::Type{ze_driver_handle_t}, drv::ZeDriver) = drv.handle

Base.:(==)(a::ZeDriver, b::ZeDriver) = a.handle == b.handle
Base.hash(e::ZeDriver, h::UInt) = hash(e.handle, h)

function api_version(drv::ZeDriver)
    version_ref = Ref{ze_api_version_t}()
    zeDriverGetApiVersion(drv, version_ref)
    unmake_version(version_ref[])
end

function Base.show(io::IO, drv::ZeDriver)
    props = properties(drv)
    print(io, "ZeDriver($(props.uuid))")
end

function Base.show(io::IO, ::MIME"text/plain", drv::ZeDriver)
    show(io, drv)
    props = properties(drv)
    print(io, ": version $(props.driverVersion)")
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

function Base.show(io::IO, mime::MIME"text/plain", iter::ZeDrivers)
    print(io, "ZeDriver iterator for $(length(iter)) drivers")
    if !isempty(iter)
        print(io, ":")
        for (i,drv) in enumerate(iter)
            print(io, "\n$(i). ")
            show(io, mime, drv)
        end
    end
end

Base.getindex(iter::ZeDrivers, i::Integer) = ZeDriver(iter.handles[i])



## properties

function properties(drv::ZeDriver)
    props_ref = Ref(ze_driver_properties_t())
    zeDriverGetProperties(drv, props_ref)

    props = props_ref[]
    return (
        uuid=Base.UUID(reinterpret(UInt128, [props.uuid.id...])[1]),
        driverVersion=VersionNumber((props.driverVersion & 0xFF000000) >> 24,
                                    (props.driverVersion & 0x00FF0000) >> 16,
                                    props.driverVersion & 0x0000FFFF),
    )
end

function ipc_properties(drv::ZeDriver)
    props_ref = Ref(ze_driver_ipc_properties_t())
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
