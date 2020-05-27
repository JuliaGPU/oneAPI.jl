export ZeDriver, api_version, properties

struct ZeDriver
    handle::ze_driver_handle_t
end

Base.unsafe_convert(::Type{ze_driver_handle_t}, drv::ZeDriver) = drv.handle

function api_version(drv::ZeDriver)
    version_ref = Ref{ze_api_version_t}()
    zeDriverGetApiVersion(drv, version_ref)
    unmake_version(version_ref[])
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
