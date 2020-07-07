

export ZeModule

mutable struct ZeModule
    handle::ze_module_handle_t
    device::ZeDevice

    function ZeModule(dev, image; build_flags="")
        log_ref = if isdebug(:ZeModule)
            log_ref = Ref{ze_module_build_log_handle_t}()
        else
            C_NULL
        end

        constants = Ref(ze_module_constants_t(
            0,
            C_NULL,
            C_NULL
        ))

        # compile the module
        GC.@preserve image build_flags constants begin
            desc_ref = Ref(ze_module_desc_t(
                ZE_MODULE_DESC_VERSION_CURRENT,
                ZE_MODULE_FORMAT_IL_SPIRV,
                sizeof(image),
                pointer(image),
                pointer(build_flags),
                Base.unsafe_convert(Ptr{ze_module_constants_t}, constants)
            ))
            handle_ref = Ref{ze_module_handle_t}()
            zeModuleCreate(dev, desc_ref, handle_ref, log_ref)
            obj = new(handle_ref[], dev)
        end

        # read the log
        if log_ref !== C_NULL
            log_size_ref = Ref{Csize_t}(0)
            zeModuleBuildLogGetString(log_ref[], log_size_ref, C_NULL)
            log_buf = Vector{UInt8}(undef, log_size_ref[])
            zeModuleBuildLogGetString(log_ref[], log_size_ref, pointer(log_buf))
            zeModuleBuildLogDestroy(log_ref[])

            log = String(log_buf)[1:end-1] # strip null terminator
            if !isempty(log)
                @debug """Build log:
                          $log"""
            end
        end

        finalizer(obj) do obj
            zeModuleDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_module_handle_t}, mod::ZeModule) = mod.handle


## kernels

export ZeKernel

mutable struct ZeKernel
    mod::ZeModule
    handle::ze_kernel_handle_t

    function ZeKernel(mod, name)
        GC.@preserve name begin
            desc_ref = Ref(ze_kernel_desc_t(
                ZE_KERNEL_DESC_VERSION_CURRENT,
                ZE_KERNEL_FLAG_NONE,
                pointer(name)
            ))
            handle_ref = Ref{ze_kernel_handle_t}()
            zeKernelCreate(mod, desc_ref, handle_ref)
        end
        obj = new(mod, handle_ref[])

        finalizer(obj) do obj
            zeKernelDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_kernel_handle_t}, kernel::ZeKernel) = kernel.handle


## kernel iteration

export kernels

struct ZeModuleKernelDict <: AbstractDict{String,ZeKernel}
    mod::ZeModule
    names::Vector{String}

    function ZeModuleKernelDict(mod)
        count_ref = Ref{UInt32}(0)
        zeModuleGetKernelNames(mod, count_ref, C_NULL)
        names_ref = Vector{Cstring}(undef, count_ref[])
        zeModuleGetKernelNames(mod, count_ref, names_ref)
        new(mod, unsafe_string.(names_ref))
    end
end

kernels(mod::ZeModule) = ZeModuleKernelDict(mod)

function Base.iterate(dict::ZeModuleKernelDict, i=1)
    i > length(dict.names) && return nothing
    name = dict.names[i]
    kernel = ZeKernel(dict.mod, name)
    return (Pair{String,ZeKernel}(name, kernel), i+1)
end

Base.length(dict::ZeModuleKernelDict) = length(dict.names)

function Base.get(dict::ZeModuleKernelDict, name::AbstractString, def)
    in(name, dict.names) || return def
    ZeKernel(dict.mod, name)
end


## group sizes

export ZeDim, suggest_groupsize, groupsize!

"""
    ZeDim3(x)

    ZeDim3((x,))
    ZeDim3((x, y))
    ZeDim3((x, y, x))

A type used to specify dimensions, consisting of 3 integers for respectively the `x`, `y`
and `z` dimension. Unspecified dimensions default to `1`.

Often accepted as argument through the `ZeDim` type alias, allowing to pass dimensions as a
plain integer or a tuple without having to construct an explicit `ZeDim3` object.
"""
struct ZeDim3
    x::Int
    y::Int
    z::Int
end

ZeDim3(dims::Integer)             = ZeDim3(dims,    1,       1)
ZeDim3(dims::NTuple{1,<:Integer}) = ZeDim3(dims[1], 1,       1)
ZeDim3(dims::NTuple{2,<:Integer}) = ZeDim3(dims[1], dims[2], 1)
ZeDim3(dims::NTuple{3,<:Integer}) = ZeDim3(dims[1], dims[2], dims[3])

# Type alias for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `ZeDim3((len, 2))`)
const ZeDim = Union{Integer,
                    Tuple{Integer},
                    Tuple{Integer, Integer},
                    Tuple{Integer, Integer, Integer}}

function suggest_groupsize(kernel::ZeKernel, global_sz::ZeDim)
    global_sz = ZeDim3(global_sz)
    group_sz_x = Ref{UInt32}()
    group_sz_y = Ref{UInt32}()
    group_sz_z = Ref{UInt32}()
    zeKernelSuggestGroupSize(kernel, global_sz.x, global_sz.y, global_sz.x,
                             group_sz_x, group_sz_y, group_sz_z)
    return ZeDim3(group_sz_x[], group_sz_y[], group_sz_z[])
end

function groupsize!(kernel::ZeKernel, sz::ZeDim)
    sz = ZeDim3(sz)
    zeKernelSetGroupSize(kernel, sz.x, sz.y, sz.z)
end


## arguments

export arguments

struct ZeKernelArgumentList
    kernel::ZeKernel
end

arguments(kernel::ZeKernel) = ZeKernelArgumentList(kernel)

function Base.setindex!(args::ZeKernelArgumentList, value::Any, index::Integer)
    @assert isbits(value)
    zeKernelSetArgumentValue(args.kernel, index-1, sizeof(value), Base.RefValue(value))
end


## attributes

export attributes

struct ZeKernelAttributeDict <: AbstractDict{ze_kernel_attribute_t,Any}
    kernel::ZeKernel
end

attributes(kernel::ZeKernel) = ZeKernelAttributeDict(kernel)

# list of known attributes, their type, and how to convert them to something Julia
const known_attributes = Dict(
    ZE_KERNEL_ATTR_INDIRECT_HOST_ACCESS    => (ze_bool_t,  Bool),
    ZE_KERNEL_ATTR_INDIRECT_DEVICE_ACCESS  => (ze_bool_t,  Bool),
    ZE_KERNEL_ATTR_INDIRECT_SHARED_ACCESS  => (ze_bool_t,  Bool),
    ZE_KERNEL_ATTR_SOURCE_ATTRIBUTE        => (Any,        val->split(String(val)[1:end-1])),
)

function Base.iterate(dict::ZeKernelAttributeDict, i=1)
    iter = iterate(known_attributes, i)
    iter === nothing && return nothing
    (attr, _), i = iter
    (Pair{ze_kernel_attribute_t,Any}(attr, dict[attr]), i+1)
end

Base.length(::ZeKernelAttributeDict) = length(known_attributes)

function Base.get(dict::ZeKernelAttributeDict, attr::ze_kernel_attribute_t, def)
    haskey(known_attributes, attr) || return def
    typ, conv = known_attributes[attr]
    data = if typ == Any
        # untyped attribute, fetch the size and return an array of bytes
        size_ref = Ref{UInt32}(0)
        zeKernelGetAttribute(dict.kernel, attr, size_ref, C_NULL)
        data = Vector{UInt8}(undef, size_ref[])
        zeKernelGetAttribute(dict.kernel, attr, size_ref, data)
        data
    else
        size_ref = Ref{UInt32}(sizeof(typ))
        ref = Ref{typ}()
        zeKernelGetAttribute(dict.kernel, attr, size_ref, ref)
        ref[]
    end
    return conv(data)
end

function Base.setindex!(dict::ZeKernelAttributeDict, value, attr::ze_kernel_attribute_t)
    typ, conv = known_attributes[attr]
    # NOTE: needs better handling of non-isbits values, but no attrs are like that
    zeKernelSetAttribute(dict.kernel, attr, sizeof(value), Base.RefValue(convert(typ, value)))
end


## properties

export properties

function properties(kernel::ZeKernel)
    props_ref = Ref{ze_kernel_properties_t}()
    unsafe_store!(convert(Ptr{ze_kernel_properties_version_t},
                          Base.unsafe_convert(Ptr{Cvoid}, props_ref)),
                  ZE_KERNEL_PROPERTIES_VERSION_CURRENT)
    zeKernelGetProperties(kernel, props_ref)

    props = props_ref[]
    return (
        name=String([props.name[1:findfirst(isequal(0), props.name)-1]...]),
        numKernelArgs=Int(props.numKernelArgs),
        requiredGroupSize=ZeDim3(props.requiredGroupSizeX,
                                 props.requiredGroupSizeY,
                                 props.requiredGroupSizeZ),
    )
end


## execution

export append_launch!

ze_group_count_t(dim::ZeDim3) = ze_group_count_t(dim.x, dim.y, dim.z)

function append_launch!(list, kernel, group_count::ZeDim, signal_event=nothing,
                        wait_events::ZeEvent...)
    group_count = ze_group_count_t(ZeDim3(group_count))
    zeCommandListAppendLaunchKernel(list, kernel, Ref(group_count),
                                    something(signal_event, C_NULL),
                                    length(wait_events), [wait_events...])
end
