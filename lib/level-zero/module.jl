

export ZeModule

mutable struct ZeModule
    handle::ze_module_handle_t

    context::ZeContext
    device::ZeDevice

    function ZeModule(ctx::ZeContext, dev::ZeDevice, image; build_flags="", log::Bool=true)
        log_ref = if log
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
            desc_ref = Ref(ze_module_desc_t(;
                format=ZE_MODULE_FORMAT_IL_SPIRV,
                inputSize=sizeof(image),
                pInputModule=pointer(image),
                pBuildFlags=pointer(build_flags),
                pConstants=Base.unsafe_convert(Ptr{ze_module_constants_t}, constants)
            ))
            handle_ref = Ref{ze_module_handle_t}()
            res = unsafe_zeModuleCreate(ctx, dev, desc_ref, handle_ref, log_ref)
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
                if res == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE
                    @error """Module compilation failed:
                              $log"""
                else
                    @debug """Build log:
                              $log"""
                end
            end
        end

        if res != RESULT_SUCCESS
            throw_api_error(res)
        end

        obj = new(handle_ref[], ctx, dev)
        finalizer(obj) do obj
            zeModuleDestroy(obj)
        end
        return obj
    end
end

Base.unsafe_convert(::Type{ze_module_handle_t}, mod::ZeModule) = mod.handle

Base.:(==)(a::ZeModule, b::ZeModule) = a.handle == b.handle
Base.hash(e::ZeModule, h::UInt) = hash(e.handle, h)


## kernels

export ZeKernel

mutable struct ZeKernel
    mod::ZeModule
    handle::ze_kernel_handle_t

    function ZeKernel(mod, name)
        GC.@preserve name begin
            desc_ref = Ref(ze_kernel_desc_t(; pKernelName=pointer(name)))
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

Base.:(==)(a::ZeKernel, b::ZeKernel) = a.handle == b.handle
Base.hash(e::ZeKernel, h::UInt) = hash(e.handle, h)


## kernel iteration

export kernels

struct ZeModuleKernelDict <: AbstractDict{String,ZeKernel}
    mod::ZeModule
    names::Vector{String}

    function ZeModuleKernelDict(mod)
        count_ref = Ref{UInt32}(0)
        zeModuleGetKernelNames(mod, count_ref, C_NULL)
        names_ref = Vector{Ptr{Cchar}}(undef, count_ref[])
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
    zeKernelSuggestGroupSize(kernel, global_sz.x, global_sz.y, global_sz.z,
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
    zeKernelSetArgumentValue(args.kernel, index-1, Core.sizeof(value), Base.RefValue(value))
end


## attributes

export indirect_access, indirect_access!, source_attributes

function indirect_access(kernel::ZeKernel)
    flags_ref = Ref{ze_kernel_indirect_access_flags_t}()
    zeKernelGetIndirectAccess(kernel, flags_ref)
    return flags_ref[]
end

indirect_access!(kernel::ZeKernel, flags) = zeKernelSetIndirectAccess(kernel, flags)

function source_attributes(kernel::ZeKernel)
    size_ref = Ref{UInt32}(0)
    zeKernelGetSourceAttributes(kernel, size_ref, C_NULL)

    data = Vector{UInt8}(undef, size_ref[])
    ptr_ref = Ref{Ptr{Cchar}}(pointer(data))
    zeKernelGetSourceAttributes(kernel, size_ref, ptr_ref)
    str = String(data)

    # the attribute string is null-terminated, with attributes separated by space
    return split(str[1:end-1])
end


## properties

export properties

function properties(kernel::ZeKernel)
    props_ref = Ref(ze_kernel_properties_t())
    zeKernelGetProperties(kernel, props_ref)

    props = props_ref[]
    return (
        numKernelArgs=Int(props.numKernelArgs),
        requiredGroupSize=ZeDim3(props.requiredGroupSizeX,
                                 props.requiredGroupSizeY,
                                 props.requiredGroupSizeZ),
        requiredNumSubGroups=Int(props.requiredNumSubGroups),
        requiredSubgroupSize=Int(props.requiredSubgroupSize),
        maxSubgroupSize=Int(props.maxSubgroupSize),
        maxNumSubgroups=Int(props.maxNumSubgroups),
        localMemSize=Int(props.localMemSize),
        privateMemSize=Int(props.privateMemSize),
        spillMemSize=Int(props.spillMemSize),
        kernel_uuid=Base.UUID(reinterpret(UInt128, [props.uuid.kid...])[1]),
        module_uuid=Base.UUID(reinterpret(UInt128, [props.uuid.mid...])[1]),
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
