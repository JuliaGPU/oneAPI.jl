module oneL0

using ..APIUtils

using CEnum

using Printf

using NEO_jll
using oneAPI_Level_Zero_Loader_jll

include("utils.jl")
include("pointer.jl")

# core API
include("libze.jl")

# level zero's structure types are often assumed to be zero-initialized (`= {}` in C).
# Julia's memory is not, so define default constructors that ensure everything is zero.
#
# at the same time, our structs are immutable, so we can't just memset to 0 and set fields.
# so instead the constructor we generate has keyword arguments for every fields, and the
# default value for every field is set to 0 (can be overridden by defining `zeroinit`).
#
# TODO: is it really required to (1) zero-initialize memory and (2) set the stype field?
#       none of these are documented...
for (structure_type_enum, _) in CEnum.name_value_pairs(ze_structure_type_t)
    structure_type_name = string(structure_type_enum)
    @assert startswith(structure_type_name, "ZE_STRUCTURE_TYPE")

    T = Symbol("ze_" * lowercase(structure_type_name[19:end]) * "_t")
    if isdefined(oneL0, T)
        struct_typ = getfield(oneL0, T)

        args = Expr[]
        for field in fieldnames(struct_typ)
            field_type = fieldtype(struct_typ, field)
            field_value = if field_type == ze_structure_type_t
                :($structure_type_enum)
            else
                :(zeroinit($field_type))
            end
            push!(args, Expr(:kw, field, field_value))
        end

        @eval begin
            @inline function $T(;$(args...))
                $(T)($(fieldnames(struct_typ)...))
            end
        end
    end
end

# alternative approach: make the structs mutable, and do memset + setfield
zeroinit(::Type{T}) where {T} = convert(T, 0)
zeroinit(::Type{T}) where {T<:CEnum.Cenum} = T(0)
zeroinit(::Type{T}) where {T<:NTuple} = ntuple(_->zeroinit(T.parameters[1]), length(T.parameters))
zeroinit(::Type{ze_driver_uuid_t}) = ze_driver_uuid_t(ntuple(_->zero(UInt8), 16))
zeroinit(::Type{ze_device_uuid_t}) = ze_device_uuid_t(ntuple(_->zero(UInt8), 16))
zeroinit(::Type{_ze_native_kernel_uuid_t}) =
    _ze_native_kernel_uuid_t(ntuple(_->zero(UInt8), 16))
zeroinit(::Type{ze_kernel_uuid_t}) =
    ze_kernel_uuid_t(ntuple(_->zero(UInt8), 16), ntuple(_->zero(UInt8), 16))

# core wrappers
include("error.jl")
include("common.jl")
include("driver.jl")
include("device.jl")
include("context.jl")
include("cmdqueue.jl")
include("cmdlist.jl")
include("fence.jl")
include("event.jl")
include("barrier.jl")
include("module.jl")
include("memory.jl")
include("copy.jl")
include("residency.jl")

const functional = Ref{Bool}(false)

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return

    if !oneAPI_Level_Zero_Loader_jll.is_available()
        @error """No oneAPI Level Zero loader found for your platform. Currently, only Linux x86 is supported.
                  If you have a local oneAPI toolchain, you can use that; refer to the documentation for more details."""
        return
    end

    if !NEO_jll.is_available()
        @error """No oneAPI driver found for your platform. Currently, only Linux x86_64 is supported.
                  If you have a local oneAPI toolchain, you can use that; refer to the documentation for more details."""
        return
    end

    try
        zeInit(0)
        functional[] = true
    catch err
        @error "Failed to initialize oneAPI" exception=(err,catch_backtrace())
        functional[] = false
        return
    end
end

end
