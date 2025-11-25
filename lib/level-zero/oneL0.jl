module oneL0

using ..APIUtils

using CEnum

using Printf

using Libdl

if Sys.iswindows()
    const libze_loader = "ze_loader"
else
    using NEO_jll
    using oneAPI_Level_Zero_Loader_jll
end

include("utils.jl")
include("pointer.jl")

# core API
include("common.jl")
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
# TODO: add support for conveniently linking extension objects in pNext
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

# link extension objects in pNext
function link_extensions(refs...)
    length(refs) >= 2 || return
    for (parent, child) in zip(refs[1:end-1], refs[2:end])
        pNext = Base.unsafe_convert(Ptr{Cvoid}, child)
        typ = eltype(parent)
        @assert fieldnames(typ)[2] == :pNext
        field = Base.unsafe_convert(Ptr{Cvoid}, parent) + fieldoffset(typ, 2)
        field = convert(Ptr{Ptr{Cvoid}}, field)
        unsafe_store!(field, pNext)
    end
    return
end

# core wrappers
include("error.jl")
include("driver.jl")
include("device.jl")

# Define OutOfGPUMemoryError after device.jl to ensure ZeDevice is available
export OutOfGPUMemoryError

"""
    OutOfGPUMemoryError(sz::Integer=0, dev::ZeDevice)

An operation allocated too much GPU memory.
"""
struct OutOfGPUMemoryError <: Exception
  sz::Int
  dev::Union{ZeDevice, Nothing}

  function OutOfGPUMemoryError(sz::Integer=0, dev::Union{ZeDevice, Nothing}=nothing)
    new(sz, dev)
  end
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
      print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    if err.dev !== nothing
        print(" on device $(properties(err.dev).name)")
        if length(memory_properties(err.dev)) == 1
            # XXX: how to handle multiple memories?
            print(" with $(Base.format_bytes(only(memory_properties(err.dev)).totalSize))")
        end
    end
    return io
end

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

const validation_layer = Ref{Bool}()
const parameter_validation = Ref{Bool}()

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return

    if Sys.iswindows()
        if Libdl.dlopen(libze_loader; throw_error=false) === nothing
            @error "The oneAPI Level Zero loader was not found. Please ensure the Intel GPU drivers are installed."
            return
        end
    else
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
    end

    try
        zeInit(0)
    catch err
        # Handle the specific case where no oneAPI device is available
        if err isa ZeError && err.code == RESULT_ERROR_UNINITIALIZED
            functional[] = false
            return
        end
        # For other errors, still report them as errors
        @error "Failed to initialize oneAPI" exception=(err,catch_backtrace())
        functional[] = false
        return
    end

    # Check if there are actually any drivers/devices available
    try
        drv_count = Ref{UInt32}(0)
        zeDriverGet(drv_count, C_NULL)
        if drv_count[] == 0
            @info "oneAPI initialized but no drivers found. oneAPI.jl will not be functional."
            functional[] = false
            return
        end
    catch err
        @error "Failed to enumerate oneAPI drivers" exception = (err, catch_backtrace())
        functional[] = false
        return
    end

    functional[] = true

    validation_layer[] = parse(Bool, get(ENV, "ZE_ENABLE_VALIDATION_LAYER", "false"))
    parameter_validation[] = parse(Bool, get(ENV, "ZE_ENABLE_PARAMETER_VALIDATION", "false"))
end

end
