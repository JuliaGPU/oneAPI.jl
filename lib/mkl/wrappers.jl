using CxxWrap

@readmodule(joinpath(@__DIR__, "../../deps/liboneapilib.so"), :define_module_mkl)
@wraptypes
CxxWrap.argument_overloads(t::Type{<:ZeCxxPtr{T}}) where {T} = [oneArray{T}]
@wrapfunctions

# XXX: CxxWrap makes use a C++-specific wrapper for ZePtr, ZeCxxPtr,
#      so forward all relevant calls to the underlying Level 0 pointer.
Base.cconvert(::Type{<:ZeCxxPtr}, x) = x
Base.unsafe_convert(::Type{<:ZeCxxPtr{T}}, x::oneArray{T}) where {T} =
    ZeCxxPtr{T}(reinterpret(Ptr{T}, Base.unsafe_convert(ZePtr{T}, x)))

function __init__()
  @initcxx
end
