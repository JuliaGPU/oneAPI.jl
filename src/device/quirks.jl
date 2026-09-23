# throw a device-side exception of type `name`, printing the type and `reason`
macro gputhrow(name::String, reason::String)
    quote
        @println "ERROR: " $name ": " $reason "."
        throw(nothing)
    end
end

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @gputhrow "DomainError" "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @gputhrow "DomainError" "Exponentiation yielding a complex result requires a complex argument"

# intfuncs.jl
@device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"

# checked.jl
@device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @gputhrow "OverflowError" "Binary operation overflowed"

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @gputhrow "InexactError" "Inexact conversion"

# abstractarray.jl
@device_override @noinline Base.throw_boundserror(A, I) =
    @gputhrow "BoundsError" "Out-of-bounds array access"

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @gputhrow "DomainError" "sincos(x) is only defined for finite x"

# diagonal.jl
# Base's version throws an ArgumentError; this one prints the reason
import LinearAlgebra
@device_override function Base.setindex!(D::LinearAlgebra.Diagonal, v, i::Int, j::Int)
    @boundscheck checkbounds(D, i, j)
    if i == j
        @inbounds D.diag[i] = v
    elseif !iszero(v)
        @gputhrow "ArgumentError" "cannot set off-diagonal entry to a nonzero value"
    end
    return v
end

# number.jl
# Base's version throws a BoundsError; this one prints the reason
@device_override @inline function Base.getindex(x::Number, I::Integer...)
    @boundscheck all(isone, I) ||
        @gputhrow "BoundsError" "Out-of-bounds access of scalar value"
    x
end

# From Metal.jl to avoid widemul and Int128
@static if VERSION >= v"1.12.0-DEV.1736" # Partially reverts JuliaLang/julia PR #56750
    const BitInteger64 = Union{Int64, UInt64}
    @device_override function Base.checkbounds(::Type{Bool}, v::StepRange{<:BitInteger64, <:BitInteger64}, i::BitInteger64)
        @inline
        return checkindex(Bool, eachindex(IndexLinear(), v), i)
    end
end
