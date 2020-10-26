# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct oneArrayStyle{N} <: AbstractGPUArrayStyle{N} end
oneArrayStyle(::Val{N}) where N = oneArrayStyle{N}()
oneArrayStyle{M}(::Val{N}) where {N,M} = oneArrayStyle{N}()

BroadcastStyle(::Type{<:oneArray{T,N}}) where {T,N} = oneArrayStyle{N}()

Base.similar(bc::Broadcasted{oneArrayStyle{N}}, ::Type{T}) where {N,T} =
    similar(oneArray{T}, axes(bc))

Base.similar(bc::Broadcasted{oneArrayStyle{N}}, ::Type{T}, dims...) where {N,T} =
    oneArray{T}(undef, dims...)


## replace base functions with libdevice alternatives

zefunc(f) = f
zefunc(::Type{T}) where T = (x...) -> T(x...) # broadcasting type ctors isn't GPU compatible

Broadcast.broadcasted(::oneArrayStyle{N}, f, args...) where {N} =
  Broadcasted{oneArrayStyle{N}}(zefunc(f), args, nothing)

const device_intrinsics = :[
    # math
    acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atan2pi, atanh, atanpi, cbrt,
    ceil, copysign, cos, cosh, cospi, erf, erfc, exp, exp10, exp2, expm1, abs, fdim,
    floor, fma, fmax, fmax, fmax, fmin, fmin, fmin, fmod, hypot, ilogb, ilogb,
    ldexp, lgamma, log, log10, log1p, log2, logb, mad, maxmag, minmag, nan, nextafter,
    pow, pown, powr, remainder, rint, rootn, round, rsqrt, sin, sinh, sinpi, sqrt,
    tan, tanh, tanpi, tgamma, trunc,

for f in device_intrinsics
  isdefined(Base, f) || continue
  @eval zefunc(::typeof(Base.$f)) = $f
end
