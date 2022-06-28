# Math Functions

# TODO: vector types
const generic_types = [Cfloat,Cdouble]
const generic_types_float = [Cfloat]
const generic_types_double = [Cdouble]


# generically typed

for gentype in generic_types
@eval begin

@device_override Base.acos(x::$gentype) = @builtin_ccall("acos", $gentype, ($gentype,), x)
@device_override Base.acosh(x::$gentype) = @builtin_ccall("acosh", $gentype, ($gentype,), x)
@device_function acospi(x::$gentype) = @builtin_ccall("acospi", $gentype, ($gentype,), x)

@device_override Base.asin(x::$gentype) = @builtin_ccall("asin", $gentype, ($gentype,), x)
@device_override Base.asinh(x::$gentype) = @builtin_ccall("asinh", $gentype, ($gentype,), x)
@device_function asinpi(x::$gentype) = @builtin_ccall("asinpi", $gentype, ($gentype,), x)

@device_override Base.atan(y_over_x::$gentype) = @builtin_ccall("atan", $gentype, ($gentype,), y_over_x)
@device_override Base.atan(y::$gentype, x::$gentype) = @builtin_ccall("atan2", $gentype, ($gentype, $gentype), y, x)
@device_override Base.atanh(x::$gentype) = @builtin_ccall("atanh", $gentype, ($gentype,), x)
@device_function atanpi(x::$gentype) = @builtin_ccall("atanpi", $gentype, ($gentype,), x)
@device_function atanpi(y::$gentype, x::$gentype) = @builtin_ccall("atan2pi", $gentype, ($gentype, $gentype), y, x)

@device_override Base.cbrt(x::$gentype) = @builtin_ccall("cbrt", $gentype, ($gentype,), x)

@device_override Base.ceil(x::$gentype) = @builtin_ccall("ceil", $gentype, ($gentype,), x)

@device_override Base.copysign(x::$gentype, y::$gentype) = @builtin_ccall("copysign", $gentype, ($gentype, $gentype), x, y)

@device_override Base.cos(x::$gentype) = @builtin_ccall("cos", $gentype, ($gentype,), x)
@device_override Base.cosh(x::$gentype) = @builtin_ccall("cosh", $gentype, ($gentype,), x)
@device_function cospi(x::$gentype) = @builtin_ccall("cospi", $gentype, ($gentype,), x)

@device_override SpecialFunctions.erfc(x::$gentype) = @builtin_ccall("erfc", $gentype, ($gentype,), x)
@device_override SpecialFunctions.erf(x::$gentype) = @builtin_ccall("erf", $gentype, ($gentype,), x)

@device_override Base.exp(x::$gentype) = @builtin_ccall("exp", $gentype, ($gentype,), x)
@device_override Base.exp2(x::$gentype) = @builtin_ccall("exp2", $gentype, ($gentype,), x)
@device_override Base.exp10(x::$gentype) = @builtin_ccall("exp10", $gentype, ($gentype,), x)
@device_override Base.expm1(x::$gentype) = @builtin_ccall("expm1", $gentype, ($gentype,), x)

@device_override Base.abs(x::$gentype) = @builtin_ccall("fabs", $gentype, ($gentype,), x)

@device_function dim(x::$gentype, y::$gentype) = @builtin_ccall("fdim", $gentype, ($gentype, $gentype), x, y)

@device_override Base.floor(x::$gentype) = @builtin_ccall("floor", $gentype, ($gentype,), x)

@device_override Base.fma(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("fma", $gentype, ($gentype, $gentype, $gentype), a, b, c)

@device_override Base.max(x::$gentype, y::$gentype) = @builtin_ccall("fmax", $gentype, ($gentype, $gentype), x, y)

@device_override Base.min(x::$gentype, y::$gentype) = @builtin_ccall("fmin", $gentype, ($gentype, $gentype), x, y)

# NOTE: Julia's mod behaves differently than fmod
#@device_override Base.mod(x::$gentype, y::$gentype) = @builtin_ccall("fmod", $gentype, ($gentype, $gentype), x, y)
# fract(x::$gentype, $gentype *iptr) = @builtin_ccall("fract", $gentype, ($gentype, $gentype *), x, iptr)

@device_override Base.hypot(x::$gentype, y::$gentype) = @builtin_ccall("hypot", $gentype, ($gentype, $gentype), x, y)

@device_override SpecialFunctions.loggamma(x::$gentype) = @builtin_ccall("lgamma", $gentype, ($gentype,), x)

@device_override Base.log(x::$gentype) = @builtin_ccall("log", $gentype, ($gentype,), x)
@device_override Base.log2(x::$gentype) = @builtin_ccall("log2", $gentype, ($gentype,), x)
@device_override Base.log10(x::$gentype) = @builtin_ccall("log10", $gentype, ($gentype,), x)
@device_override Base.log1p(x::$gentype) = @builtin_ccall("log1p", $gentype, ($gentype,), x)
@device_function logb(x::$gentype) = @builtin_ccall("logb", $gentype, ($gentype,), x)

@device_function mad(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("mad", $gentype, ($gentype, $gentype, $gentype), a, b, c)

@device_function maxmag(x::$gentype, y::$gentype) = @builtin_ccall("maxmag", $gentype, ($gentype, $gentype), x, y)
@device_function minmag(x::$gentype, y::$gentype) = @builtin_ccall("minmag", $gentype, ($gentype, $gentype), x, y)

# modf(x::$gentype, $gentype *iptr) = @builtin_ccall("modf", $gentype, ($gentype, $gentype *), x, iptr)

@device_function nextafter(x::$gentype, y::$gentype) = @builtin_ccall("nextafter", $gentype, ($gentype, $gentype), x, y)

@device_override Base.:(^)(x::$gentype, y::$gentype) = @builtin_ccall("pow", $gentype, ($gentype, $gentype), x, y)
@device_function powr(x::$gentype, y::$gentype) = @builtin_ccall("powr", $gentype, ($gentype, $gentype), x, y)

@device_override Base.rem(x::$gentype, y::$gentype) = @builtin_ccall("remainder", $gentype, ($gentype, $gentype), x, y)

@device_function rint(x::$gentype) = @builtin_ccall("rint", $gentype, ($gentype,), x)

@device_override Base.round(x::$gentype) = @builtin_ccall("round", $gentype, ($gentype,), x)

@device_function rsqrt(x::$gentype) = @builtin_ccall("rsqrt", $gentype, ($gentype,), x)

@device_override Base.sin(x::$gentype) = @builtin_ccall("sin", $gentype, ($gentype,), x)
# sincos(x::$gentype, $gentype *cosval) = @builtin_ccall("sincos", $gentype, ($gentype, $gentype *), x, cosval)
@device_override Base.sinh(x::$gentype) = @builtin_ccall("sinh", $gentype, ($gentype,), x)
@device_function sinpi(x::$gentype) = @builtin_ccall("sinpi", $gentype, ($gentype,), x)

@device_override Base.sqrt(x::$gentype) = @builtin_ccall("sqrt", $gentype, ($gentype,), x)

@device_override Base.tan(x::$gentype) = @builtin_ccall("tan", $gentype, ($gentype,), x)
@device_override Base.tanh(x::$gentype) = @builtin_ccall("tanh", $gentype, ($gentype,), x)
@device_function tanpi(x::$gentype) = @builtin_ccall("tanpi", $gentype, ($gentype,), x)

@device_override SpecialFunctions.gamma(x::$gentype) = @builtin_ccall("tgamma", $gentype, ($gentype,), x)

@device_override Base.trunc(x::$gentype) = @builtin_ccall("trunc", $gentype, ($gentype,), x)

end
end


# generically typed -- only floats

for gentypef in generic_types_float

if gentypef !== Cfloat
@eval begin
@device_override Base.max(x::$gentypef, y::Cfloat) = @builtin_ccall("fmax", $gentypef, ($gentypef, Cfloat), x, y)
@device_override Base.min(x::$gentypef, y::Cfloat) = @builtin_ccall("fmin", $gentypef, ($gentypef, Cfloat), x, y)
end
end

end


# generically typed -- only doubles

for gentyped in generic_types_double

if gentyped !== Cdouble
@eval begin
@device_override Base.min(x::$gentyped, y::Cdouble) = @builtin_ccall("fmin", $gentyped, ($gentyped, Cdouble), x, y)
@device_override Base.max(x::$gentyped, y::Cdouble) = @builtin_ccall("fmax", $gentyped, ($gentyped, Cdouble), x, y)
end
end

end


# specifically typed

# frexp(x::Cfloat{n}, Cint{n} *exp) = @builtin_ccall("frexp", Cfloat{n}, (Cfloat{n}, Cint{n} *), x, exp)
# frexp(x::Cfloat, Cint *exp) = @builtin_ccall("frexp", Cfloat, (Cfloat, Cint *), x, exp)
# frexp(x::Cdouble{n}, Cint{n} *exp) = @builtin_ccall("frexp", Cdouble{n}, (Cdouble{n}, Cint{n} *), x, exp)
# frexp(x::Cdouble, Cint *exp) = @builtin_ccall("frexp", Cdouble, (Cdouble, Cint *), x, exp)

# ilogb(x::Cfloat{n}) = @builtin_ccall("ilogb", Cint{n}, (Cfloat{n},), x)
@device_function ilogb(x::Cfloat) = @builtin_ccall("ilogb", Cint, (Cfloat,), x)
# ilogb(x::Cdouble{n}) = @builtin_ccall("ilogb", Cint{n}, (Cdouble{n},), x)
@device_function ilogb(x::Cdouble) = @builtin_ccall("ilogb", Cint, (Cdouble,), x)

# ldexp(x::Cfloat{n}, k::Cint{n}) = @builtin_ccall("ldexp", Cfloat{n}, (Cfloat{n}, Cint{n}), x, k)
# ldexp(x::Cfloat{n}, k::Cint) = @builtin_ccall("ldexp", Cfloat{n}, (Cfloat{n}, Cint), x, k)
@device_override Base.ldexp(x::Cfloat, k::Cint) = @builtin_ccall("ldexp", Cfloat, (Cfloat, Cint), x, k)
# ldexp(x::Cdouble{n}, k::Cint{n}) = @builtin_ccall("ldexp", Cdouble{n}, (Cdouble{n}, Cint{n}), x, k)
# ldexp(x::Cdouble{n}, k::Cint) = @builtin_ccall("ldexp", Cdouble{n}, (Cdouble{n}, Cint), x, k)
@device_override Base.ldexp(x::Cdouble, k::Cint) = @builtin_ccall("ldexp", Cdouble, (Cdouble, Cint), x, k)

# lgamma_r(x::Cfloat{n}, Cint{n} *signp) = @builtin_ccall("lgamma_r", Cfloat{n}, (Cfloat{n}, Cint{n} *), x, signp)
# lgamma_r(x::Cfloat, Cint *signp) = @builtin_ccall("lgamma_r", Cfloat, (Cfloat, Cint *), x, signp)
# lgamma_r(x::Cdouble{n}, Cint{n} *signp) = @builtin_ccall("lgamma_r", Cdouble{n}, (Cdouble{n}, Cint{n} *), x, signp)
# Cdouble lgamma_r(x::Cdouble, Cint *signp) = @builtin_ccall("lgamma_r", Cdouble, (Cdouble, Cint *), x, signp)

# nan(nancode::uintn) = @builtin_ccall("nan", Cfloat{n}, (uintn,), nancode)
@device_function nan(nancode::Cuint) = @builtin_ccall("nan", Cfloat, (Cuint,), nancode)
# nan(nancode::Culong{n}) = @builtin_ccall("nan", Cdouble{n}, (Culong{n},), nancode)
@device_function nan(nancode::Culong) = @builtin_ccall("nan", Cdouble, (Culong,), nancode)

# pown(x::Cfloat{n}, y::Cint{n}) = @builtin_ccall("pown", Cfloat{n}, (Cfloat{n}, Cint{n}), x, y)
@device_override Base.:(^)(x::Cfloat, y::Cint) = @builtin_ccall("pown", Cfloat, (Cfloat, Cint), x, y)
# pown(x::Cdouble{n}, y::Cint{n}) = @builtin_ccall("pown", Cdouble{n}, (Cdouble{n}, Cint{n}), x, y)
@device_override Base.:(^)(x::Cdouble, y::Cint) = @builtin_ccall("pown", Cdouble, (Cdouble, Cint), x, y)

# remquo(x::Cfloat{n}, y::Cfloat{n}, Cint{n} *quo) = @builtin_ccall("remquo", Cfloat{n}, (Cfloat{n}, Cfloat{n}, Cint{n} *), x, y, quo)
# remquo(x::Cfloat, y::Cfloat, Cint *quo) = @builtin_ccall("remquo", Cfloat, (Cfloat, Cfloat, Cint *), x::Cfloat, y, quo)
# remquo(x::Cdouble{n}, y::Cdouble{n}, Cint{n} *quo) = @builtin_ccall("remquo", Cdouble{n}, (Cdouble{n}, Cdouble{n}, Cint{n} *), x, y, quo)
# remquo(x::Cdouble, y::Cdouble, Cint *quo) = @builtin_ccall("remquo", Cdouble, (Cdouble, Cdouble, Cint *), x, y, quo)

# rootn(x::Cfloat{n}, y::Cint{n}) = @builtin_ccall("rootn", Cfloat{n}, (Cfloat{n}, Cint{n}), x, y)
@device_function rootn(x::Cfloat, y::Cint) = @builtin_ccall("rootn", Cfloat, (Cfloat, Cint), x, y)
# rootn(x::Cdouble{n}, y::Cint{n}) = @builtin_ccall("rootn", Cdouble{n}, (Cdouble{n}, Cint{n}), x, y)
# rootn(x::Cdouble, y::Cint) = @builtin_ccall("rootn", Cdouble{n}, (Cdouble, Cint), x, y)


# TODO: half and native
