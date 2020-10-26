# Integer Functions

# TODO: vector types
const generic_integer_types = [Cchar, Cuchar, Cshort, Cushort, Cint, Cuint, Clong, Culong]


# generically typed

for gentype in generic_integer_types
@eval begin

abs(x::$gentype) = @builtin_ccall("abs", $gentype, ($gentype,), x)
abs_diff(x::$gentype, y::$gentype) = @builtin_ccall("abs_diff", $gentype, ($gentype, $gentype), x, y)

add_sat(x::$gentype, y::$gentype) = @builtin_ccall("add_sat", $gentype, ($gentype, $gentype), x, y)
hadd(x::$gentype, y::$gentype) = @builtin_ccall("hadd", $gentype, ($gentype, $gentype), x, y)
rhadd(x::$gentype, y::$gentype) = @builtin_ccall("rhadd", $gentype, ($gentype, $gentype), x, y)

clamp(x::$gentype, minval::$gentype, maxval::$gentype) = @builtin_ccall("clamp", $gentype, ($gentype, $gentype, $gentype), x, minval, maxval)

clz(x::$gentype) = @builtin_ccall("clz", $gentype, ($gentype,), x)
ctz(x::$gentype) = @builtin_ccall("ctz", $gentype, ($gentype,), x)

mad_hi(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("mad_hi", $gentype, ($gentype, $gentype, $gentype), a, b, c)
mad_sat(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("mad_sat", $gentype, ($gentype, $gentype, $gentype), a, b, c)

max(x::$gentype) = @builtin_ccall("max", $gentype, ($gentype,), x)
min(x::$gentype) = @builtin_ccall("min", $gentype, ($gentype,), x)

mul_hi(x::$gentype, y::$gentype) = @builtin_ccall("mul_hi", $gentype, ($gentype, $gentype), x, y)

rotate(v::$gentype, i::$gentype) = @builtin_ccall("rotate", $gentype, ($gentype, $gentype), v, i)

sub_sat(x::$gentype, y::$gentype) = @builtin_ccall("sub_sat", $gentype, ($gentype, $gentype), x, y)

popcount(x::$gentype) = @builtin_ccall("popcount", $gentype, ($gentype,), x)

mad24(x::$gentype, y::$gentype, z::$gentype) = @builtin_ccall("mad24", $gentype, ($gentype, $gentype, $gentype), x, y, z)
mul24(x::$gentype, y::$gentype) = @builtin_ccall("mul24", $gentype, ($gentype, $gentype), x, y)

end
end


# specifically typed

upsample(hi::Cchar, lo::Cuchar) = @builtin_ccall("upsample", Cshort, (Cchar, Cuchar), hi, lo)
upsample(hi::Cuchar, lo::Cuchar) = @builtin_ccall("upsample", Cushort, (Cuchar, Cuchar), hi, lo)
upsample(hi::Cshort, lo::Cushort) = @builtin_ccall("upsample", Cint, (Cshort, Cushort), hi, lo)
upsample(hi::Cushort, lo::Cushort) = @builtin_ccall("upsample", Cuint, (Cushort, Cushort), hi, lo)
upsample(hi::Cint, lo::Cuint) = @builtin_ccall("upsample", Clong, (Cint, Cuint), hi, lo)
upsample(hi::Cuint, lo::Cuint) = @builtin_ccall("upsample", Culong, (Cuint, Cuint), hi, lo)
