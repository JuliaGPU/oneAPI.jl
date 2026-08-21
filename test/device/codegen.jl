# Compile-only coverage of Base/Math functions on the oneAPI target.
#
# oneAPI.jl has no device `malloc`, so any heap allocation that survives optimization — in
# practice an exception object on a throw path that `src/device/quirks.jl` does not cover —
# fails compilation with
#
#     InvalidIRError: unsupported call to an unknown function (call to gpu_malloc)
#
# Which throw paths survive depends on what the optimizer happens to delete, so a routine
# Julia/GPUCompiler/LLVM bump can silently change the answer (GPUCompiler 2.2.2 did, for
# `sqrt(::Complex)` via `exponent`). This grid compiles each (function, eltype) cell through
# the real pipeline — the same target, method table and validation a launch uses — without
# launching anything, so a regression names its cell instead of failing three layers deep in
# a GPUArrays test.
#
# Cells known to fail are listed in `BROKEN` with the reason; a cell that starts passing is
# reported as an unexpected pass, which is the signal to remove it from the list.

import LinearAlgebra

# `oneAPI.code_llvm` and friends disable IR validation, so go through the validating path
# `zefunction` takes: the job is built for the current device (LTS or rolling back-end, fp
# capabilities, quirks method table) and compiled to SPIR-V, which also exercises the
# translator. Bypasses the kernel cache so cells neither pollute nor hit it.
function compiles(f, tt)
    config = oneAPI.compiler_config(device(); kernel = true)
    job = oneAPI.CompilerJob(oneAPI.methodinstance(typeof(f), tt), config)
    oneAPI.compile_to_obj(job)
    return true
end

# The value flows through memory so the call cannot be folded away.
function smoke_kernel(f, out, x)
    @inbounds out[1] = f(x[1])
    return
end

const DevVec{T} = oneDeviceVector{T, oneAPI.AS.CrossWorkgroup}

# Result type from a host evaluation where possible; the GPU interpreter otherwise (it
# respects the device overrides, unlike the host).
function result_type(f, T)
    R = try
        typeof(f(one(T)))
    catch
        oneAPI.return_type(f, Tuple{T})
    end
    # a throwing host evaluation on a `Union{}`-returning cell is still worth compiling
    return R === Union{} ? Nothing : R
end

# Cells that do not compile today, with the reason. None of these is the `gpu_malloc`
# failure; they are limitations of the SPIR-V code generator the active stack uses (the
# Khronos translator on the LTS stack, the LLVM SPIR-V back-end otherwise, see
# `_compiler_config`). Remove an entry once the cell compiles again.
const BROKEN = Set{Tuple{String, DataType}}(
    if oneL0.LTS[]
        [
            # llvm-spirv: `InvalidBitWidth: 63` — `_cpow` narrows an `Int` to `i63`
            ("x^x", ComplexF16), ("x^x", ComplexF32), ("x^x", ComplexF64),
            # llvm-spirv: `Unexpected llvm intrinsic: llvm.smul.with.overflow`
            ("checked_mul", Int32), ("checked_mul", Int64),
        ]
    else
        [
            # SPIR-V back-end: malformed `select`/`phi` after its own legalization of `_cpow`
            ("x^x", ComplexF16), ("x^x", ComplexF32),
            # SPIR-V back-end: `cannot select: G_SADDO`
            ("checked_add", Int32), ("checked_add", Int64),
        ]
    end
)

function smoke(label, f, T)
    R = result_type(f, T)
    tt = Tuple{typeof(f), DevVec{R}, DevVec{T}}
    broken = (label, T) in BROKEN
    @testset "$label($T)" begin
        if broken
            # the SPIR-V tools print their diagnostics to stderr; known failures stay quiet
            redirect_stderr(devnull) do
                @test compiles(smoke_kernel, tt) broken = true
            end
        else
            @test compiles(smoke_kernel, tt)
        end
    end
    return
end

# Named wrappers: the obvious spelling would take a different Base path (`x^-1` is
# `literal_pow`/`inv`, never `throw_domerr_powbysq`) or needs a second operand.
negpow(x) = x^(-one(x))
powsame(x) = x^x
divsame(x) = x / x
intdiv(x) = div(x, x)
intrem(x) = rem(x, x)
intfld(x) = fld(x, x)
intcld(x) = cld(x, x)
intmod(x) = mod(x, x)
hypotsame(x) = hypot(x, x)
atan2same(x) = atan(x, x)
copysignsame(x) = copysign(x, -x)
checked_add_same(x) = Base.checked_add(x, x)
checked_sub_same(x) = Base.checked_sub(x, x)
checked_mul_same(x) = Base.checked_mul(x, x)
to_int32(x) = Int32(x)
to_int64(x) = Int64(x)
trunc_int32(x) = trunc(Int32, x)
round_int64(x) = round(Int64, x)
to_complexf32(x) = ComplexF32(x)

float_types = DataType[Float32]
float64_supported && push!(float_types, Float64)
float16_supported && push!(float_types, Float16)
complex_types = DataType[Complex{T} for T in float_types]
int_types = DataType[Int32, Int64, UInt32, UInt64]

@testset "unary real" begin
    for (label, f) in (
                ("sqrt", sqrt), ("cbrt", cbrt),
                ("exp", exp), ("exp2", exp2), ("exp10", exp10), ("expm1", expm1),
                ("log", log), ("log2", log2), ("log10", log10), ("log1p", log1p),
                ("sin", sin), ("cos", cos), ("tan", tan),
                ("asin", asin), ("acos", acos), ("atan", atan),
                ("sinh", sinh), ("cosh", cosh), ("tanh", tanh),
                ("abs", abs), ("abs2", abs2), ("sign", sign), ("inv", inv),
                ("exponent", exponent), ("significand", significand), ("frexp", frexp),
                ("trunc", trunc), ("round", round), ("floor", floor), ("ceil", ceil),
            ), T in float_types
        smoke(label, f, T)
    end
end

@testset "unary complex" begin
    for (label, f) in (
                ("sqrt", sqrt), ("exp", exp), ("log", log), ("sin", sin), ("cos", cos),
                ("abs", abs), ("abs2", abs2), ("sign", sign), ("inv", inv), ("angle", angle),
            ), T in complex_types
        smoke(label, f, T)
    end
end

@testset "binary real" begin
    for (label, f) in (
                ("x^x", powsame), ("x/x", divsame), ("hypot", hypotsame), ("atan2", atan2same),
                ("mod", intmod), ("rem", intrem), ("div", intdiv), ("fld", intfld), ("cld", intcld),
                ("copysign", copysignsame),
            ), T in float_types
        smoke(label, f, T)
    end
end

@testset "binary complex" begin
    for (label, f) in (("x^x", powsame), ("x/x", divsame)), T in complex_types
        smoke(label, f, T)
    end
end

@testset "integer" begin
    for (label, f) in (
                ("x^-1", negpow),
                ("div", intdiv), ("rem", intrem), ("fld", intfld), ("mod", intmod),
                ("checked_add", checked_add_same), ("checked_sub", checked_sub_same),
                ("checked_mul", checked_mul_same),
            ), T in int_types
        smoke(label, f, T)
    end
end

@testset "conversions" begin
    for (label, f, T) in (
            ("Int32", to_int32, Float32), ("Int64", to_int64, Float64),
            ("trunc(Int32)", trunc_int32, Float32), ("round(Int64)", round_int64, Float64),
            ("ComplexF32", to_complexf32, ComplexF64),
        )
        (T == Float64 || T == ComplexF64) && !float64_supported && continue
        smoke(label, f, T)
    end
end

# Bounds-checked array access (the `throw_boundserror` quirk) and the `Diagonal`
# `setindex!` quirk; compile only, never launched.
function checked_index_kernel(out, x)
    out[2] = x[2]
    return
end
function diagonal_setindex_kernel(out, x)
    D = LinearAlgebra.Diagonal(x)
    D[1, 2] = x[1]
    out[1] = D[1, 1]
    return
end

@testset "array" begin
    for T in (Int32, Float32)
        @testset "checked indexing($T)" begin
            @test compiles(checked_index_kernel, Tuple{DevVec{T}, DevVec{T}})
        end
        @testset "Diagonal setindex!($T)" begin
            @test compiles(diagonal_setindex_kernel, Tuple{DevVec{T}, DevVec{T}})
        end
    end
end
