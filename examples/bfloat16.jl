using oneAPI, Test

@static if !isdefined(Core, :BFloat16)
    @info "BFloat16 requires Julia 1.12+, skipping."
    exit()
end

bfloat16_supported = oneAPI._device_supports_bfloat16()

@info "BFloat16 support: $bfloat16_supported"

if !bfloat16_supported
    @info "Device does not support BFloat16, skipping."
    exit()
end

# Conversions: Core.BFloat16 in Julia 1.12 may not have Float32 constructors yet
float32_to_bf16(x::Float32) = reinterpret(Core.BFloat16, (reinterpret(UInt32, x) >> 16) % UInt16)
bf16_to_float32(x::Core.BFloat16) = reinterpret(Float32, UInt32(reinterpret(UInt16, x)) << 16)

# Simple kernel: scale BFloat16 values by 2 via Float32 round-trip
# (BFloat16 arithmetic is done by promoting to Float32 on device)
function scale_bf16(input, output)
    i = get_global_id()
    @inbounds begin
        val = reinterpret(UInt16, input[i])
        # BFloat16 -> Float32: shift left 16 bits
        f = reinterpret(Float32, UInt32(val) << 16)
        f *= 2.0f0
        # Float32 -> BFloat16: take upper 16 bits
        output[i] = reinterpret(Core.BFloat16, (reinterpret(UInt32, f) >> 16) % UInt16)
    end
    return
end

n = 1024
a = float32_to_bf16.(rand(Float32, n))

d_a = oneArray(a)
d_out = oneArray{Core.BFloat16}(undef, n)

@oneapi items=n scale_bf16(d_a, d_out)
result = Array(d_out)

# Verify: each output should be 2x the input (in Float32 space)
result_f32 = bf16_to_float32.(result)
expected_f32 = bf16_to_float32.(a) .* 2.0f0
@test result_f32 ≈ expected_f32
@info "BFloat16 scale-by-2 kernel passed!"
