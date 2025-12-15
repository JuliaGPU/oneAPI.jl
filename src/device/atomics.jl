# Atomic operation device overrides and fallbacks

# Fallback wrappers for Float32 atomic_inc!/atomic_dec!
# Intel Level Zero doesn't support these directly for floating-point types,
# so we implement them using atomic_add!/atomic_sub!

@device_override @inline function SPIRVIntrinsics.atomic_inc!(p::LLVMPtr{Float32, AS}) where {AS}
    SPIRVIntrinsics.atomic_add!(p, Float32(1))
end

@device_override @inline function SPIRVIntrinsics.atomic_dec!(p::LLVMPtr{Float32, AS}) where {AS}
    SPIRVIntrinsics.atomic_sub!(p, Float32(1))
end

# Float64 fallbacks (if Float64 is supported on device)
@device_override @inline function SPIRVIntrinsics.atomic_inc!(p::LLVMPtr{Float64, AS}) where {AS}
    SPIRVIntrinsics.atomic_add!(p, Float64(1))
end

@device_override @inline function SPIRVIntrinsics.atomic_dec!(p::LLVMPtr{Float64, AS}) where {AS}
    SPIRVIntrinsics.atomic_sub!(p, Float64(1))
end
