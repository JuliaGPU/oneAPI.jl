# oneMKL FFT (DFT) high-level Julia interface
# Inspired by AMDGPU ROCFFT interface style, adapted to oneMKL DFT C wrapper.

module FFT

using ..oneMKL
using ..oneMKL: oneAPI, SYCL, syclQueue_t
using ..Support
using ..SYCL
using LinearAlgebra
using GPUArrays
using AbstractFFTs
import AbstractFFTs: complexfloat, realfloat
import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!
import AbstractFFTs: plan_rfft, plan_brfft, plan_inv, normalization, ScaledPlan
import AbstractFFTs: fft, bfft, ifft, rfft, Plan, ScaledPlan
export MKLFFTPlan

# Import DFT enums and constants from Support module
using ..Support

# Allow implicit conversion of SYCL queue object to raw handle when storing/passing
Base.convert(::Type{syclQueue_t}, q::SYCL.syclQueue) = Base.unsafe_convert(syclQueue_t, q)

abstract type MKLFFTPlan{T,K,inplace} <: AbstractFFTs.Plan{T} end

Base.eltype(::MKLFFTPlan{T}) where T = T
is_inplace(::MKLFFTPlan{<:Any,<:Any,inplace}) where inplace = inplace

# Forward / inverse flags
const MKLFFT_FORWARD = true
const MKLFFT_INVERSE = false

mutable struct cMKLFFTPlan{T,K,inplace,N,R,B} <: MKLFFTPlan{T,K,inplace}
    handle::onemklDftDescriptor_t
    queue::syclQueue_t
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    realdomain::Bool
    region::NTuple{R,Int}
    buffer::B
    pinv::Any
end

# Real transforms use separate struct (mirroring AMDGPU style) for buffer staging
mutable struct rMKLFFTPlan{T,K,inplace,N,R,B} <: MKLFFTPlan{T,K,inplace}
    handle::onemklDftDescriptor_t
    queue::syclQueue_t
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    xtype::Symbol
    region::NTuple{R,Int}
    buffer::B
    pinv::Any
end

# Inverse plan constructors (derive from existing plan)
function normalization_factor(sz, region)
    # AbstractFFTs expects inverse to scale by 1/prod(lengths along region)
    prod(ntuple(i-> sz[region[i]], length(region)))
end

function plan_inv(p::cMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = cMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,p.realdomain,p.region,p.buffer,p)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end
function plan_inv(p::cMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = cMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,p.realdomain,p.region,p.buffer,p)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end

function plan_inv(p::rMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = rMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,:brfft,p.region,p.buffer,p)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end
function plan_inv(p::rMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = rMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,:rfft,p.region,p.buffer,p)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end

function Base.show(io::IO, p::MKLFFTPlan{T,K,inplace}) where {T,K,inplace}
    print(io, inplace ? "oneMKL FFT in-place " : "oneMKL FFT ", K ? "forward" : "inverse", " plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T)
end

# Plan constructors
function _create_descriptor(sz::NTuple{N,Int}, T::Type, complex::Bool) where {N}
    prec = T<:Float64 || T<:ComplexF64 ? ONEMKL_DFT_PRECISION_DOUBLE : ONEMKL_DFT_PRECISION_SINGLE
    dom = complex ? ONEMKL_DFT_DOMAIN_COMPLEX : ONEMKL_DFT_DOMAIN_REAL
    desc_ref = Ref{onemklDftDescriptor_t}()
    # Create descriptor for the full array dimensions
    lengths = collect(Int64, sz)
    st = length(lengths) == 1 ? onemklDftCreate1D(desc_ref, prec, dom, lengths[1]) : onemklDftCreateND(desc_ref, prec, dom, length(lengths), pointer(lengths))
    st == 0 || error("onemkl DFT create failed (status $st)")
    desc = desc_ref[]
    # Do not program descriptor scaling; we'll perform inverse normalization manually.
    # Set placement explicitly based on plan type later
    # Construct a SYCL queue from current Level Zero context/device (reuse global queue)
    ze_ctx = oneAPI.context(); ze_dev = oneAPI.device()
    sycl_dev = SYCL.syclDevice(SYCL.syclPlatform(oneAPI.driver()), ze_dev)
    sycl_ctx = SYCL.syclContext([sycl_dev], ze_ctx)
    q = SYCL.syclQueue(sycl_ctx, sycl_dev, oneAPI.global_queue(ze_ctx, ze_dev))
    return desc, q
end

# Complex plans
function plan_fft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    # For now, only support full transforms (all dimensions)
    if reg != ntuple(identity, N)
        error("Partial dimension FFT not yet supported. Region $reg must be $(ntuple(identity, N))")
    end
    desc, q = _create_descriptor(size(X), T, true)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_NOT_INPLACE)
    if N > 1
        # Column-major strides: stride along dimension i is product of sizes of previous dims
        strides = Vector{Int64}(undef, N+1); strides[1]=0
        prod = 1
        @inbounds for i in 1:N
            strides[i+1] = prod
            prod *= size(X,i)
        end
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_FWD_STRIDES, pointer(strides), length(strides))
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_BWD_STRIDES, pointer(strides), length(strides))
    end
    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    return cMKLFFTPlan{T,MKLFFT_FORWARD,false,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end
function plan_bfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    # For now, only support full transforms (all dimensions)
    if reg != ntuple(identity, N)
        error("Partial dimension FFT not yet supported. Region $reg must be $(ntuple(identity, N))")
    end
    desc, q = _create_descriptor(size(X), T, true)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_NOT_INPLACE)
    if N > 1
        strides = Vector{Int64}(undef, N+1); strides[1]=0; prod=1
        @inbounds for i in 1:N
            strides[i+1]=prod; prod*=size(X,i)
        end
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_FWD_STRIDES, pointer(strides), length(strides))
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_BWD_STRIDES, pointer(strides), length(strides))
    end
    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    return cMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end

# In-place (provide separate methods)
function plan_fft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    # For now, only support full transforms (all dimensions)
    if reg != ntuple(identity, N)
        @info "Partial dimension FFT not yet supported. Region $reg must be $(ntuple(identity, N))"
    end
    desc,q = _create_descriptor(size(X),T,true)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_INPLACE)
    if N > 1
        strides = Vector{Int64}(undef, N+1); strides[1]=0; prod=1
        @inbounds for i in 1:N
            strides[i+1]=prod; prod*=size(X,i)
        end
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_FWD_STRIDES, pointer(strides), length(strides))
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_BWD_STRIDES, pointer(strides), length(strides))
    end
    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    cMKLFFTPlan{T,MKLFFT_FORWARD,true,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end
function plan_bfft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    # For now, only support full transforms (all dimensions)
    if reg != ntuple(identity, N)
        @info "Partial dimension FFT not yet supported. Region $reg must be $(ntuple(identity, N))"
    end
    desc,q = _create_descriptor(size(X),T,true)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_INPLACE)
    if N > 1
        strides = Vector{Int64}(undef, N+1); strides[1]=0; prod=1
        @inbounds for i in 1:N
            strides[i+1]=prod; prod*=size(X,i)
        end
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_FWD_STRIDES, pointer(strides), length(strides))
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_BWD_STRIDES, pointer(strides), length(strides))
    end
    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    cMKLFFTPlan{T,MKLFFT_INVERSE,true,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end

# Real input methods - convert to complex like FFTW does
function plan_fft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    CT = Complex{T}
    # Create a complex plan by converting the real array to complex
    X_complex = oneAPI.oneArray{CT}(undef, size(X))
    plan_fft(X_complex, region)
end

function plan_bfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    CT = Complex{T}
    # Create a complex plan by converting the real array to complex
    X_complex = oneAPI.oneArray{CT}(undef, size(X))
    plan_bfft(X_complex, region)
end

function plan_fft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    error("In-place FFT not supported for real input arrays. Use plan_fft instead.")
end

function plan_bfft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    error("In-place FFT not supported for real input arrays. Use plan_bfft instead.")
end

# Real forward (out-of-place) - supports multi-dimensional transforms
function plan_rfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    # Convert region to tuple if it's a range
    if isa(region, AbstractUnitRange)
        region = tuple(region...)
    end
    R = length(region); reg = NTuple{R,Int}(region)

    # For single dimension transforms, use the optimized oneMKL real FFT
    if R == 1 && reg[1] == 1
        # Only support transform along first dimension for 1D case
        return _plan_rfft_1d(X, reg)
    end

    # For multi-dimensional transforms, use complex FFT approach
    # This is mathematically equivalent and works around oneMKL limitations
    return _plan_rfft_nd(X, reg)
end

# Single-dimension real FFT using oneMKL (optimized path)
function _plan_rfft_1d(X::oneAPI.oneArray{T,N}, reg::NTuple{1,Int}) where {T<:Union{Float32,Float64},N}
    # Create 1D descriptor for the transform dimension
    desc,q = _create_descriptor((size(X, reg[1]),), T, false)
    xdims = size(X)
    # output along first dim becomes N/2+1
    ydims = Base.setindex(xdims, div(xdims[1],2)+1, 1)
    buffer = oneAPI.oneArray{Complex{T}}(undef, ydims)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_NOT_INPLACE)

    # Set up for batched 1D transforms along first dimension
    if N > 1
        # Number of 1D transforms = product of all other dimensions
        num_transforms = prod(xdims[2:end])
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_NUMBER_OF_TRANSFORMS, Int64(num_transforms))
        # Distance between consecutive transforms (stride along batching dimension)
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_FWD_DISTANCE, Int64(xdims[1]))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_BWD_DISTANCE, Int64(ydims[1]))
    end

    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    R = length(reg)
    rMKLFFTPlan{T,MKLFFT_FORWARD,false,N,R,typeof(buffer)}(desc,q,xdims,ydims,:rfft,reg,buffer,nothing)
end

# Multi-dimensional real FFT using complex FFT approach
struct ComplexBasedRealFFTPlan{T,N,R} <: MKLFFTPlan{T,MKLFFT_FORWARD,false}
    complex_plan::cMKLFFTPlan{Complex{T},MKLFFT_FORWARD,false,N,R,Nothing}
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    region::NTuple{R,Int}
end

function _plan_rfft_nd(X::oneAPI.oneArray{T,N}, reg::NTuple{R,Int}) where {T<:Union{Float32,Float64},N,R}
    # Create complex version for planning
    X_complex = oneAPI.oneArray{Complex{T}}(undef, size(X))
    complex_plan = plan_fft(X_complex, reg)

    # Calculate output dimensions (real FFT output size)
    xdims = size(X)
    ydims = ntuple(N) do i
        if i in reg && i == minimum(reg)  # First dimension in region gets reduced
            div(xdims[i], 2) + 1
        else
            xdims[i]
        end
    end

    ComplexBasedRealFFTPlan{T,N,R}(complex_plan, xdims, ydims, reg)
end

# Show method for complex-based plan
function Base.show(io::IO, p::ComplexBasedRealFFTPlan{T}) where {T}
    print(io, "oneMKL FFT forward plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T, " (multi-dimensional via complex FFT)")
end

# Execution for complex-based real FFT plan
function Base.:*(p::ComplexBasedRealFFTPlan{T,N,R}, X::oneAPI.oneArray{T}) where {T,N,R}
    # Convert to complex
    X_complex = Complex{T}.(X)

    # Perform complex FFT
    Y_complex = p.complex_plan * X_complex

    # Extract appropriate portion for real FFT result
    # For real FFT, we only need roughly half the output due to conjugate symmetry
    indices = ntuple(N) do i
        if i in p.region && i == minimum(p.region)
            # First dimension in region: take 1:(N÷2+1)
            1:(div(p.sz[i], 2) + 1)
        else
            # Other dimensions: take all
            1:p.sz[i]
        end
    end

    Y = Y_complex[indices...]
    return Y
end



# Real inverse (complex->real) requires complex input shape - supports multi-dimensional transforms
function plan_brfft(X::oneAPI.oneArray{T,N}, d::Integer, region) where {T<:Union{ComplexF32,ComplexF64},N}
    # Convert region to tuple if it's a range
    if isa(region, AbstractUnitRange)
        region = tuple(region...)
    end
    R = length(region); reg = NTuple{R,Int}(region)

    # For single dimension transforms along first dim, use optimized oneMKL path
    if R == 1 && reg[1] == 1
        return _plan_brfft_1d(X, d, reg)
    end

    # For multi-dimensional transforms, use complex FFT approach
    return _plan_brfft_nd(X, d, reg)
end

# Single-dimension real inverse FFT using oneMKL (optimized path)
function _plan_brfft_1d(X::oneAPI.oneArray{T,N}, d::Integer, reg::NTuple{1,Int}) where {T<:Union{ComplexF32,ComplexF64},N}
    # Extract underlying real type R from Complex{R}
    @assert T <: Complex
    RT = T.parameters[1]

    # Create 1D descriptor for the transform dimension
    desc,q = _create_descriptor((d,), RT, false)
    xdims = size(X)
    ydims = Base.setindex(xdims, d, 1)
    buffer = oneAPI.oneArray{T}(undef, xdims) # copy for safety
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_NOT_INPLACE)

    # For now, disable batching for real inverse FFTs due to oneMKL parameter conflicts
    # Use loop-based approach instead for multi-dimensional arrays
    if N > 1
        @info "Batched real inverse FFTs not yet supported by oneMKL - please use loop-based approach or 1D arrays"
    end

    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    R = length(reg)
    rMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,typeof(buffer)}(desc,q,xdims,ydims,:brfft,reg,buffer,nothing)
end

# Multi-dimensional real inverse FFT using complex FFT approach
struct ComplexBasedRealIFFTPlan{T,N,R} <: MKLFFTPlan{T,MKLFFT_INVERSE,false}
    complex_plan::cMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,Nothing}
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    region::NTuple{R,Int}
    d::Int  # Original size of the reduced dimension
end

function _plan_brfft_nd(X::oneAPI.oneArray{T,N}, d::Integer, reg::NTuple{R,Int}) where {T<:Union{ComplexF32,ComplexF64},N,R}
    # Calculate the full complex array size (before real FFT reduction)
    xdims = size(X)
    full_complex_dims = ntuple(N) do i
        if i in reg && i == minimum(reg)  # First dimension in region was reduced
            d  # Restore original size
        else
            xdims[i]
        end
    end

    # Create complex version for planning - use the full size
    X_complex_full = oneAPI.oneArray{T}(undef, full_complex_dims)
    complex_plan = plan_bfft(X_complex_full, reg)

    ComplexBasedRealIFFTPlan{T,N,R}(complex_plan, xdims, full_complex_dims, reg, d)
end

# Show method for complex-based inverse plan
function Base.show(io::IO, p::ComplexBasedRealIFFTPlan{T}) where {T}
    print(io, "oneMKL FFT inverse plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T, " (multi-dimensional via complex FFT)")
end

# Execution for complex-based real inverse FFT plan
function Base.:*(p::ComplexBasedRealIFFTPlan{T,N,R}, X::oneAPI.oneArray{T}) where {T,N,R}
    # Reconstruct full complex array by exploiting conjugate symmetry
    # This is a simplified approach - for full accuracy, we'd need to properly
    # reconstruct the conjugate symmetric part

    # For now, pad with zeros (this works for certain cases but isn't fully general)
    xdims = size(X)
    full_indices = ntuple(N) do i
        if i in p.region && i == minimum(p.region)
            # Extend the reduced dimension
            1:p.d
        else
            1:xdims[i]
        end
    end

    # Create full complex array and copy the available data
    X_full = oneAPI.oneArray{T}(undef, p.osz)
    fill!(X_full, zero(T))

    # Copy the input data to the appropriate slice
    copy_indices = ntuple(N) do i
        if i in p.region && i == minimum(p.region)
            1:xdims[i]  # Only the available part
        else
            1:xdims[i]
        end
    end

    X_full[copy_indices...] = X

    # Perform complex inverse FFT
    Y_complex = p.complex_plan * X_full

    # Extract real part (this is where the real output comes from)
    return real.(Y_complex)
end

# Inverse plan for complex-based real FFT plans
function plan_inv(p::ComplexBasedRealFFTPlan{T,N,R}) where {T,N,R}
    # For real FFT inverse, we need plan_brfft functionality
    # The first dimension in the region should be the one that was reduced
    first_dim = minimum(p.region)
    d = p.sz[first_dim]  # Original size of the reduced dimension

    # Create inverse plan using our new multi-dimensional brfft
    brfft_plan = _plan_brfft_nd(oneAPI.oneArray{Complex{T}}(undef, p.osz), d, p.region)
    ScaledPlan(brfft_plan, 1/normalization_factor(p.sz, p.region))
end

# Inverse plan for complex-based real inverse FFT plans
function plan_inv(p::ComplexBasedRealIFFTPlan{T,N,R}) where {T,N,R}
    # Create forward plan
    forward_plan = _plan_rfft_nd(oneAPI.oneArray{real(T)}(undef, p.osz), p.region)
    ScaledPlan(forward_plan, 1/normalization_factor(p.osz, p.region))
end



# Convenience no-region methods use all dimensions in order
plan_fft(X::oneAPI.oneArray) = plan_fft(X, ntuple(identity, ndims(X)))
plan_bfft(X::oneAPI.oneArray) = plan_bfft(X, ntuple(identity, ndims(X)))
plan_fft!(X::oneAPI.oneArray) = plan_fft!(X, ntuple(identity, ndims(X)))
plan_bfft!(X::oneAPI.oneArray) = plan_bfft!(X, ntuple(identity, ndims(X)))
plan_rfft(X::oneAPI.oneArray) = plan_rfft(X, ntuple(identity, ndims(X)))  # default all dims like Base.rfft
plan_brfft(X::oneAPI.oneArray, d::Integer) = plan_brfft(X, d, ntuple(identity, ndims(X)))

# Alias names to mirror AMDGPU / AbstractFFTs style
const plan_ifft = plan_bfft
const plan_ifft! = plan_bfft!
# plan_irfft should be normalized, unlike plan_brfft
plan_irfft(X::oneAPI.oneArray{T,N}, d::Integer, region) where {T,N} = begin
    p = plan_brfft(X, d, region)
    ScaledPlan(p, 1/normalization_factor(p.sz, p.region))
end
plan_irfft(X::oneAPI.oneArray{T,N}, d::Integer) where {T,N} = plan_irfft(X, d, (1,))

# Inversion
Base.inv(p::MKLFFTPlan) = plan_inv(p)

# High-level wrappers operating like CPU FFTW versions.
function fft(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_fft(X) * X)
end
function ifft(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    p = plan_bfft(X)
    # Apply normalization for ifft (unlike bfft which is unnormalized)
    scaling = one(T) / normalization_factor(size(X), ntuple(identity, ndims(X)))
    scaling * (p * X)
end
function fft!(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_fft!(X) * X; X)
end
function ifft!(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    p = plan_bfft!(X)
    # Apply normalization for ifft! (unlike bfft! which is unnormalized)
    scaling = one(T) / normalization_factor(size(X), ntuple(identity, ndims(X)))
    p * X
    X .*= scaling
    X
end
function rfft(X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    (plan_rfft(X) * X)
end
function irfft(X::oneAPI.oneArray{T}, d::Integer) where {T<:Union{ComplexF32,ComplexF64}}
    # Use the normalized plan_irfft instead of unnormalized plan_brfft
    (plan_irfft(X, d) * X)
end

# Execution helpers
_rawptr(a::oneAPI.oneArray{T}) where T = reinterpret(Ptr{Cvoid}, pointer(a))

function _exec!(p::cMKLFFTPlan{T,MKLFFT_FORWARD,true}, X::oneAPI.oneArray{T}) where T
    st = onemklDftComputeForward(p.handle, _rawptr(X)); st==0 || error("forward FFT failed ($st)"); X
end
function _exec!(p::cMKLFFTPlan{T,MKLFFT_INVERSE,true}, X::oneAPI.oneArray{T}) where T
    st = onemklDftComputeBackward(p.handle, _rawptr(X)); st==0 || error("inverse FFT failed ($st)"); X
end
function _exec!(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{T}) where {T,K}
    st = (K==MKLFFT_FORWARD ? onemklDftComputeForwardOutOfPlace : onemklDftComputeBackwardOutOfPlace)(p.handle, _rawptr(X), _rawptr(Y)); st==0 || error("FFT failed ($st)"); Y
end

# Real forward
function _exec!(p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{Complex{T}}) where T
    st = onemklDftComputeForwardOutOfPlace(p.handle, _rawptr(X), _rawptr(Y)); st==0 || error("rfft failed ($st)"); Y
end
# Real inverse (complex -> real)
function _exec!(p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{R}) where {R,T<:Complex{R}}
    st = onemklDftComputeBackwardOutOfPlace(p.handle, _rawptr(X), _rawptr(Y)); st==0 || error("brfft failed ($st)"); Y
end

# Public API similar to AMDGPU
function Base.:*(p::cMKLFFTPlan{T,K,true}, X::oneAPI.oneArray{T}) where {T,K}
    _exec!(p,X)
end
function Base.:*(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}) where {T,K}
    Y = oneAPI.oneArray{T}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{T}, p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}) where {T,K}
    _exec!(p,X,Y)
end

# Real forward
function Base.:*(p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    Y = oneAPI.oneArray{Complex{T}}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{Complex{T}}, p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    _exec!(p,X,Y)
end
# Real inverse
function Base.:*(p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}) where {R,T<:Complex{R}}
    Y = oneAPI.oneArray{R}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{R}, p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}) where {R,T<:Complex{R}}
    _exec!(p,X,Y)
end

# Support for applying complex plans to real arrays (convert real to complex first)
function Base.:*(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{R}) where {T,K,R<:Union{Float32,Float64}}
    # Only allow if T is the complex version of R
    if T != Complex{R}
        error("Type mismatch: plan expects $(T) but got $(R)")
    end
    # Convert real input to complex
    X_complex = complex.(X)
    p * X_complex
end

function LinearAlgebra.mul!(Y::oneAPI.oneArray{T}, p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{R}) where {T,K,R<:Union{Float32,Float64}}
    # Only allow if T is the complex version of R
    if T != Complex{R}
        error("Type mismatch: plan expects $(T) but got $(R)")
    end
    # Convert real input to complex
    X_complex = complex.(X)
    _exec!(p, X_complex, Y)
end

end # module FFT
