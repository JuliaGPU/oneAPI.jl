using Test
using oneAPI
using oneAPI.oneMKL.FFT
using AbstractFFTs
using FFTW
using Random
Random.seed!(1234)

# Helper to move data to GPU
gpu(A::AbstractArray{T}) where T = oneAPI.oneArray{T}(A)
struct _Plan end
struct _FFT end

const MYRTOL = 1e-5
const MYATOL = 1e-8

function cmp(a,b; rtol=MYRTOL, atol=MYATOL)
    @test isapprox(Array(a), Array(b); rtol=rtol, atol=atol)
end

function test_plan(::_Plan, plan, X::AbstractArray{T,N}) where {T,N}
    p = plan(X)
    Y = p * X
    return Y
end

function test_plan(::_FFT, f, X::AbstractArray{T,N}) where {T,N}
    Y = if f === AbstractFFTs.irfft || f === AbstractFFTs.brfft
        f(X, size(X, ndims(X))*2 - 2)
    else
        f(X)
    end
    return Y
end

function test_plan(t, plan::Function, dim::Tuple, T::Type, iplan=nothing)
    X = rand(T, dim)
    dX = gpu(X)
    Y = test_plan(t, plan, X)
    dY = test_plan(t, plan, dX)
    cmp(dY, Y)
    if iplan !== nothing
        iX = test_plan(t, iplan, Y)
        idX = test_plan(t, iplan, dY)
        cmp(idX, iX)
    end
end

@testset "FFT" begin
@testset "$(length(dim))D" for dim in [(8,), (8,32), (8,32,64)]
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, ComplexF32, AbstractFFTs.plan_ifft)
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, ComplexF32, AbstractFFTs.plan_bfft)
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, Float32, AbstractFFTs.plan_ifft)
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, Float32, AbstractFFTs.plan_bfft)
    test_plan(_Plan(), AbstractFFTs.plan_rfft, dim, Float32)
    test_plan(_Plan(), AbstractFFTs.plan_fft!, dim, ComplexF32, AbstractFFTs.plan_bfft!)
    # Not part of FFTW
    # test_plan(AbstractFFTs.plan_rfft!, Float32)
    test_plan(_FFT(), AbstractFFTs.fft, dim, ComplexF32, AbstractFFTs.ifft)
    test_plan(_FFT(), AbstractFFTs.fft, dim, ComplexF32, AbstractFFTs.bfft)
    test_plan(_FFT(), AbstractFFTs.rfft, dim, Float32, AbstractFFTs.irfft)
    test_plan(_FFT(), AbstractFFTs.rfft, dim, Float32, AbstractFFTs.brfft)

    # # Test real inverse FFTs (irfft/brfft) if 1D
    # if length(dim) == 1
    #     # Test irfft - create appropriate complex input
    #     X_real = rand(Float32, dim)
    #     X_complex = AbstractFFTs.rfft(X_real)
    #     dX_complex = gpu(X_complex)
    #     Y_real = AbstractFFTs.irfft(Array(dX_complex), dim[1])
    #     dY_real = AbstractFFTs.irfft(dX_complex, dim[1])
    #     cmp(dY_real, Y_real)
    # end

    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, ComplexF64, AbstractFFTs.plan_ifft)
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, ComplexF64, AbstractFFTs.plan_bfft)
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, Float64, AbstractFFTs.plan_ifft)
    test_plan(_Plan(), AbstractFFTs.plan_fft, dim, Float64, AbstractFFTs.plan_bfft)
    test_plan(_Plan(), AbstractFFTs.plan_rfft, dim, Float64)
    test_plan(_Plan(), AbstractFFTs.plan_fft!, dim, ComplexF64, AbstractFFTs.plan_bfft!)
    # Not part of FFTW
    # test_plan(AbstractFFTs.plan_rfft!, Float64)
    test_plan(_FFT(), AbstractFFTs.fft, dim, ComplexF64, AbstractFFTs.ifft)
    test_plan(_FFT(), AbstractFFTs.fft, dim, ComplexF64, AbstractFFTs.bfft)
    test_plan(_FFT(), AbstractFFTs.rfft, dim, Float64, AbstractFFTs.irfft)
    test_plan(_FFT(), AbstractFFTs.rfft, dim, Float64, AbstractFFTs.brfft)
end
end
