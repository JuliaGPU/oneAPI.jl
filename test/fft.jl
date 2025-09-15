using Test
using oneAPI
using oneAPI.oneMKL.FFT
using AbstractFFTs
using FFTW

# Helper to move data to GPU
gpu(A::AbstractArray{T}) where T = oneAPI.oneArray{T}(A)
struct _Plan end
struct _FFT end

const MYRTOL = 1e-5
const MYATOL = 1e-8

function cmp(a,b; rtol=MYRTOL, atol=MYATOL)
    @show typeof(a), typeof(b)
    @test isapprox(Array(a), Array(b); rtol=rtol, atol=atol)
end

Ns = (8,32,64,8)

@testset "FFT" begin
function test_plan(::_Plan, plan, X::AbstractArray{T,N}) where {T,N}
    p = plan(X)
    println(p)
    Y = p * X
    return Y
end

function test_plan(::_FFT, f, X::AbstractArray{T,N}) where {T,N}
    Y = f(X)
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

# 1D
dim = (8,32,64)
for dim in [(8,), (8,32), (8,32,64)]
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
end
end
