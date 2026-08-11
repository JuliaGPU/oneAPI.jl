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
    if length(dim) == 1  # irfft/brfft only for 1D
        test_plan(_FFT(), AbstractFFTs.rfft, dim, Float32, AbstractFFTs.irfft)
        test_plan(_FFT(), AbstractFFTs.rfft, dim, Float32, AbstractFFTs.brfft)
    end
    if (ComplexF64 in eltypes) && (Float64 in eltypes)
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
        if length(dim) == 1  # irfft/brfft only for 1D
            test_plan(_FFT(), AbstractFFTs.rfft, dim, Float64, AbstractFFTs.irfft)
            test_plan(_FFT(), AbstractFFTs.rfft, dim, Float64, AbstractFFTs.brfft)
        end
    end
end

    @testset "partial regions" begin
        for (dim, regions) in [
                ((8, 32), [(1,), (2,), 1:2]),
                ((9, 6), [(1,), (2,)]),
                ((8, 32, 64), [(1,), (2,), (3,), (1, 2), (2, 3)]),
            ]
            @testset "$(length(dim))D region=$region" for region in regions
                regdims = collect(region)
                batchlen = prod(dim[regdims])

                # complex transforms
                X = rand(ComplexF32, dim)
                dX = gpu(X)
                cmp(AbstractFFTs.fft(dX, region), FFTW.fft(X, region))
                p = AbstractFFTs.plan_fft(dX, region)
                Y = FFTW.fft(X, region)
                dY = p * dX
                cmp(dY, Y)
                cmp(AbstractFFTs.plan_ifft(dX, region) * dY, X)
                cmp(AbstractFFTs.plan_bfft(dX, region) * dY, X .* batchlen)

                # in-place complex transforms
                dXc = copy(dX)
                AbstractFFTs.plan_fft!(dXc, region) * dXc
                cmp(dXc, Y)
                AbstractFFTs.plan_bfft!(dXc, region) * dXc
                cmp(dXc, X .* batchlen)

                # real forward and inverse transforms
                Xr = rand(Float32, dim)
                dXr = gpu(Xr)
                Yr = FFTW.rfft(Xr, region)
                cmp(AbstractFFTs.rfft(dXr, region), Yr)
                d1 = dim[first(regdims)]
                cmp(AbstractFFTs.irfft(gpu(Yr), d1, region), FFTW.irfft(Yr, d1, region))
            end
        end

        # multidimensional irfft over all dimensions (uses the conjugate-symmetric
        # reconstruction path rather than the 1D real descriptor)
        @testset "full-region ND irfft $(dim)" for dim in [(8, 32), (9, 6), (8, 32, 64)]
            Xr = rand(Float32, dim)
            Yr = FFTW.rfft(Xr)
            cmp(AbstractFFTs.irfft(gpu(Yr), dim[1]), FFTW.irfft(Yr, dim[1]))
            cmp(AbstractFFTs.brfft(gpu(Yr), dim[1]), FFTW.brfft(Yr, dim[1]))
        end

        # non-contiguous regions cannot be expressed as a single oneMKL descriptor
        @test_throws ErrorException AbstractFFTs.plan_fft(gpu(rand(ComplexF32, 4, 4, 4)), (1, 3))
    end

@testset "shared queue lifetime across plans" begin
    # Plans must share the single cached task-local SYCL queue rather than each owning a
    # throwaway one (whose finalizer would tear down shared SYCL/oneMKL state). Assert the
    # shared handle deterministically, independent of whether a stale queue would crash.
    cached_handle = Base.unsafe_convert(oneAPI.oneMKL.syclQueue_t,
        oneAPI.sycl_queue(oneAPI.global_queue(oneAPI.context(), oneAPI.device())))

    dX1 = gpu(rand(ComplexF32, 8))
    p1 = AbstractFFTs.plan_fft(dX1)
    @test p1.queue == cached_handle
    dY1 = p1 * dX1
    p1i = AbstractFFTs.plan_ifft(dX1)
    p1i * dY1

    GC.gc(true)  # run finalizers of any throwaway per-plan SYCL wrappers

    X2 = rand(ComplexF32, 8, 32)
    dX2 = gpu(X2)
    p2 = AbstractFFTs.plan_fft(dX2)
    @test p2.queue == cached_handle
    cmp(p2 * dX2, fft(X2))
end

@testset "stream interleave" begin
    # FFT plans capture their SYCL queue at construction and execute through _exec!,
    # which must apply the Julia → MKL ordering boundary itself. Broadcast → fft →
    # broadcast with no intermediate synchronization.
    X = gpu(rand(ComplexF32, 256))
    hX = Array(X)
    X .= X .* 2f0
    Y = fft(X)
    Z = abs.(Y)
    cmp(Z, abs.(fft(hX .* 2f0)))
end
end
