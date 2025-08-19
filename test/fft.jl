using Test
using oneAPI
using oneAPI.oneMKL.FFT
using AbstractFFTs
using FFTW

# Helper to move data to GPU
gpu(A::AbstractArray{T}) where T = oneAPI.oneArray{T}(A)

const MYRTOL = 1e-5
const MYATOL = 1e-8

function cmp(a,b; rtol=MYRTOL, atol=MYATOL)
    @test isapprox(Array(a), Array(b); rtol=rtol, atol=atol)
end

function cmp_broken(a,b; rtol=MYRTOL, atol=MYATOL)
    @test_broken isapprox(Array(a), Array(b); rtol=rtol, atol=atol)
end

@testset "FFT" begin
    Ns = (8,32,64,8)

    # Complex tests
    for T in intersect(eltypes, [ComplexF32, ComplexF64])
        @testset "complex $T" begin
            # 1D out-of-place
            X = rand(T, Ns[1])
            dX = gpu(X)
            p = plan_fft(dX)
            dY = p * dX
            cmp(dY, fft(X))
            @test X == Array(dX)

            pinv = plan_ifft(dY)
            dZ = pinv * dY
            cmp(dZ, X)

            # in-place
            X2 = rand(T, Ns[1])
            dX2 = gpu(X2)
            p2 = plan_fft!(dX2)
            p2 * dX2
            cmp(dX2, fft(X2))
            pinv2 = plan_ifft!(dX2)
            pinv2 * dX2
            cmp(dX2, X2)

            # 2D
            X = rand(T, Ns[1], Ns[2])
            dX = gpu(X)
            p = plan_fft(dX)
            dY = p * dX
            cmp(dY, fft(X))
            pinv = plan_ifft(dY)
            dZ = pinv * dY
            cmp(dZ, X)

            # region/batched (1D along dim 1)
            # Not yet supported
            X = rand(T, Ns[1], Ns[2])
            dX = gpu(X)
            p = plan_fft!(dX, 1)
            p * dX
            cmp_broken(dX, fft(X,1))
            pinv = plan_ifft!(dX,1)
            pinv * dX
            cmp_broken(dX, X)
        end
    end

    # Real tests
    for T in intersect(eltypes, [Float32, Float64])
        @testset "real $T" begin
            X = rand(T, Ns[1])
            dX = gpu(X)
            p = plan_rfft(dX)
            dY = p * dX
            cmp(dY, rfft(X))
            pinv = plan_irfft(dY, size(X,1))
            dZ = pinv * dY
            cmp(dZ, X)

            # 2D real rfft along first dim default
            X = rand(T, Ns[1], Ns[2])
            dX = gpu(X)
            p = plan_rfft(dX)
            dY = p * dX
            cmp(dY, rfft(X, (1,)))  # Compare with 1D FFT along first dim, not multi-dimensional FFT
            # Something's wrong in oneAPI
            pinv = plan_irfft(dY, size(X,1))
            dZ = pinv * dY
            cmp_broken(dZ, X)
        end
    end

    # Wrapper convenience
    for T in intersect(eltypes, [ComplexF32, ComplexF64])
        X = gpu(rand(T, Ns[1], Ns[2]))
        Y = fft(X)
        cmp(Y, fft(Array(X)))
        Z = ifft(Y)
        cmp(Z, Array(X))
    end

    for T in intersect(eltypes, [Float32, Float64])
        X = gpu(rand(T, Ns[1], Ns[2]))
        Y = rfft(X)
        cmp(Y, rfft(Array(X), (1,)))  # Compare with 1D FFT along first dim, not multi-dimensional FFT
        # Doesn't work
        Z = irfft(Y, size(X,1))
        cmp_broken(Z, Array(X))
    end
end
