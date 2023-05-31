using Random

@testset "rand" begin

# in-place
for (f,T) in ((rand!,Float16),
              (rand!,Float32),
              (randn!,Float16),
              (randn!,Float32)),
    d in (2, (2,2), (2,2,2), 3, (3,3), (3,3,3))
    A = oneArray{T}(undef, d)
    fill!(A, T(0))
    f(A)
    @test !iszero(collect(A))
end

# out-of-place, with implicit type
for (f,T) in ((oneAPI.rand,Float32), (oneAPI.randn,Float32)),
    args in ((2,), (2, 2), (3,), (3, 3))
    A = f(args...)
    @test eltype(A) == T
end

# out-of-place, with type specified
for (f,T) in ((oneAPI.rand,Float32), (oneAPI.randn,Float32),
              (rand,Float32), (randn,Float32)),
    args in ((T, 2), (T, 2, 2), (T, (2, 2)), (T, 3), (T, 3, 3), (T, (3, 3)))
    A = f(args...)
    @test eltype(A) == T
end

## seeding
oneAPI.seed!(1)
a = oneAPI.rand(Int32, 1)
oneAPI.seed!(1)
b = oneAPI.rand(Int32, 1)
@test iszero(collect(a) - collect(b))

end # testset
