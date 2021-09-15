using oneAPI.oneL0

# constructors
voidptr_a = ZePtr{Cvoid}(Int(0xDEADBEEF))
@test reinterpret(Ptr{Cvoid}, voidptr_a) == Ptr{Cvoid}(Int(0xDEADBEEF))

# getters
@test eltype(voidptr_a) == Cvoid

# comparisons
voidptr_b = ZePtr{Cvoid}(Int(0xCAFEBABE))
@test voidptr_a != voidptr_b


@testset "conversions" begin

# between host and device pointers
@test_throws ArgumentError convert(Ptr{Cvoid}, voidptr_a)

# between device pointers
intptr_a = ZePtr{Int}(Int(0xDEADBEEF))
@test convert(typeof(intptr_a), voidptr_a) == intptr_a

# convert back and forth from UInt
intptr_b = ZePtr{Int}(Int(0xDEADBEEF))
@test convert(UInt, intptr_b) == 0xDEADBEEF
@test convert(ZePtr{Int}, Int(0xDEADBEEF)) == intptr_b
@test Int(intptr_b) == Int(0xDEADBEEF)

# pointer arithmetic
intptr_c = ZePtr{Int}(Int(0xDEADBEEF))
intptr_d = 2 + intptr_c
@test isless(intptr_c, intptr_d)
@test intptr_d - intptr_c == 2
@test intptr_d - 2 == intptr_c
end


@testset "GPU or CPU integration" begin

a = [1]
ccall(:clock, Nothing, (Ptr{Int},), a)
@test_throws Exception ccall(:clock, Nothing, (ZePtr{Int},), a)
ccall(:clock, Nothing, (PtrOrZePtr{Int},), a)

b = oneArray{eltype(a), ndims(a)}(undef, size(a))
ccall(:clock, Nothing, (ZePtr{Int},), b)
@test_throws Exception ccall(:clock, Nothing, (Ptr{Int},), b)
ccall(:clock, Nothing, (PtrOrZePtr{Int},), b)

end


@testset "reference values" begin
    # Ref

    @test typeof(Base.cconvert(Ref{Int}, 1)) == typeof(Ref(1))
    @test Base.unsafe_convert(Ref{Int}, Base.cconvert(Ref{Int}, 1)) isa Ptr{Int}

    ptr = reinterpret(Ptr{Int}, C_NULL)
    @test Base.cconvert(Ref{Int}, ptr) == ptr
    @test Base.unsafe_convert(Ref{Int}, Base.cconvert(Ref{Int}, ptr)) == ptr

    arr = [1]
    @test Base.cconvert(Ref{Int}, arr) isa Base.RefArray{Int, typeof(arr)}
    @test Base.unsafe_convert(Ref{Int}, Base.cconvert(Ref{Int}, arr)) == pointer(arr)


    # ZeRef

    @test typeof(Base.cconvert(ZeRef{Int}, 1)) == typeof(ZeRef(1))
    @test Base.unsafe_convert(ZeRef{Int}, Base.cconvert(ZeRef{Int}, 1)) isa ZeRef{Int}

    zeptr = reinterpret(ZePtr{Int}, C_NULL)
    @test Base.cconvert(ZeRef{Int}, zeptr) == zeptr
    @test Base.unsafe_convert(ZeRef{Int}, Base.cconvert(ZeRef{Int}, zeptr)) == Base.bitcast(ZeRef{Int}, zeptr)

    zearr = oneAPI.oneArray([1])
    @test Base.cconvert(ZeRef{Int}, zearr) isa oneL0.ZeRefArray{Int, typeof(zearr)}
    @test Base.unsafe_convert(ZeRef{Int}, Base.cconvert(ZeRef{Int}, zearr)) == Base.bitcast(ZeRef{Int}, pointer(zearr))


    # RefOrZeRef

    @test typeof(Base.cconvert(RefOrZeRef{Int}, 1)) == typeof(Ref(1))
    @test Base.unsafe_convert(RefOrZeRef{Int}, Base.cconvert(RefOrZeRef{Int}, 1)) isa RefOrZeRef{Int}

    @test Base.cconvert(RefOrZeRef{Int}, ptr) == ptr
    @test Base.unsafe_convert(RefOrZeRef{Int}, Base.cconvert(RefOrZeRef{Int}, ptr)) == Base.bitcast(RefOrZeRef{Int}, ptr)

    @test Base.cconvert(RefOrZeRef{Int}, zeptr) == zeptr
    @test Base.unsafe_convert(RefOrZeRef{Int}, Base.cconvert(RefOrZeRef{Int}, zeptr)) == Base.bitcast(RefOrZeRef{Int}, zeptr)

    @test Base.cconvert(RefOrZeRef{Int}, arr) isa Base.RefArray{Int, typeof(arr)}
    @test Base.unsafe_convert(RefOrZeRef{Int}, Base.cconvert(RefOrZeRef{Int}, arr)) == Base.bitcast(RefOrZeRef{Int}, pointer(arr))

    @test Base.cconvert(RefOrZeRef{Int}, zearr) isa oneL0.ZeRefArray{Int, typeof(zearr)}
    @test Base.unsafe_convert(RefOrZeRef{Int}, Base.cconvert(RefOrZeRef{Int}, zearr)) == Base.bitcast(RefOrZeRef{Int}, pointer(zearr))
end
