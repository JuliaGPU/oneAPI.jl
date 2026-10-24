import KernelInterface as KI

# Arrays of isbits unions keep one selector byte per element past the data (see
# `typetagdata` in src/array.jl), like `Array` does.

const buffer_types = (oneL0.DeviceBuffer, oneL0.SharedBuffer, oneL0.HostBuffer)

# a struct with a union field is stored inline, with its selector inside the element
struct UnionField
    x::Union{Int,Float32}
end

# compare the elements' types too, which `isequal` ignores (e.g. `isequal(Int8(1), 1)`)
same(x, y) = size(x) == size(y) && all(map(===, x, y))

# increment the integer elements of `x`
function bump!(x)
    i = get_global_id()
    if i <= length(x)
        val = x[i]
        if val isa Int
            x[i] = val + 1
        end
    end
    return
end

@testset "size" begin
    a = oneArray(Union{Int,Nothing}[1, nothing, 3])
    @test sizeof(a) == 3 * sizeof(Int)  # excludes the selector bytes

    # members smaller than the union's alignment get padded
    T = Union{Int64,NTuple{3,Int32}}
    b = oneArray(T[1, (Int32(2), Int32(3), Int32(4))])
    @test sizeof(b) == 2 * 16
    @test pointer(b, 2) - pointer(b, 1) == 16
    @test pointer(view(b, reshape(2:2, 1, 1))) == pointer(b, 2)
end

@testset "round trips ($B)" for B in buffer_types
    for a in (Union{Int,Nothing}[1, nothing, 3],
              Union{Int8,Int64}[1, Int8(2), 3],
              Union{NTuple{3,Int8},Int16}[Int8.((1, 2, 3)), Int16(4), Int8.((5, 6, 7))],
              Union{Int64,NTuple{3,Int32}}[1, (Int32(2), Int32(3), Int32(4)), 5],
              Union{Nothing,Missing}[nothing, missing, missing, nothing],
              Union{Union{Missing,Nothing},Int}[0, missing, nothing])
        T = eltype(a)
        b = oneArray{T,1,B}(a)
        @test same(Array(b), a)
        @test same(Array(copy(b)), a)
        @test same(Array(similar(b) .= b), a)

        # between device arrays, also of other buffer types
        for B′ in buffer_types
            c = oneArray{T,1,B′}(undef, length(a))
            copyto!(c, b)
            @test same(Array(c), a)
            copyto!(c, 2, b, 1, 2)
            @test same(Array(c)[1:3], T[a[1], a[1], a[2]])
        end

        # from and to offsets in host arrays
        d = oneArray{T,1,B}(undef, 2)
        copyto!(d, 1, a, 2, 2)
        @test same(Array(d), a[2:3])
        e = similar(a)
        copyto!(e, 1, a, 1, 1)
        copyto!(e, 2, d, 1, 2)
        @test same(e[1:3], a[1:3])
    end
end

@testset "scalar indexing ($B)" for B in buffer_types
    a = oneArray{Union{Int,Nothing},1,B}([1, nothing, 3])
    @allowscalar begin
        @test a[1] === 1
        @test a[2] === nothing
        a[1] = nothing
        a[2] = 2
    end
    @test Array(a) == [nothing, 2, 3]
end

@testset "views" begin
    # the parent's first selector bytes differ from the view's
    a = Union{Int,Nothing}[nothing, nothing, 1, nothing, 3, 5]
    b = oneArray(a)
    v = view(b, 3:5)
    @test Array(v) == a[3:5]
    @test Array(copy(v)) == a[3:5]
    @test @allowscalar v[2] === nothing

    # kernels see the view's elements, and leave the parent alone outside of it
    @oneapi items=length(v) bump!(v)
    @test Array(v) == [2, nothing, 4]
    @test Array(b) == [nothing, nothing, 2, nothing, 4, 5]

    # views of views, and reshaped views
    w = view(v, 2:3)
    @test Array(w) == [nothing, 4]
    @oneapi items=length(w) bump!(w)
    @test Array(b) == [nothing, nothing, 2, nothing, 5, 5]
    r = reshape(view(b, 3:6), 2, 2)
    @test Array(r) == [2 5; nothing 5]
    @oneapi items=length(r) bump!(r)
    @test Array(b) == [nothing, nothing, 3, nothing, 6, 6]

    # copies into a view
    copyto!(view(b, 1:2), Union{Int,Nothing}[7, 8])
    @test Array(b) == [7, 8, 3, nothing, 6, 6]
    copyto!(view(b, 3:4), view(b, 1:2))
    @test Array(b) == [7, 8, 7, 8, 6, 6]
    view(b, 5:6) .= nothing
    @test Array(b) == [7, 8, 7, 8, nothing, nothing]

    # non-contiguous views index into their parent
    nc = view(b, 1:2:5)
    @oneapi items=length(nc) bump!(nc)
    @test Array(b) == [8, 8, 8, 8, nothing, nothing]

    # unions of singletons have no data, so their views don't have a byte offset either
    c = oneArray([nothing, nothing, missing, missing])
    @test isequal(Array(view(c, 3:4)), [missing, missing])
    @test isequal(Array(copy(view(c, 2:3))), [nothing, missing])
    function ismissing_kernel(ok, x)
        i = get_global_id()
        if i <= length(x)
            ok[i] = x[i] === missing
        end
        return
    end
    ok = oneArray(fill(false, 2))
    @oneapi items=2 ismissing_kernel(ok, view(c, 3:4))
    @test all(Array(ok))
end

@testset "kernels" begin
    a = oneArray(Union{Int,Nothing}[1, nothing, 3])
    @oneapi items=length(a) bump!(a)
    @test Array(a) == [2, nothing, 4]

    b = oneArray(Union{Union{Missing,Nothing},Int}[0, missing, nothing])
    @oneapi items=length(b) bump!(b)
    @test isequal(Array(b), [1, missing, nothing])

    c = oneArray(Union{Int64,NTuple{3,Int32}}[1, (Int32(2), Int32(3), Int32(4)), 5])
    @oneapi items=length(c) bump!(c)
    @test Array(c) == Union{Int64,NTuple{3,Int32}}[2, (Int32(2), Int32(3), Int32(4)), 6]

    function fill_struct!(x)
        i = get_global_id()
        if i <= length(x)
            x[i] = UnionField(isodd(i) ? Int(i) : Float32(i))
        end
        return
    end
    d = oneArray{UnionField}(undef, 4)
    @oneapi items=4 fill_struct!(d)
    @test Array(d) == [UnionField(1), UnionField(2f0), UnionField(3), UnionField(4f0)]
    @test Array(map(u -> u.x isa Int, d)) == [true, false, true, false]
end

@testset "local memory" begin
    function kernel_localmemory!(out)
        i = get_local_id()
        tmp = oneLocalArray(Union{Int64,NTuple{3,Int32}}, 4)
        tmp[i] = isodd(i) ? Int64(i) : (Int32(i), Int32(i), Int32(i))
        barrier(oneAPI.LOCAL_MEM_FENCE)
        val = tmp[5 - i]
        out[i] = val isa Int64 ? val : Int64(val[1]) + 100
        return
    end

    out = oneArray(zeros(Int64, 4))
    # The elements have size 12 but stride 16: reserve 64 data bytes and 4 selectors.
    # Checking the allocation catches undersizing even if the out-of-bounds accesses
    # happen to produce the expected result on a particular driver.
    ir = sprint() do io
        @device_code_llvm io=io dump_module=true @oneapi launch=false kernel_localmemory!(out)
    end
    local_sizes = [parse(Int, m[1]) for m in
                   eachmatch(r"addrspace\(3\) global \[(\d+) x i8\]", ir)]
    @test local_sizes == [68]
    @oneapi items=4 groups=1 kernel_localmemory!(out)
    @test Array(out) == [104, 3, 102, 1]
end

@testset "array operations" begin
    x = oneArray(Union{Missing,Bool}[missing, true, false])
    @test isequal(Array(x .| false), [missing, true, false])
    @test isequal(Array(map(!, x)), [missing, false, true])
    @test any(x) === true
    @test all(x) === false
    @test isequal(Array(reverse(x)), [false, true, missing])
    @test isequal(Array(vcat(x, x)), vcat(Array(x), Array(x)))

    y = similar(x)
    fill!(y, missing)
    @test all(ismissing, Array(y))
    fill!(y, true)
    @test Array(y) == [true, true, true]

    z = oneArray(Union{Int,Nothing}[1, nothing, 3, nothing])
    @test sum(something.(z, 0)) == 4
    @test mapreduce(v -> something(v, 10), +, z) == 24
    @test Array(reshape(z, 2, 2)) == [1 3; nothing nothing]

    w = similar(z)
    KI.copyto!(oneAPIBackend(), w, z)
    @test Array(w) == Array(z)
end

@testset "resize!" begin
    a = oneArray(Union{Int,Nothing}[1, nothing, 3])
    resize!(a, 5)
    @test length(a) == 5
    @test Array(a)[1:3] == [1, nothing, 3]
    resize!(a, 2)
    @test Array(a) == [1, nothing]

    # arrays derived from a view start at an offset
    b = vec(view(oneArray(Union{Int,Nothing}[nothing, 1, nothing, 3]), 2:4))
    resize!(b, 4)
    @test Array(b)[1:3] == [1, nothing, 3]

    c = oneArray([nothing, missing, missing])
    resize!(c, 4)
    @test isequal(Array(c)[1:3], [nothing, missing, missing])
    resize!(c, 1)
    @test Array(c) == [nothing]
    d = vec(view(oneArray([nothing, missing, missing]), 2:3))
    resize!(d, 1)
    @test isequal(Array(d), [missing])
end

@testset "unsupported operations" begin
    a = oneArray{Union{Int,Nothing},1,oneL0.SharedBuffer}([1, nothing])
    @test_throws ArgumentError unsafe_wrap(Array, a)
end
