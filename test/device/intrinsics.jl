@testset "work items" begin
    @on_device get_work_dim() |> sink

    @on_device get_global_size() |> sink
    @on_device get_global_id() |> sink

    @on_device get_local_size() |> sink
    @on_device get_enqueued_local_size() |> sink
    @on_device get_local_id() |> sink

    @on_device get_num_groups() |> sink
    @on_device get_group_id() |> sink

    @on_device get_global_offset() |> sink

    @on_device get_global_linear_id() |> sink
    @on_device get_local_linear_id() |> sink
end



############################################################################################

@testset "math" begin
    @testset "log10" begin
        @test testf(a->log10.(a), Float32[100])
    end

    for op in (exp, exp2, exp10, expm1)
        @testset "$op" begin
            typs = [Float32]
            float64_supported && push!(typs, Float64)
            for T in typs
                @test testf(x->op.(x), rand(T, 1))
                @test testf(x->op.(x), -rand(T, 1))
            end

        end
    end

    @testset "exp" begin
        @test testf(a->exp.(a), Matrix{ComplexF32}([1.0 + 1.0im 1.0 - 1.0im; -1.0 + 1.0im -1.0 - 1.0im]))

    end
end


############################################################################################

endline = Sys.iswindows() ? "\r\n" : "\n"

@testset "formatted output" begin
    # BROKEN: cintel/compute-runtime#635
    #_, out = @grab_output @on_device oneAPI.@printf("")
    #@test out == ""

    _, out = @grab_output @on_device oneAPI.@printf("Testing...\n")
    @test out == "Testing...$endline"

    # narrow integer
    _, out = @grab_output @on_device oneAPI.@printf("Testing %d %d...\n", Int32(1), Int32(2))
    @test out == "Testing 1 2...$endline"

    # wide integer
    _, out = @grab_output if Sys.iswindows()
        @on_device oneAPI.@printf("Testing %lld %lld...\n", Int64(1), Int64(2))
    else
        @on_device oneAPI.@printf("Testing %ld %ld...\n", Int64(1), Int64(2))
    end
    @test out == "Testing 1 2...$endline"

    _, out = @grab_output @on_device begin
        oneAPI.@printf("foo")
        oneAPI.@printf("bar\n")
    end
    @test out == "foobar$endline"

    # c argument promotions
    if float64_supported
        function kernel(A)
            oneAPI.@printf("%f %f\n", A[1], A[1])
            return
        end
        x = oneArray(ones(Float64, 2, 2))
        _, out = @grab_output begin
            @oneapi kernel(x)
            synchronize()
        end
        @test out == "1.000000 1.000000$endline"
    end
end

@testset "@print" begin
    # basic @print/@println

    _, out = @grab_output @on_device oneAPI.@print("Hello, World\n")
    @test out == "Hello, World$endline"

    _, out = @grab_output @on_device oneAPI.@println("Hello, World")
    @test out == "Hello, World$endline"


    # argument interpolation (by the macro, so can use literals)

    _, out = @grab_output @on_device oneAPI.@print("foobar")
    @test out == "foobar"

    _, out = @grab_output @on_device oneAPI.@print(:foobar)
    @test out == "foobar"

    _, out = @grab_output @on_device oneAPI.@print("foo", "bar")
    @test out == "foobar"

    _, out = @grab_output @on_device oneAPI.@print("foobar ", 42)
    @test out == "foobar 42"

    _, out = @grab_output @on_device oneAPI.@print("foobar $(42)")
    @test out == "foobar 42"

    _, out = @grab_output @on_device oneAPI.@print("foobar $(4)", 2)
    @test out == "foobar 42"

    _, out = @grab_output @on_device oneAPI.@print("foobar ", 4, "$(2)")
    @test out == "foobar 42"

    _, out = @grab_output @on_device oneAPI.@print(42)
    @test out == "42"

    _, out = @grab_output @on_device oneAPI.@print(4, 2)
    @test out == "42"

    # bug: @println failed to invokce @print with endline in the case of interpolation
    _, out = @grab_output @on_device oneAPI.@println("foobar $(42)")
    @test out == "foobar 42$endline"


    # argument types

    # we're testing the generated functions now, so can't use literals
    function test_output(val, str)
        canary = rand(Int32) # if we mess up the main arg, this one will print wrong
        _, out = @grab_output @on_device oneAPI.@print(val, " (", canary, ")")
        @test out == "$(str) ($(Int(canary)))"
    end

    for typ in (Int16, Int32, Int64, UInt16, UInt32, UInt64)
        test_output(typ(42), "42")
    end

    if float64_supported
        for typ in (Float32, Float64)
            test_output(typ(42), "42.000000")
        end
    end

    test_output(Cchar('c'), "c")

    for typ in (Ptr{Cvoid}, Ptr{Int})
        ptr = convert(typ, Int(0x12345))
        test_output(ptr, Sys.iswindows() ? "0000000000012345" : "0x12345")
    end

    test_output(true, "1")
    test_output(false, "0")


    # escaping

    kernel1(val) = (oneAPI.@print(val); nothing)
    _, out = @grab_output @on_device kernel1(42)
    @test out == "42"

    kernel2(val) = (oneAPI.@println(val); nothing)
    _, out = @grab_output @on_device kernel2(42)
    @test out == "42$endline"
end

float64_supported && @testset "@show" begin
    function kernel()
        seven_i32 = Int32(7)
        three_f64 = Float64(3)
        oneAPI.@show seven_i32
        oneAPI.@show three_f64 1f0 + 4f0
        return nothing
    end

    _, out = @grab_output @on_device kernel()
    @test out == "seven_i32 = 7$(endline)three_f64 = 3.000000$(endline)1.0f0 + 4.0f0 = 5.000000$(endline)"
end



############################################################################################

# a composite type to test for more complex element types
@eval struct RGB{T}
    r::T
    g::T
    b::T
end

@testset "local memory" begin

n = 256

@testset "constructors" begin
    # static
    @on_device oneLocalArray(Float32, 1)
    @on_device oneLocalArray(Float32, (1,2))
    @on_device oneLocalArray(Tuple{Float32, Float32}, 1)
    @on_device oneLocalArray(Tuple{Float32, Float32}, (1,2))
    @on_device oneLocalArray(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device oneLocalArray(Tuple{RGB{Float32}, UInt32}, (1,2))
end


@testset "static" begin

@testset "statically typed" begin
    function kernel(d, n)
        t = get_local_id()
        tr = n-t+1

        s = oneLocalArray(Float32, 1024)
        s2 = oneLocalArray(Float32, 1024)  # catch aliasing

        s[t] = d[t]
        s2[t] = 2*d[t]
        barrier(0)
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = oneArray(a)

    @oneapi items=n kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    typs = [Int32, Int64, Float32]
    float64_supported && push!(typs, Float64)
    @testset for typ in typs
        function kernel(d::oneDeviceArray{T}, n) where {T}
            t = get_local_id()
            tr = n-t+1

            s = oneLocalArray(T, 1024)
            s2 = oneLocalArray(T, 1024)  # catch aliasing

            s[t] = d[t]
            s2[t] = d[t]
            barrier(0)
            d[t] = s[tr]

            return
        end

        a = rand(typ, n)
        d_a = oneArray(a)

        @oneapi items=n kernel(d_a, n)
        @test reverse(a) == Array(d_a)
    end
end

end

end



############################################################################################

@testset "atomics (low level)" begin

@testset "atomic_add($T)" for T in [Int32, UInt32]
    a = oneArray([zero(T)])

    function kernel(a, b)
        oneAPI.atomic_add!(pointer(a), b)
        return
    end

    @oneapi items=256 kernel(a, one(T))
    @test Array(a)[1] == T(256)
end

@testset "atomic_sub($T)" for T in [Int32, UInt32]
    a = oneArray([T(256)])

    function kernel(a, b)
        oneAPI.atomic_sub!(pointer(a), b)
        return
    end

    @oneapi items=256 kernel(a, one(T))
    @test Array(a)[1] == T(0)
end

@testset "atomic_inc($T)" for T in [Int32, UInt32]
    a = oneArray([zero(T)])

    function kernel(a)
        oneAPI.atomic_inc!(pointer(a))
        return
    end

    @oneapi items=256 kernel(a)
    @test Array(a)[1] == T(256)
end

@testset "atomic_dec($T)" for T in [Int32, UInt32]
    a = oneArray([T(256)])

    function kernel(a)
        oneAPI.atomic_dec!(pointer(a))
        return
    end

    @oneapi items=256 kernel(a)
    @test Array(a)[1] == T(0)
end

@testset "atomic_min($T)" for T in [Int32, UInt32]
    a = oneArray([T(256)])

    function kernel(a, T)
        i = get_global_id()
        oneAPI.atomic_min!(pointer(a), i%T)
        return
    end

    @oneapi items=256 kernel(a, T)
    @test Array(a)[1] == one(T)
end

@testset "atomic_max($T)" for T in [Int32, UInt32]
    a = oneArray([zero(T)])

    function kernel(a, T)
        i = get_global_id()
        oneAPI.atomic_max!(pointer(a), i%T)
        return
    end

    @oneapi items=256 kernel(a, T)
    @test Array(a)[1] == T(256)
end

@testset "atomic_and($T)" for T in [Int32, UInt32]
    a = oneArray([T(1023)])

    function kernel(a, T)
        i = get_global_id() - 1
        k = 1
        for i = 1:i
            k *= 2
        end
        b = 1023 - k  # 1023 - 2^i
        oneAPI.atomic_and!(pointer(a), T(b))
        return
    end

    @oneapi items=10 kernel(a, T)
    @test Array(a)[1] == zero(T)
end

@testset "atomic_or($T)" for T in [Int32, UInt32]
    a = oneArray([zero(T)])

    function kernel(a, T)
        i = get_global_id()
        b = 1  # 2^(i-1)
        for i = 1:i
            b *= 2
        end
        b ÷= 2
        oneAPI.atomic_or!(pointer(a), T(b))
        return
    end

    @oneapi items=10 kernel(a, T)
    @test Array(a)[1] == T(1023)
end

@testset "atomic_xor($T)" for T in [Int32, UInt32]
    a = oneArray([T(1023)])

    function kernel(a, T)
        i = get_global_id()
        b = 1  # 2^(i-1)
        for i = 1:i
            b *= 2
        end
        b ÷= 2
        oneAPI.atomic_xor!(pointer(a), T(b))
        return
    end

    @oneapi items=10 kernel(a, T)
    @test Array(a)[1] == zero(T)
end

@testset "atomic_xchg($T)" for T in [Int32, UInt32, Float32]
    a = oneArray([zero(T)])

    function kernel(a, b)
        oneAPI.atomic_xchg!(pointer(a), b)
        return
    end

    @oneapi items=256 kernel(a, one(T))
    @test Array(a)[1] == one(T)
end

end



############################################################################################



@testset "atomics (high-level)" begin

@testset "add" begin
    @testset for T in [Int32, UInt32, Float32]
        a = oneArray([zero(T)])

        function kernel(T, a)
            oneAPI.@atomic a[1] = a[1] + 1
            oneAPI.@atomic a[1] += 1
            return
        end

        @oneapi items=256 kernel(T, a)
        @test Array(a)[1] == 512
    end
end

@testset "sub" begin
    @testset for T in [Int32, UInt32, Float32]
        a = oneArray(T[1024])

        function kernel(T, a)
            oneAPI.@atomic a[1] = a[1] - 1
            oneAPI.@atomic a[1] -= 1
            return
        end

        @oneapi items=256 kernel(T, a)
        @test Array(a)[1] == 512
    end
end

@testset "and" begin
    @testset for T in [Int32, UInt32]
        a = oneArray([~zero(T), ~zero(T)])

        function kernel(T, a)
            i = get_local_id()
            mask = ~(T(1) << (i-1))
            oneAPI.@atomic a[1] = a[1] & mask
            oneAPI.@atomic a[2] &= mask
            return
        end

        @oneapi items=8*sizeof(T) kernel(T, a)
        @test Array(a)[1] == zero(T)
        @test Array(a)[2] == zero(T)
    end
end

@testset "or" begin
    @testset for T in [Int32, UInt32]
        a = oneArray([zero(T), zero(T)])

        function kernel(T, a)
            i = get_local_id()
            mask = T(1) << (i-1)
            oneAPI.@atomic a[1] = a[1] | mask
            oneAPI.@atomic a[2] |= mask
            return
        end

        @oneapi items=8*sizeof(T) kernel(T, a)
        @test Array(a)[1] == ~zero(T)
        @test Array(a)[2] == ~zero(T)
    end
end

@testset "xor" begin
    @testset for T in [Int32, UInt32]
        a = oneArray([zero(T), zero(T)])

        function kernel(T, a)
            i = get_local_id()
            mask = T(1) << ((i-1)%(8*sizeof(T)))
            oneAPI.@atomic a[1] = a[1] ⊻ mask
            oneAPI.@atomic a[2] ⊻= mask
            return
        end

        nb = 4
        @oneapi items=(8*sizeof(T)+nb) kernel(T, a)
        @test Array(a)[1] == ~zero(T) & ~((one(T) << nb) - one(T))
        @test Array(a)[2] == ~zero(T) & ~((one(T) << nb) - one(T))
    end
end

@testset "max" begin
    @testset for T in [Int32, UInt32, Float32]
        a = oneArray([zero(T)])

        function kernel(T, a)
            i = get_local_id()
            oneAPI.@atomic a[1] = max(a[1], i)
            return
        end

        @oneapi items=32 kernel(T, a)
        @test Array(a)[1] == 32
    end
end

@testset "min" begin
    @testset for T in [Int32, UInt32, Float32]
        a = oneArray([typemax(T)])

        function kernel(T, a)
            i = get_local_id()
            oneAPI.@atomic a[1] = min(a[1], i)
            return
        end

        @oneapi items=32 kernel(T, a)
        @test Array(a)[1] == 1
    end
end

@testset "mul" begin
    @testset for T in [Int32, UInt32, Float32]
        a = oneArray(T[1])

        function kernel(T, a)
            oneAPI.@atomic a[1] = a[1] * 2
            oneAPI.@atomic a[1] *= 2
            return
        end

        @oneapi items=8 kernel(T, a)
        @test Array(a)[1] == 65536
    end
end

@testset "div" begin
    @testset for T in [Int32, UInt32, Float32]
        a = oneArray(T[65536])

        function kernel(T, a)
            oneAPI.@atomic a[1] = a[1] ÷ 2
            oneAPI.@atomic a[1] ÷= 2
            return
        end

        @oneapi items=8 kernel(T, a)
        @test Array(a)[1] == 1
    end
end

@testset "macro" begin
    using oneAPI: AtomicError

    @test_throws AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        oneAPI.@atomic a[1] = 1
    end
    @test_throws AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        oneAPI.@atomic a[1] = b ? 1 : 2
    end

    @test_throws AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side") @macroexpand begin
        oneAPI.@atomic a[1] = a[2] + 1
    end

    @test_throws AtomicError("unknown @atomic expression") @macroexpand begin
        oneAPI.@atomic wat(a[1])
    end

    @test_throws AtomicError("@atomic should be applied to an array reference expression") @macroexpand begin
        oneAPI.@atomic a = a + 1
    end
end

end
