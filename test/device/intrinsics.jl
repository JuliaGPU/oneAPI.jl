@testset "work items" begin
    @on_device get_work_dim()

    @on_device get_global_size(0)
    @on_device get_global_id(0)

    @on_device get_local_size(0)
    @on_device get_enqueued_local_size(0)
    @on_device get_local_id(0)

    @on_device get_num_groups(0)
    @on_device get_group_id(0)

    @on_device get_global_offset(0)

    @on_device get_global_linear_id()
    @on_device get_local_linear_id()
end



############################################################################################

@testset "math" begin
    buf = oneArray(zeros(Float32))

    function kernel(a, i)
        a[] = oneAPI.log10(i)
        return
    end

    @oneapi kernel(buf, Float32(100))
    val = Array(buf)
    @test val[] ≈ 2.0


    # dictionary of key=>tuple, where the tuple should
    # contain the cpu command and the cuda function to test.
    ops = Dict("exp"=>(exp, oneAPI.exp),
               "exp2"=>(exp2, oneAPI.exp2),
               "exp10"=>(exp10, oneAPI.exp10),
               "expm1"=>(expm1, oneAPI.expm1))

    @testset "$key" for key=keys(ops)
        cpu_op, cuda_op = ops[key]

        buf = oneArray(zeros(Float32))

        function cuda_kernel(a, x)
            a[] = cuda_op(x)
            return
        end

        #op(::Float32)
        x   = rand(Float32)
        @oneapi cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ cpu_op(x)
        @oneapi cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ cpu_op(-x)

        #op(::Float64)
        x   = rand(Float64)
        @oneapi cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ cpu_op(x)
        @oneapi cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ cpu_op(-x)
    end
end


############################################################################################

endline = Sys.iswindows() ? "\r\n" : "\n"

@testset "formatted output" begin
    _, out = @grab_output @on_device oneAPI.@printf("")
    @test out == ""

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
    function kernel(A)
        oneAPI.@printf("%f %f\n", A[1], A[1])
        return
    end
    x = oneArray(ones(2, 2))
    _, out = @grab_output begin
        @oneapi kernel(x)
        synchronize()
    end
    @test out == "1.000000 1.000000$endline"
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

    for typ in (Float32, Float64)
        test_output(typ(42), "42.000000")
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

@testset "@show" begin
    function kernel()
        seven_i32 = Int32(7)
        three_f64 = Float64(3)
        oneAPI.@show seven_i32
        oneAPI.@show three_f64
        oneAPI.@show 1f0 + 4f0
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
    @on_device @LocalMemory(Float32, 1)
    @on_device @LocalMemory(Float32, (1,2))
    @on_device @LocalMemory(Tuple{Float32, Float32}, 1)
    @on_device @LocalMemory(Tuple{Float32, Float32}, (1,2))
    @on_device @LocalMemory(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device @LocalMemory(Tuple{RGB{Float32}, UInt32}, (1,2))
end


@testset "static" begin

@testset "statically typed" begin
    function kernel(d, n)
        t = get_local_id(0)
        tr = n-t+1

        s = @LocalMemory(Float32, 1024)
        s2 = @LocalMemory(Float32, 1024)  # catch aliasing

        s[t] = d[t]
        s2[t] = 2*d[t]
        barrier()
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = oneArray(a)

    @oneapi items=n kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    @testset for typ in [Int32, Int64, Float32, Float64]
        function kernel(d::oneDeviceArray{T}, n) where {T}
            t = get_local_id(0)
            tr = n-t+1

            s = @LocalMemory(T, 1024)
            s2 = @LocalMemory(T, 1024)  # catch aliasing

            s[t] = d[t]
            s2[t] = d[t]
            barrier()
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
