import Adapt
import LinearAlgebra

using StaticArrays

dummy() = return

function store_checked(a, i)
    a[i] = 1
    return
end

@testset "@oneapi" begin

@test_throws UndefVarError @oneapi undefined()
@test_throws MethodError @oneapi dummy(1)


@testset "low-level interface" begin
    k = zefunction(dummy)
    k()
    k(; items=1)
end


@testset "launch configuration" begin
    @oneapi dummy()

    items = 1
    @oneapi items dummy()
    @oneapi items=1 dummy()
    @oneapi items=(1,1) dummy()
    @oneapi items=(1,1,1) dummy()

    groups = 1
    @oneapi groups dummy()
    @oneapi groups=1 dummy()
    @oneapi groups=(1,1) dummy()
    @oneapi groups=(1,1,1) dummy()
end


@testset "launch=false" begin
    k = @oneapi launch=false dummy()
    k()
    k(; items=1)
end


function store_max_sub_group_size(a)
    @inbounds a[1] = get_max_sub_group_size()
    return
end

@testset "sub-group size" begin
    a = oneArray{UInt32}(undef, 1)
    sizes = oneL0.compute_properties(device()).subGroupSizes
    for sub_group_size in sizes
        @oneapi items=64 sub_group_size store_max_sub_group_size(a)
        @test Array(a)[1] == sub_group_size
    end
    @test_throws ArgumentError @oneapi sub_group_size=maximum(sizes)+1 dummy()
end


@testset "debug level" begin
    T = Tuple{oneDeviceVector{Int, oneAPI.AS.CrossWorkgroup}, Int}
    ir(level) = sprint(io -> oneAPI.code_llvm(io, store_checked, T; kernel = true,
                                              debug_level = level))
    @test !occursin("report_exception", ir(0))
    @test occursin("report_exception", ir(1)) && !occursin("report_exception_frame", ir(1))
    @test occursin("report_exception_frame", ir(2))
end


@testset "inference" begin
    foo() = @oneapi dummy()
    @inferred foo()

    # with arguments, we call kernel_convert
    kernel(a) = return
    bar(a) = @oneapi kernel(a)
    @inferred bar(oneArray([1]))
end


function store_kernel(a, x)
    @inbounds a[1] = x
    return
end
function store_twice_kernel(a, x)
    @inbounds a[1] = x + x
    return
end

@testset "launch specialization" begin
    # the launch is specialized on the argument types, so it doesn't dispatch at run time
    a = oneArray(zeros(Float32, 1))
    @oneapi items=1 store_kernel(a, 1f0)
    k = @oneapi launch=false store_kernel(a, 1f0)
    k(a, 2f0)
    @test Array(a)[1] == 2f0
    for m in [methods(k)..., methods(Core.kwcall, Tuple{NamedTuple, typeof(k), Vararg})...],
        mi in Base.specializations(m)
        params = mi.specTypes.parameters
        any(p -> p === typeof(k), params) || continue
        @test !Base.isvarargtype(last(params))
    end

    # creating and launching a kernel mostly doesn't depend on its signature, so launching
    # kernels with new signatures shouldn't compile that again
    shared = (oneAPI.zefunction_link, oneAPI.launch_kernel, oneL0.set_argument_value!)
    nspecializations() =
        sum(f -> sum(m -> length(collect(Base.specializations(m))), methods(f)), shared)
    before = nspecializations()
    for T in (Int32, Int16, UInt8)
        b = oneArray(zeros(T, 1))
        @oneapi items=1 store_kernel(b, one(T))
        @test Array(b)[1] == one(T)
        @oneapi items=1 store_twice_kernel(b, one(T))
        @test Array(b)[1] == T(2)
    end
    @test nspecializations() == before
end


@testset "reflection" begin
    oneAPI.code_lowered(dummy, Tuple{})
    oneAPI.code_typed(dummy, Tuple{})
    oneAPI.code_warntype(devnull, dummy, Tuple{})
    oneAPI.code_llvm(devnull, dummy, Tuple{})
    oneAPI.code_spirv(devnull, dummy, Tuple{})

    @device_code_lowered @oneapi dummy()
    @device_code_typed @oneapi dummy()
    @device_code_warntype io=devnull @oneapi dummy()
    @device_code_llvm io=devnull @oneapi dummy()
    @device_code_spirv io=devnull @oneapi dummy()

    mktempdir() do dir
        @device_code dir=dir @oneapi dummy()
    end

    @test_throws ErrorException @device_code_lowered nothing

    # make sure kernel name aliases are preserved in the generated code
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io optimize=false @oneapi dummy())))
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io @oneapi dummy())))
    @test occursin("dummy", sprint(io->(@device_code_spirv io=io @oneapi dummy())))

    # make sure invalid kernels can be partially reflected upon
    let
        invalid_kernel() = throw()
        @test_throws oneAPI.InvalidIRError @oneapi invalid_kernel()
        @test_throws oneAPI.InvalidIRError @grab_output @device_code_warntype @oneapi invalid_kernel()
        out, err = @grab_output begin
            try
                @device_code_warntype @oneapi invalid_kernel()
            catch
            end
        end
        @test occursin("Body::Union{}", err)
    end

    # set name of kernel
    @test occursin("mykernel", sprint(io->(@device_code_llvm io=io begin
        k = zefunction(dummy, name="mykernel")
        k()
    end)))

    @test oneAPI.return_type(identity, Tuple{Int}) === Int
    @test oneAPI.return_type(sin, Tuple{Float32}) === Float32
    @test oneAPI.return_type(getindex, Tuple{oneDeviceArray{Float32,1,1},Int32}) === Float32
    @test oneAPI.return_type(getindex, Tuple{Base.RefValue{Integer}}) === Integer
end


@testset "external kernels" begin
    @eval module KernelModule
        export external_dummy
        external_dummy() = return
    end
    import .KernelModule
    @oneapi KernelModule.external_dummy()
    @eval begin
        using .KernelModule
        @oneapi external_dummy()
    end

    @eval module WrapperModule
        using oneAPI
        @eval dummy() = return
        wrapper() = @oneapi dummy()
    end
    WrapperModule.wrapper()
end


@testset "shared method table" begin
    mod = @eval module $(gensym())
        on_device() = Int32(0)
        Base.Experimental.@overlay $(oneAPI.GPUCompiler.SHARED_METHOD_TABLE) on_device() = Int32(1)
        function kernel(a)
            @inbounds a[1] = on_device()
            return
        end
    end

    a = oneArray(Int32[0])
    Base.invokelatest() do
        @oneapi mod.kernel(a)
    end
    @test Array(a)[1] == 1
end

@testset "calling device function" begin
    @noinline child(i) = sink(i)
    function parent()
        child(1)
        return
    end

    @oneapi parent()
end


@testset "varargs" begin
    function kernel(args...)
        oneAPI.@print(args[2])
        return
    end

    _, out = @grab_output begin
        @oneapi kernel(1, 2, 3)
        synchronize()
    end
    @test out == "2"
end

end


############################################################################################

@testset "argument passing" begin

dims = (16, 16)
len = prod(dims)

@testset "CPU array argument" begin
    kernel(a) = return
    a = oneArray([1f0])
    b = [1f0]
    for f in (() -> @oneapi(kernel(b)), () -> a .+ b)
        err = try f(); nothing catch err; err end
        @test err isa ArgumentError
        @test occursin("oneArray", sprint(showerror, err))
    end
end

@testset "manually allocated" begin
    function kernel(input, output)
        i = get_global_id()

        val = input[i]
        output[i] = val

        return
    end

    input = round.(rand(Float32, dims) * 100)
    output = similar(input)

    input_dev = oneArray(input)
    output_dev = oneArray(output)

    @oneapi items=len kernel(input_dev, output_dev)
    @test input ≈ Array(output_dev)
end


@testset "scalar through single-value array" begin
    function kernel(a, x)
        i = get_global_id()
        max = get_global_size()
        if i == max
            _val = a[i]
            x[] = _val
        end
        return
    end

    arr = round.(rand(Float32, dims) * 100)
    val = [0f0]

    arr_dev = oneArray(arr)
    val_dev = oneArray(val)

    @oneapi items=len kernel(arr_dev, val_dev)
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "scalar through single-value array, using device function" begin
    @noinline child(a, i) = a[i]
    function parent(a, x)
        i = get_global_id()
        max = get_global_size()
        if i == max
            _val = child(a, i)
            x[] = _val
        end
        return
    end

    arr = round.(rand(Float32, dims) * 100)
    val = [0f0]

    arr_dev = oneArray(arr)
    val_dev = oneArray(val)

    @oneapi items=len parent(arr_dev, val_dev)
    @test arr[dims...] ≈ Array(val_dev)[1]
end


@testset "tuples" begin
    # issue #7: tuples not passed by pointer

    function kernel(keeps, out)
        if keeps[1]
            out[] = 1
        else
            out[] = 2
        end
        return
    end

    keeps = (true,)
    d_out = oneArray(zeros(Int))

    @oneapi kernel(keeps, d_out)
    @test Array(d_out)[] == 1
end


@testset "ghost function parameters" begin
    # bug: ghost type function parameters are elided by the compiler

    len = 60
    a = rand(Float32, len)
    b = rand(Float32, len)
    c = similar(a)

    d_a = oneArray(a)
    d_b = oneArray(b)
    d_c = oneArray(c)

    @eval struct ExecGhost end

    function kernel(ghost, a, b, c)
        i = get_global_id()
        c[i] = a[i] + b[i]
        return
    end
    @oneapi items=len kernel(ExecGhost(), d_a, d_b, d_c)
    @test a+b == Array(d_c)


    # bug: ghost type function parameters confused aggregate type rewriting

    function kernel(ghost, out, aggregate)
        i = get_global_id()
        out[i] = aggregate[1]
        return
    end
    @oneapi items=len kernel(ExecGhost(), d_c, (42,))

    @test all(val->val==42, Array(d_c))
end


@testset "arguments of a union of singleton types" begin
    # JuliaLang/julia#63482: `GC.@preserve` of such a value crashed codegen
    function kernel(a, op)
        @inbounds a[1] = op(a[1], Int16(1))
        return
    end

    function compile(a)
        kernels = []
        for op in (max, min)
            push!(kernels, @oneapi launch=false kernel(a, op))
        end
        kernels
    end
    @test length(compile(oneArray(Int16[5]))) == 2

    function launch(a)
        for op in (max, min)
            @oneapi kernel(a, op)
        end
        Array(a)
    end
    @test launch(oneArray(Int16[5])) == Int16[1]
end


@testset "immutables" begin
    # issue #15: immutables not passed by pointer

    function kernel(ptr, b)
        ptr[] = imag(b)
        return
    end

    arr = oneArray(zeros(Float32))
    x = ComplexF32(2,2)

    @oneapi kernel(arr, x)
    @test Array(arr)[] == imag(x)
end


@testset "automatic recompilation" begin
    arr = oneArray(zeros(Int))

    function kernel(ptr)
        ptr[] = 1
        return
    end

    @oneapi kernel(arr)
    @test Array(arr)[] == 1

    function kernel2(ptr)
        ptr[] = 2
        return
    end

    @oneapi kernel2(arr)
    @test Array(arr)[] == 2
end


@testset "automatic recompilation (bis)" begin
    arr = oneArray(zeros(Int))

    @eval doit(ptr) = ptr[] = 1

    function kernel(ptr)
        doit(ptr)
        return
    end

    @oneapi kernel(arr)
    @test Array(arr)[] == 1

    @eval doit(ptr) = ptr[] = 2

    @oneapi kernel(arr)
    @test Array(arr)[] == 2
end


@testset "non-isbits arguments" begin
    function kernel1(T, i)
        sink(i)
        return
    end
    @oneapi kernel1(Int, 1)

    function kernel2(T, i)
        sink(unsafe_trunc(T,i))
        return
    end
    @oneapi kernel2(Int, 1f0)
end


@testset "splatting" begin
    function kernel(out, a, b)
        out[] = a+b
        return
    end

    out = [0]
    out_dev = oneArray(out)

    @oneapi kernel(out_dev, 1, 2)
    @test Array(out_dev)[1] == 3

    all_splat = (out_dev, 3, 4)
    @oneapi kernel(all_splat...)
    @test Array(out_dev)[1] == 7

    partial_splat = (5, 6)
    @oneapi kernel(out_dev, partial_splat...)
    @test Array(out_dev)[1] == 11
end

@testset "object invoke" begin
    # this mimics what is generated by closure conversion

    @eval struct KernelObject{T} <: Function
        val::T
    end

    function (self::KernelObject)(a)
        a[] = self.val
        return
    end

    function outer(a, val)
       inner = KernelObject(val)
       @oneapi inner(a)
    end

    a = [1f0]
    a_dev = oneArray(a)

    outer(a_dev, 2f0)

    @test Array(a_dev) ≈ [2f0]
end

@testset "closures" begin
    function outer(a_dev, val)
       function inner(a)
            # captures `val`
            a[] = val
            return
       end
       @oneapi inner(a_dev)
    end

    a = [1f0]
    a_dev = oneArray(a)

    outer(a_dev, 2f0)

    @test Array(a_dev) ≈ [2f0]
end

@testset "conversions" begin
    @eval struct Host   end
    @eval struct Device end

    Adapt.adapt_storage(::oneAPI.KernelAdaptor, a::Host) = Device()

    Base.convert(::Type{Int}, ::Host)   = 1
    Base.convert(::Type{Int}, ::Device) = 2

    out = [0]

    # convert arguments
    out_dev = oneArray(out)
    let arg = Host()
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg)
            return
        end
        @oneapi kernel(arg, out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # convert tuples
    out_dev = oneArray(out)
    let arg = (Host(),)
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg[1])
            return
        end
        @oneapi kernel(arg, out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # convert named tuples
    out_dev = oneArray(out)
    let arg = (a=Host(),)
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg.a)
            return
        end
        @oneapi kernel(arg, out_dev)
        @test Array(out_dev) ≈ [2]
    end

    # don't convert structs
    out_dev = oneArray(out)
    @eval struct Nested
        a::Host
    end
    let arg = Nested(Host())
        @test Array(out_dev) ≈ [0]
        function kernel(arg, out)
            out[] = convert(Int, arg.a)
            return
        end
        @oneapi kernel(arg, out_dev)
        @test Array(out_dev) ≈ [1]
    end
end

@testset "argument count" begin
    val = [0]
    val_dev = oneArray(val)
    for i in (1, 10, 20, 34)
        variables = ('a':'z'..., 'A':'Z'...)
        params = [Symbol(variables[j]) for j in 1:i]
        # generate a kernel
        body = quote
            function kernel(arr, $(params...))
                arr[] = $(Expr(:call, :+, params...))
                return
            end
        end
        eval(body)
        args = [j for j in 1:i]
        call = Expr(:call, :kernel, val_dev, args...)
        cudacall = :(@oneapi $call)
        eval(cudacall)
        @test Array(val_dev)[1] == sum(args)
    end
end

@testset "keyword arguments" begin
    @eval inner_kwargf(foobar;foo=1, bar=2) = nothing

    @oneapi (()->inner_kwargf(42;foo=1,bar=2))()

    @oneapi (()->inner_kwargf(42))()

    @oneapi (()->inner_kwargf(42;foo=1))()

    @oneapi (()->inner_kwargf(42;bar=2))()

    @oneapi (()->inner_kwargf(42;bar=2,foo=1))()
end

@testset "captured values" begin
    function f(capture::T) where {T}
        function kernel(ptr)
            ptr[] = capture
            return
        end

        arr = oneArray(zeros(T))
        @oneapi kernel(arr)

        return Array(arr)[1]
    end

    using Test
    @test f(1) == 1
    @test f(2) == 2
end

end

############################################################################################

@testset "#55: invalid integers created by alloc_opt" begin
    function f(a)
        x = SVector(0f0, 0f0)
        v = MVector{3, Float32}(undef)
        for (i,_) in enumerate(x)
            v[i] = 1f0
        end
        a[1] = v[1]
        return nothing
    end
    @oneapi f(oneArray(zeros(Float32, 1)))
end

@testset "#172: switch narrowed to an invalid integer" begin
    # without native integer widths in the datalayout, InstCombine narrows this to a
    # `switch i2`, which the Khronos translator rejects or crashes on
    function f(out, x)
        i = get_global_id()
        v = x[i] & 3
        if v == 1
            out[i] = 10
        elseif v == 2
            out[i] = 20
        elseif v == 3
            out[i] = 30
        else
            out[i] = 0
        end
        return
    end
    x = oneArray(collect(0:7))
    out = similar(x)
    @oneapi items=8 f(out, x)
    @test Array(out) == [0, 10, 20, 30, 0, 10, 20, 30]
end

@testset "#160: barrier intrinsincs should be convergent" begin
    # Solve L*x = r and store the result in r.
    function cpu(n::Int, r::Vector{Float32})
        for j=1:n
            temp = r[j]/2f0
            for k=j+1:n
                r[k] = r[k] - 2f0*temp
            end
            r[j] = temp
        end
    end
    function gpu(::Val{n},r_) where {n}
        tx = get_local_id()
        bx = get_group_id()

        r = oneLocalArray(Float32, n)

        r[tx] = r_[tx]

        barrier(0)

        for j=1:n
            if tx == 1
                r[j] = r[j] / 2f0
            end
            barrier(0)

            if tx > j && tx <= 4
                r[tx] = r[tx] - 2f0*r[j]
            end
            barrier(0)
        end

        if bx == 1
            r_[tx] = r[tx]
        end

        return
    end

    A = Float32[10, 10]
    n = length(A)

    hA = copy(A)
    cpu(n,hA)

    dA = oneArray(A)
    @oneapi items=n gpu(Val(n),dA)

    @test Array(dA) == hA
end

@testset "NEO#172" begin
    # conversions from integers to pointers resulted in lost memory stores

    function kernel(ptr)
        ptr = reinterpret(Core.LLVMPtr{Float32, AS.CrossWorkgroup}, ptr)
        unsafe_store!(ptr, 42)
        return
    end

    if VERSION < v"1.12"
        a = oneArray(Float32[0])
        @oneapi kernel(reinterpret(Ptr{Float32}, pointer(a)))
        @test Array(a) == [42]
    else
        @test_broken false
    end
end

@testset "device pointer arguments" begin
    # pointers to device memory are passed to kernels as global pointers; a plain `Ptr`
    # would be a private pointer in SPIR-V, losing the store (#458)
    function kernel(ptr, val)
        unsafe_store!(ptr, val)
        return
    end

    for B in (oneL0.DeviceBuffer, oneL0.SharedBuffer, oneL0.HostBuffer)
        a = oneVector{Int,B}([1])
        @oneapi kernel(pointer(a), 42)
        @test Array(a) == [42]
    end
end

############################################################################################

# A chain of N live accumulators forces the register allocator to spill once N exceeds
# the register file (256 spills on both PVC and DG2); the reversed index in the update
# keeps every element live across the chain, and everything is unrolled through Val so
# no value is boxed. Top-level definitions: inside the testset the self-recursion of
# `hedge_rounds` becomes a boxed closure capture, which is not a bitstype.
hedge_step(acc::NTuple{N, T}) where {N, T} =
    ntuple(j -> muladd(acc[j], T(1.0000001), acc[N - j + 1]), Val(N))
hedge_rounds(acc, ::Val{0}) = acc
hedge_rounds(acc, ::Val{R}) where {R} = hedge_rounds(hedge_step(acc), Val(R - 1))
function hedge_spill_kernel(out, a, ::Val{N}, ::Val{R}) where {N, R}
    i = get_global_id()
    acc = hedge_rounds(ntuple(j -> a[i] + Float32(j), Val(N)), Val(R))
    s = 0.0f0
    @inbounds for j = 1:N
        s += acc[j]
    end
    @inbounds out[i] = s
    return
end

@testset "scratch hedge" begin
    a = oneAPI.ones(Float32, 256)
    out = oneAPI.zeros(Float32, 256)
    synchronize()

    k = @oneapi launch=false hedge_spill_kernel(out, a, Val(256), Val(2))
    spill = oneL0.spill_mem_size(k.fun)

    if spill == 0
        # the hedge only acts on spilling kernels; nothing to observe on this device
        @test_skip "kernel did not spill on this device/compiler"
    else
        # queues are task-local, so a fresh task gets a fresh queue with a zero
        # high-water mark; observe there, assert on the test task
        observed = fetch(@async begin
            s = global_stream(context(), device())
            hwm0 = s.scratch_hwm
            c0 = oneL0.SCRATCH_HEDGE_COUNT[]
            @oneapi items=64 groups=4 hedge_spill_kernel(out, a, Val(256), Val(2))
            c1 = oneL0.SCRATCH_HEDGE_COUNT[]
            # same spill tier: must not fire again
            @oneapi items=64 groups=4 hedge_spill_kernel(out, a, Val(256), Val(2))
            c2 = oneL0.SCRATCH_HEDGE_COUNT[]
            # a storm of no-spill kernels must not fire the hedge either
            for _ in 1:32
                @oneapi dummy()
            end
            c3 = oneL0.SCRATCH_HEDGE_COUNT[]
            (hwm0, s.scratch_hwm, c1 - c0, c2 - c1, c3 - c2)
        end)
        @test observed == (0, spill, 1, 0, 0)

        # with the hedge disabled the high-water mark is still maintained
        old = oneL0.SCRATCH_HEDGE[]
        oneL0.SCRATCH_HEDGE[] = false
        try
            observed = fetch(@async begin
                s = global_stream(context(), device())
                c0 = oneL0.SCRATCH_HEDGE_COUNT[]
                @oneapi items=64 groups=4 hedge_spill_kernel(out, a, Val(256), Val(2))
                (oneL0.SCRATCH_HEDGE_COUNT[] - c0, s.scratch_hwm)
            end)
            @test observed == (0, spill)
        finally
            oneL0.SCRATCH_HEDGE[] = old
        end
    end
end

# ordering probe for the immediate submission stream
function stream_iota_kernel(a, off)
    i = get_global_id()
    @inbounds a[i] = i + off
    return
end

@testset "immediate submission stream" begin
    n = 1024
    a = oneAPI.zeros(Int32, n)
    host = zeros(Int32, n)
    expected = zeros(Int32, n)

    # in-order: kernel write then synchronizing D2H readback, iterated
    ok = true
    for iter in 1:100
        @oneapi items=256 groups=4 stream_iota_kernel(a, Int32(iter))
        copyto!(host, a)               # copies to the host synchronize the stream
        expected .= Int32.(1:n) .+ Int32(iter)
        ok &= host == expected
    end
    @test ok

    # a raw command queue is still accepted as an explicit submission target
    queue = oneL0.ZeCommandQueue(context(), device();
                                 flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER)
    k = @oneapi launch=false stream_iota_kernel(a, Int32(0))
    k(a, Int32(0); items=256, groups=4, queue)
    oneL0.synchronize(queue)
    @test Array(a) == Int32.(1:n)

    # the LTS sync-each-submission knob applies to immediate lists
    oneL0.sync_each_submission(true) do
        @oneapi items=256 groups=4 stream_iota_kernel(a, Int32(41))
    end
    @test Array(a) == Int32.(1:n) .+ Int32(41)

    # concurrent tasks each get their own stream; alloc/free churn while launching
    # exercises the cross-task drain-before-free path on the LTS stack
    results = map(fetch, [@async begin
        b = oneAPI.ones(Float32, 4096)
        acc = true
        for i in 1:50
            c = b .* Float32(i)
            acc &= isapprox(sum(c), Float32(4096 * i); rtol=1e-4)
            oneAPI.unsafe_free!(c)
        end
        acc
    end for _ in 1:2])
    @test all(results)
end

function fill_kernel(a, x)
    @inbounds a[get_global_id()] = x
    return
end

# a step of a recurrence that doesn't commute, so its result depends on the order of launches
function recurrence_kernel(a, i)
    @inbounds a[1] = 3 * a[1] + i
    return
end
function recurrence(n)
    x = 0
    for i in 1:n
        x = 3 * x + i
    end
    return x
end

function ptr_increment_kernel(ptr)
    unsafe_store!(ptr, unsafe_load(ptr) + 1)
    return
end

@testset "command batching" begin
    n = 100  # spans several batches
    batching = oneAPI.global_stream(context(), device()).batching &&
               !oneL0.sync_each_submission()

    # dependent launches execute in order, across batches
    a = oneArray(Int64[0])
    for i in 1:n
        @oneapi recurrence_kernel(a, i)
    end
    @test Array(a)[1] == recurrence(n)

    # interleaved with operations that are submitted directly
    a = oneArray(Int64[0])
    for i in 1:n
        @oneapi recurrence_kernel(a, i)
        i % 10 == 0 && copyto!(a, a)
        i % 25 == 0 && copyto!(oneArray{Int64}(undef, 1), [i])
    end
    @test Array(a)[1] == recurrence(n)

    # host accesses wait for launches that haven't been submitted yet
    a = oneArray{Int64, 1, oneL0.SharedBuffer}([0])
    for i in 1:n
        @oneapi recurrence_kernel(a, i)
    end
    @test a[1] == recurrence(n)

    # also when they were recorded by another task
    a = oneArray(Int64[0])
    wait(@async begin
        for i in 1:n
            @oneapi recurrence_kernel(a, i)
        end
    end)
    @test Array(a)[1] == recurrence(n)

    # synchronizing submits what tasks recorded before finishing, also when the memory it
    # uses isn't tracked, and when another task records on the same stream later
    a = oneArray(Int64[0])
    ptr = reinterpret(Core.LLVMPtr{Int64, oneAPI.AS.CrossWorkgroup}, pointer(a))
    stream = fetch(@async begin
        for _ in 1:3
            @oneapi ptr_increment_kernel(ptr)
        end
        oneAPI.global_stream(context(), device())
    end)
    synchronize()
    @test Array(a)[1] == 3
    wait(@async begin
        for _ in 1:3
            @oneapi queue=stream ptr_increment_kernel(ptr)
        end
    end)
    synchronize()
    @test Array(a)[1] == 6

    # including on the stream of a task that is still running
    streams = Channel{Any}(1)
    done = Channel{Nothing}(1)
    owner = @async begin
        # with a batch recorded, which the other task records into
        for _ in 1:2
            @oneapi ptr_increment_kernel(ptr)
        end
        put!(streams, oneAPI.global_stream(context(), device()))
        take!(done)
    end
    stream = take!(streams)
    @test stream.recorder.value === owner
    wait(@async begin
        for _ in 1:3
            @oneapi queue=stream ptr_increment_kernel(ptr)
        end
    end)
    synchronize()
    @test Array(a)[1] == 11
    put!(done, nothing)
    wait(owner)

    # memory used by recorded launches stays alive until they have executed
    a = oneArray(Int64[0])
    GC.enable(false)
    try
        @oneapi recurrence_kernel(a, 1)  # submitted directly
        bytes = 1 << 23                  # not sub-allocated
        before = oneAPI._allocated_bytes[]
        b = oneArray{Int64}(undef, bytes ÷ sizeof(Int64))
        @oneapi items=256 groups=4 fill_kernel(b, 1)
        oneAPI.unsafe_free!(b)
        batching && @test oneAPI._allocated_bytes[] == before + bytes
        synchronize()
        @test oneAPI._allocated_bytes[] == before
    finally
        GC.enable(true)
    end

    # recorded launches are submitted on request
    a = oneArray{Int64, 1, oneL0.SharedBuffer}([0])
    synchronize()
    for i in 1:3
        @oneapi recurrence_kernel(a, i)
    end
    oneAPI.flush!()
    # wait for the stream without waiting for the work on `a`, which would submit it, and
    # read the memory directly
    oneL0.synchronize(oneAPI.global_stream(context(), device()).immediate)
    @test unsafe_load(reinterpret(Ptr{Int64}, oneAPI.device_pointer(a))) == recurrence(3)

    # with all batches executing, launches are submitted directly instead of waiting for
    # one, which may not finish before this task does something (here, open the gate)
    if batching
        pool = oneL0.ZeEventPool(context(), 1, device(); flags=oneL0.ZE_EVENT_POOL_FLAG_HOST_VISIBLE)
        gate = pool[1]
        a = oneArray(Int64[0])
        # (only a few more: many submissions waiting for the gate make the driver stall, and
        # eventually give up on the device)
        m = oneAPI.MAX_BATCHES * oneAPI.BATCH_SIZE + 10
        GC.enable(false)  # freeing memory may wait for the stream on some stacks
        try
            oneL0.execute!(oneAPI.global_stream(context(), device())) do list
                oneL0.append_wait!(list, gate)
            end
            for i in 1:m
                @oneapi recurrence_kernel(a, i)
            end
            oneL0.signal(gate)
        finally
            GC.enable(true)
        end
        @test Array(a)[1] == recurrence(m)
    end

    # work recorded before using oneMKL is executed first, and work after it waits for it
    x = oneArray(ones(Float32, 1024))
    y = oneArray(zeros(Float32, 1024))
    for _ in 1:40
        x .+= 1
    end
    LinearAlgebra.axpy!(2f0, x, y)
    for _ in 1:40
        y .+= 1
    end
    @test Array(y) == fill(2f0 * 41 + 40, 1024)

    # a batch appended to an idle stream is dispatched from its own command buffer, which
    # NEO 26.35 overflowed with a pending tag update after recycling its ring buffer
    # (intel/compute-runtime#1027)
    x = oneArray(zeros(Float32, 1024))
    y = similar(x)
    for _ in 1:2000
        copyto!(y, x)
        oneAPI.mkl_boundary!()
        for _ in 1:4
            x .+= 1
        end
        oneAPI.mkl_boundary!()
    end
    @test Array(x) == fill(8000f0, 1024)

    # a failed submission is reported until the stream is synchronized
    @test fetch(@async begin
        a = oneArray(Int64[0])
        stream = oneAPI.global_stream(context(), device())
        err = ErrorException("submission failed")
        stream.error = err
        caught(f) = try f(); nothing catch e; e end
        results = [caught(() -> @oneapi recurrence_kernel(a, 1)),
                   caught(() -> copyto!(a, [1])),
                   caught(() -> oneAPI.flush!(stream)),
                   caught(() -> oneAPI.drain(stream)),
                   caught(() -> synchronize())]
        # also when crossing a new spill tier, which synchronizes first
        stream.error = err
        stream.scratch_hwm = -1
        push!(results, caught(() -> @oneapi recurrence_kernel(a, 1)))
        stream.error = nothing
        all(r -> r === err, results) && caught(() -> synchronize()) === nothing
    end)

    # and reported once by whoever synchronizes when the task that recorded didn't
    err = ErrorException("submission failed")
    a = oneArray(Int64[0])
    wait(@async begin
        for i in 1:2
            @oneapi recurrence_kernel(a, i)
        end
        oneAPI.global_stream(context(), device()).error = err
    end)
    @test try synchronize(); nothing catch e; e end === err
    @test try synchronize(); nothing catch e; e end === nothing
end

@testset "cooperative synchronize" begin
    # keep the work on the task's stream waiting for an event until another task signals
    # it, which makes these tests independent of timing: synchronizing can only return after
    # that task got to run. if it cannot (e.g., because synchronizing blocks the thread), a
    # watchdog on another thread signals the event after a while instead, and records that.
    pool = oneL0.ZeEventPool(context(), 1, device();
                             flags=oneL0.ZE_EVENT_POOL_FLAG_HOST_VISIBLE)
    gate = pool[1]
    timeout = UInt64(60_000_000_000)    # ns

    # run `f` with the task's stream waiting for the gate, which another task on the same
    # thread opens, but only after it got to run many more times than the polling at the
    # start of a synchronization yields. returns whether `f` only returned after the gate
    # had been opened, and the watchdog did not have to.
    function gated(f)
        reset(gate)
        # while the gate is closed, nothing on the host may wait for the GPU to become
        # idle, as freeing memory does on some stacks. so avoid running finalizers, by
        # collecting beforehand and not collecting while the gate is closed.
        GC.gc(true)
        gc_enabled = GC.enable(false)
        watching = Threads.Atomic{Bool}(false)
        opened = Threads.Atomic{Bool}(false)
        in_time, timed_out = false, true
        local watchdog, opener
        try
            # the watchdog waits on one of `cooperative_wait`'s worker threads, which keep
            # running when this thread is blocked
            watchdog = @async oneL0.cooperative_wait(gate; spin=false) do gate
                watching[] = true
                res = oneL0.unchecked_zeEventHostSynchronize(gate, timeout)
                res == oneL0.RESULT_NOT_READY || return false
                oneL0.signal(gate)
                return true
            end
            while !watching[]
                yield()
            end

            # (without synchronizing after the submission, which would wait for the gate)
            oneL0.sync_each_submission(false) do
                oneL0.execute!(oneAPI.global_stream(context(), device())) do list
                    oneL0.append_wait!(list, gate)
                end
            end
            opener = @async begin
                for _ in 1:10_000
                    yield()
                end
                opened[] = true
                oneL0.signal(gate)
            end
            f()
            in_time = opened[]
        finally
            # also when `f` failed, as the gate is reused
            oneL0.signal(gate)
            @isdefined(opener) && wait(opener)
            @isdefined(watchdog) && (timed_out = something(fetch(watchdog)))
            GC.enable(gc_enabled)
        end
        synchronize()
        return in_time && !timed_out
    end

    # compile everything beforehand, as that could wait for the GPU to become idle
    a = oneArray{Int32}(undef, 64)
    @oneapi items=64 fill_kernel(a, Int32(0))
    synchronize()
    @test gated(synchronize)

    @test gated() do
        oneL0.sync_each_submission(false) do
            @oneapi items=64 fill_kernel(a, Int32(1))
        end
        synchronize()
    end
    @test Array(a) == fill(Int32(1), 64)

    # freeing memory, e.g. from a finalizer, doesn't wait for the GPU
    if !oneL0.LTS[]
        b = oneArray{Int32}(undef, 2^20)    # too large to be sub-allocated
        @test gated() do
            oneAPI.unsafe_free!(b)
            synchronize()
        end
    end

    @test gated() do
        oneL0.sync_each_submission(false) do
            @oneapi items=64 fill_kernel(a, Int32(2))
        end
        synchronize(oneAPI.global_stream(context(), device()))
    end
    @test Array(a) == fill(Int32(2), 64)

    # synchronizing after every launch, as on the LTS stack
    @test gated() do
        oneL0.sync_each_submission(true) do
            @oneapi items=64 fill_kernel(a, Int32(3))
        end
    end
    @test Array(a) == fill(Int32(3), 64)

    # blocking synchronization is still available
    @oneapi items=64 fill_kernel(a, Int32(4))
    @test synchronize(; blocking=true) === nothing
    @test Array(a) == fill(Int32(4), 64)
end

############################################################################################

# Keep allocation consumers at top level so kernels do not capture test state.

@noinline heap_consume(r::Base.RefValue{Float32}) = r[] + 1.0f0

struct HeapAnyBox
    x::Any
end
@noinline heap_consume(b::HeapAnyBox) = (b.x::Float32) * 2.0f0

@testset "device heap" begin
    # Keep objects alive across a call so allocation survives Julia/LLVM optimization.
    function ref_kernel(a)
        i = get_global_id()
        @inbounds a[i] = heap_consume(Ref(a[i]))
        return
    end
    a = oneArray(Float32[41])
    @oneapi ref_kernel(a)
    @test Array(a) == [42]

    # so is a struct whose `Any` field boxes its value
    function anybox_kernel(a)
        i = get_global_id()
        @inbounds a[i] = heap_consume(HeapAnyBox(a[i]))
        return
    end
    a = oneArray(Float32[21])
    @oneapi anybox_kernel(a)
    @test Array(a) == [42]

    # every work-item has its own heap
    n = 4096
    a = oneArray(Float32.(1:n))
    @oneapi items = 256 groups = n ÷ 256 ref_kernel(a)
    @test Array(a) == Float32.(2:(n + 1))

    # objects stay valid across later allocations by the same work-item. Selected with
    # branches rather than indexed from a tuple: under `--check-bounds=yes` the bounds check
    # of a dynamic tuple index reads the field count through a host pointer, which cannot
    # work on the device.
    function select_kernel(a, idx)
        i = get_global_id()
        r1 = Ref(a[i] * 1)
        r2 = Ref(a[i] * 2)
        r3 = Ref(a[i] * 3)
        r4 = Ref(a[i] * 4)
        r = idx == 1 ? r1 : idx == 2 ? r2 : idx == 3 ? r3 : r4
        @inbounds a[i] = heap_consume(r)
        return
    end
    a = oneArray(Float32[1, 2, 3, 4])
    @oneapi items = 4 select_kernel(a, 3)
    @test Array(a) == Float32[4, 7, 10, 13]

    # nothing is freed: a work-item that allocates more than the heap holds runs out of
    # memory, which is reported, and exits without writing its result
    function loop_kernel(a, n)
        i = get_global_id()
        @inbounds x = a[i]
        for _ in 1:n
            x = heap_consume(Ref(x))
        end
        @inbounds a[i] = x
        return
    end
    fits = oneAPI.HEAP_SIZE ÷ oneAPI.HEAP_ALIGNMENT
    a = oneArray(Float32.(1:256))
    @oneapi items = 256 loop_kernel(a, fits)
    @test Array(a) == Float32.(1:256) .+ fits
    # A later launch starts with an empty heap, even after using the entire arena.
    @oneapi items = 256 loop_kernel(a, fits)
    @test Array(a) == Float32.(1:256) .+ 2 * fits
    a = oneArray(Float32[1])
    err, out = @grab_output begin
        @oneapi loop_kernel(a, fits + 1)
        try synchronize(); nothing catch err; err end
    end
    @test err isa KernelException
    @test occursin("Out of dynamic GPU memory", out)
    @test Array(a) == [1]

    # Failed requests must not consume space. Exercise the size arithmetic directly,
    # including overflow when rounding up and allocations with different sizes.
    function allocation_kernel(out, sizes)
        for i in eachindex(sizes)
            ptr = oneAPI.malloc(sizes[i])
            out[i] = UInt(ptr)
        end
        return
    end
    sizes = oneArray(
        Csize_t[
            typemax(Csize_t), oneAPI.HEAP_SIZE + 1, 1, 17,
            oneAPI.HEAP_SIZE - 3 * oneAPI.HEAP_ALIGNMENT, 1,
        ]
    )
    out = oneAPI.zeros(UInt, length(sizes))
    @oneapi allocation_kernel(out, sizes)
    ptrs = Array(out)
    @test ptrs[[1, 2, 6]] == [0, 0, 0]
    @test all(!iszero, ptrs[3:5])
    @test all(p -> p % oneAPI.HEAP_ALIGNMENT == 0, ptrs[3:5])
    @test ptrs[4] - ptrs[3] == oneAPI.HEAP_ALIGNMENT
    @test ptrs[5] - ptrs[4] == 2 * oneAPI.HEAP_ALIGNMENT

    # These valid inputs must compile even if exception paths retain boxed arguments
    # (GPUCompiler.jl#906, exposed by preserving inferred invoke specializations).
    powers = Float32[1, 2, 4, 8]
    @test Array(exponent.(oneArray(powers))) == exponent.(powers)
    z = ComplexF32[1 + 2im, -3 + 4im, 0, 2 - 3im]
    @test Array(sqrt.(oneArray(z))) ≈ sqrt.(z)

    # only kernels that allocate carry a heap
    function plain_kernel(a)
        i = get_global_id()
        @inbounds a[i] += 1.0f0
        return
    end
    T = Tuple{oneDeviceVector{Float32, oneAPI.AS.CrossWorkgroup}}
    @test occursin(r"%heap\d* = alloca", sprint(io -> oneAPI.code_llvm(io, ref_kernel, T; kernel = true)))
    @test !occursin(r"%heap\d* = alloca", sprint(io -> oneAPI.code_llvm(io, plain_kernel, T; kernel = true)))
end

# the exception a kernel threw, as reported by synchronizing `stream`
function kernel_exception(stream=oneAPI.global_stream(context(), device()))
    try
        synchronize(stream)
        return nothing
    catch err
        err isa KernelException || rethrow()
        return err
    end
end

function div_kernel(a, d)
    a[1] = div(a[1], d)
    return
end

@testset "exceptions" begin
    a = oneArray([0])

    # at debug level 0, only that an exception was thrown
    @oneapi debug_level=0 store_checked(a, 2)
    err = kernel_exception()
    @test err isa KernelException
    @test err.name == "" && err.reason == "" && isempty(err.backtrace)
    @test err.work_item == err.work_group == (0, 0, 0)
    @test kernel_exception() === nothing

    # at level 1, also the exception's type and reason, as passed to `@gputhrow`
    @oneapi debug_level=1 store_checked(a, 2)
    err = kernel_exception()
    @test err.name == "BoundsError"
    @test err.reason == "Out-of-bounds array access"
    @test err.work_item == (0, 0, 0) && isempty(err.backtrace)
    @test occursin("A BoundsError was thrown on", sprint(showerror, err))
    # or as derived by GPUCompiler
    @oneapi debug_level=1 div_kernel(a, 0)
    err = kernel_exception()
    @test err.name == "exception" && err.reason == ""

    # at level 2, also the position of the work-item, and a backtrace
    @oneapi debug_level=2 items=4 groups=3 store_checked(a, 2)
    err = kernel_exception()
    @test err.name == "BoundsError"
    @test all(1 .<= err.work_item .<= (4, 1, 1))
    @test all(1 .<= err.work_group .<= (3, 1, 1))
    @test endswith(err.backtrace[1][1], "throw_boundserror")
    @test err.backtrace[end][1] == "store_checked"
    @test endswith(err.backtrace[end][2], "execution.jl")
    @test occursin(r"thrown by work-item \d×1×1 in work-group \d×1×1", sprint(showerror, err))
    @oneapi debug_level=2 div_kernel(a, 0)
    err = kernel_exception()
    @test err.name == "exception"
    @test err.backtrace[end][1] == "div_kernel"

    # strings are reported by address, which the host only reads if it is valid
    stream = oneAPI.global_stream(context(), device())
    @oneapi debug_level=2 store_checked(a, 2)
    oneAPI.flush!(stream)
    oneL0.synchronize(stream.immediate)
    report = convert(Ptr{UInt8}, stream.exceptions.report)
    field(T, name) = report + fieldoffset(T, Base.fieldindex(T, name))
    unsafe_store!(convert(Ptr{UInt64}, field(oneAPI.ExceptionReport, :name)),
                  0x00000deadbeef000)
    unsafe_store!(convert(Ptr{UInt64}, field(oneAPI.ExceptionReport, :frames)),
                  typemax(UInt64))
    err = kernel_exception()
    @test err.name == "" && err.reason == "Out-of-bounds array access"
    @test err.backtrace[1][1] == "" && err.backtrace[end][1] == "store_checked"
    @test kernel_exception() === nothing

    # the stream remains usable, and other work-items keep running
    function partially_oob_kernel(a)
        i = get_global_id()
        a[i + 1] = i
        return
    end
    a = oneArray(zeros(Int, 256))
    @oneapi items=256 partially_oob_kernel(a)
    @test kernel_exception() isa KernelException
    @test Array(a)[2:end] == 1:255
    @oneapi store_checked(a, 1)
    @test kernel_exception() === nothing
    @test Array(a)[1] == 1

    # an exception is reported by the synchronization following it, also from a batch
    a = oneArray([0])
    for i in 1:10
        @oneapi store_checked(a, i == 5 ? 2 : 1)
    end
    @test kernel_exception() isa KernelException
    @test kernel_exception() === nothing

    # and by host accesses that synchronize
    @oneapi store_checked(a, 2)
    @test_throws KernelException Array(a)
    b = oneArray{Int, 1, oneL0.SharedBuffer}([0])
    @oneapi store_checked(b, 2)
    @test_throws KernelException b[1]
    # also when the memory is another task's, which the host waits for
    wait(@async @oneapi store_checked(a, 2))
    @test_throws KernelException Array(a)
    @test kernel_exception() === nothing

    # kernels that use the device heap too
    function heap_oob_kernel(a, i)
        x = heap_consume(Ref(a[1]))
        a[i] = x
        return
    end
    a = oneArray(Float32[1])
    @oneapi debug_level=1 heap_oob_kernel(a, 2)
    @test kernel_exception().name == "BoundsError"
    @oneapi debug_level=1 heap_oob_kernel(a, 1)
    @test kernel_exception() === nothing
    @test Array(a) == [2]

    # every stream reports the exceptions of its own kernels
    a = oneArray([0])
    launched = Channel{Nothing}(1)
    checked = Channel{Nothing}(1)
    owner = @async begin
        @oneapi store_checked(a, 2)
        put!(launched, nothing)
        take!(checked)
        kernel_exception()
    end
    take!(launched)
    @test kernel_exception() === nothing
    put!(checked, nothing)
    @test fetch(owner) isa KernelException
    # also when launching on another task's stream
    stream = fetch(@async oneAPI.global_stream(context(), device()))
    @oneapi queue=stream store_checked(a, 2)
    @test kernel_exception(stream) isa KernelException
    @test kernel_exception() === nothing

    # and of tasks that finished without synchronizing, by whoever adopts their stream
    adopted() = try synchronize(); nothing catch err; err end
    wait(@async @oneapi store_checked(a, 2))
    @test adopted() isa KernelException
    wait(@async begin
        @oneapi store_checked(a, 1)
        @oneapi store_checked(a, 2)
    end)
    @test adopted() isa KernelException
    @test adopted() === nothing
    # unless something waited for their results first
    wait(@async @oneapi store_checked(a, 2))
    @test_throws KernelException Array(a)
    @test adopted() === nothing
    task = @async begin
        @oneapi store_checked(a, 2)
        try Array(a); nothing catch err; err end
    end
    @test fetch(task) isa KernelException
    @test adopted() === nothing
    # a failed submission is reported first, the exception by the next synchronization
    err = ErrorException("submission failed")
    wait(@async begin
        @oneapi store_checked(a, 2)
        oneAPI.global_stream(context(), device()).error = err
    end)
    @test adopted() === err
    @test adopted() isa KernelException
    @test adopted() === nothing
    # but not once a task that is still running launched on their stream, while adopting
    ctx, dev = context(), device()
    pool = oneL0.ZeEventPool(ctx, 1, dev; flags=oneL0.ZE_EVENT_POOL_FLAG_HOST_VISIBLE)
    gate = pool[1]
    # nothing may wait for the GPU while the gate is closed, as freeing memory does
    GC.gc(true)
    gc_enabled = GC.enable(false)
    try
        # the adopter's own stream is created first: doing so while the gate is closed
        # makes other streams look idle on some stacks
        ready, go = Channel{Nothing}(1), Channel{Nothing}(1)
        adopter = @async begin
            synchronize()
            put!(ready, nothing)
            take!(go)
            adopted()
        end
        take!(ready)
        # (with a kernel after the wait, as waits alone don't keep a stream busy)
        stream = fetch(@async begin
            oneL0.sync_each_submission(false) do
                oneL0.execute!(oneAPI.global_stream(ctx, dev)) do list
                    oneL0.append_wait!(list, gate)
                end
                @oneapi store_checked(a, 1)
            end
            oneAPI.global_stream(ctx, dev)
        end)
        put!(go, nothing)
        for _ in 1:100
            yield()
        end
        oneL0.sync_each_submission(false) do
            @oneapi queue=stream store_checked(a, 2)
            oneAPI.flush!(stream)
        end
        oneL0.signal(gate)
        @test fetch(adopter) === nothing
    finally
        oneL0.signal(gate)
        GC.enable(gc_enabled)
    end
    @test kernel_exception(stream) isa KernelException

    # explicitly created queues have no report
    queue = oneL0.ZeCommandQueue(context(), device();
                                 flags = oneL0.ZE_COMMAND_QUEUE_FLAG_IN_ORDER)
    k = @oneapi launch=false store_checked(a, 2)
    k(a, 2; queue)
    oneL0.synchronize(queue)
    @test kernel_exception() === nothing

    # collected streams return their report, unless their work may still use it
    ctx, dev = context(), device()
    pool = oneL0.ZeEventPool(ctx, 1, dev; flags=oneL0.ZE_EVENT_POOL_FLAG_HOST_VISIBLE)
    gate = pool[1]
    busy, idle = oneAPI.create_stream(ctx, dev), oneAPI.create_stream(ctx, dev)
    oneL0.sync_each_submission(false) do
        oneL0.execute!(busy) do list
            oneL0.append_wait!(list, gate)
        end
    end
    free = oneAPI.free_report_slots[(ctx, dev)]
    finalize(busy)
    @test !(busy.exceptions in free)
    oneL0.signal(gate)
    oneL0.synchronize(busy)
    oneL0.synchronize(idle)
    finalize(idle)
    @test idle.exceptions in free

    # reporting needs no per-work-item memory, which would slow down every kernel
    for level in 0:2
        k = @oneapi launch=false debug_level=level store_checked(a, 2)
        @test oneL0.properties(k.fun).privateMemSize == 0
    end
end

@testset "loop exiting on counter wrap-around" begin
    # The SPIR-V back-end turns `i += 1; i == 0` into OpIAddCarry, and IGC's optimizer sinks
    # the extracts of its {i64, i1} result through the loop PHI. Before libigc_jll 2.34.4+5,
    # IGC's legalization then erased the intrinsic while that PHI still referenced it, a
    # use-after-free that crashed or silently corrupted the host
    # (https://buildkite.com/julialang/oneapi-dot-jl/builds/1800).
    function count_until_wrap(out, flags, n)
        i = n
        while true
            i += one(i)
            i == 0 && break
            if @inbounds(flags[1])
                @inbounds out[1] = i
                break
            end
        end
        return
    end
    out = oneAPI.zeros(UInt64, 1)
    flags = oneArray([false])
    @oneapi count_until_wrap(out, flags, typemax(UInt64) - 3)
    @test Array(out) == [0]
    copyto!(flags, [true])
    @oneapi count_until_wrap(out, flags, typemax(UInt64) - 3)
    @test Array(out) == [typemax(UInt64) - 2]
end

@testset "sub-group shuffle of a mixed-width tuple" begin
    # The shuffle packs the Bool, Int8 and Float16 into one word and unpacks them again. On
    # Gen12LP, IGC could place the byte that receives the Int8 at byte 3 of a dword, which
    # the hardware doesn't write, returning 0 for the Int8 (intel/intel-graphics-compiler#456,
    # fixed in libigc_jll 2.41.5+2). The pointers avoid bounds checks, whose throw paths
    # change the register allocation, and thus whether the bug shows.
    function shuffle_kernel(out, a, lane)
        i = get_local_id()
        val = oneAPI.KernelInterface.shfl(unsafe_load(pointer(a), i), lane)
        unsafe_store!(pointer(out), val, i)
        return
    end
    T = Tuple{Bool, Int8, Float16, UInt16}
    for n in (1, 8)
        a = T[(isodd(i), -i % Int8, Float16(i), UInt16(1000i)) for i in 1:n]
        out = oneArray{T}(undef, n)
        @oneapi items=n shuffle_kernel(out, oneArray(a), 1)
        @test Array(out) == fill(a[1], n)
    end
end
