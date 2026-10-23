@testset "profile" begin

kern(a) = (a[1] += 1f0; return)

@testset "integrated" begin
    if oneAPI.PTI.is_available()
        a = oneArray(zeros(Float32, 4))
        @oneapi kern(a)
        synchronize()

        r = oneAPI.@profile begin
            @oneapi kern(a)
            fill!(a, 0f0)
            copyto!(a, oneArray(ones(Float32, 4)))
        end
        @test r isa oneAPI.Profiling.ProfileResults
        @test Array(a) == ones(Float32, 4)

        # the launch is linked to the kernel it submitted
        i = findfirst(startswith("kern("), r.device.name)
        @test i !== nothing
        @test r.device.kind[i] == :kernel
        j = findfirst(==(r.device.id[i]), r.host.id)
        @test j !== nothing && r.host.name[j] == "zeCommandListAppendLaunchKernel"
        @test 0 <= r.device.start[i] <= r.device.stop[i] <= r.wall

        @test any(==(:copy), r.device.kind)
        @test r.device.bytes[findfirst(==(:copy), r.device.kind)] == sizeof(a)

        str = sprint(show, r)
        @test occursin("Host-side activity", str)
        @test occursin("Device-side activity", str)
        @test occursin("kern(", str)

        # work that oneAPI.jl batches is included
        if oneAPI.global_stream(context(), device()).batching && !oneL0.sync_each_submission()
            r = oneAPI.@profile trace=true begin
                for i in 1:3; @oneapi kern(a); end
            end
            @test count(startswith("kern("), r.device.name) == 3
            @test occursin("ID", sprint(show, r))
        end

        # oneMKL submits its own work
        A = oneArray(rand(Float32, 64, 64))
        A * A
        synchronize()
        r = oneAPI.@profile A * A
        @test !isempty(r.device.name)

        r = oneAPI.@bprofile time=0.1 @oneapi kern(a)
        @test count(startswith("kern("), r.device.name) > 1

        # polls for unfinished work are only shown with `raw=true`
        function spin(a, n)
            x = a[1]
            for i in 1:n
                x = x * 1.0000001f0 + 1f-7
            end
            a[1] = x
            return
        end
        @oneapi spin(a, 1)
        synchronize()
        r = oneAPI.@profile (@oneapi spin(a, 5_000_000); synchronize())
        @test any(r.host.unfinished)
        nshown(r) = parse(Int, match(r"Host-side activity: (\d+)", sprint(show, r))[1])
        @test nshown(r) <= count(!, r.host.unfinished)
        @test nshown(oneAPI.Profiling.ProfileResults(; r.device, r.host, r.wall, r.trace, raw=true)) ==
              length(r.host.name)

        # other tasks can submit work before the profile starts
        @test oneAPI.Profiling.relative_seconds(UInt64(1_000), UInt64(3_000)) == -2e-6

        # a failing session doesn't leave its records to the next one
        @test_throws "body" oneAPI.@profile (@oneapi kern(a); error("body"))
        r = oneAPI.@profile nothing
        @test isempty(r.device.name)
        @test occursin("No device-side activity", sprint(show, r))
    else
        @test_throws "not supported on this platform" oneAPI.@profile nothing
    end
end

@testset "external" begin
    a = oneArray(zeros(Float32, 4))

    # no ITT collector is attached during testing
    ret = @test_logs (:warn, r"No ITT collector") oneAPI.@profile external=true begin
        a .+= 1
        42
    end
    @test ret == 42
    @test Array(a) == ones(Float32, 4)

    # work on the stream completes before the profiler is resumed and before it is paused,
    # so that launches the stream batches end up on the right side of the boundary
    s = oneAPI.global_stream(context(), device())
    if s.batching && !oneL0.sync_each_submission()
        pending(s) = s.batch !== nothing || !isempty(s.submitted)
        for i in 1:3; @oneapi kern(a); end
        @test pending(s)
        oneAPI.@profile external=true begin
            @test !pending(s)
            for i in 1:3; @oneapi kern(a); end
            @test pending(s)
        end
        @test !pending(s)
    end

    # sessions cannot be nested, and a failing session doesn't leave one behind
    @test_throws "nested" oneAPI.@profile external=true begin
        oneAPI.@profile external=true nothing
    end
    @test_throws "nested" oneAPI.@profile external=true begin
        oneAPI.@bprofile time=0.1 nothing
    end
    @test_throws "body" oneAPI.@profile external=true error("body")
    @test oneAPI.@profile(external=true, 1) == 1

    # only oneAPI.jl's copy of the Level Zero libraries is loaded
    @test_logs oneAPI.check_level_zero_libraries()
end

@testset "macro" begin
    @test_throws "literal boolean" macroexpand(@__MODULE__, :(oneAPI.@profile external=x 1))
    @test_throws "Invalid keyword" macroexpand(@__MODULE__, :(oneAPI.@profile foo=true 1))
    @test_throws "does not support other" macroexpand(@__MODULE__, :(oneAPI.@profile external=true trace=true 1))
    @test_throws "not supported by oneAPI.@bprofile" macroexpand(@__MODULE__, :(oneAPI.@bprofile external=true 1))
    @test_throws "Invalid keyword" macroexpand(@__MODULE__, :(oneAPI.@bprofile foo=true 1))
end

end
