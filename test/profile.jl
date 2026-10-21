@testset "profile" begin

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
        kern(a) = (a[1] += 1f0; return)
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
    @test_throws "body" oneAPI.@profile external=true error("body")
    @test oneAPI.@profile(external=true, 1) == 1

    # only oneAPI.jl's copy of the Level Zero libraries is loaded
    @test_logs oneAPI.check_level_zero_libraries()
end

@testset "macro" begin
    @test_throws "integrated profiling" macroexpand(@__MODULE__, :(oneAPI.@profile 1))
    @test_throws "integrated profiling" macroexpand(@__MODULE__, :(oneAPI.@profile external=false 1))
    @test_throws "literal boolean" macroexpand(@__MODULE__, :(oneAPI.@profile external=x 1))
    @test_throws "Invalid keyword" macroexpand(@__MODULE__, :(oneAPI.@profile trace=true 1))
end

end
