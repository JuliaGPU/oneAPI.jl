using Test
using oneAPI

@testset "sorting" begin
    data = oneArray([3, 1, 4, 1, 5])
    sort!(data)
    @test Array(data) == [1, 1, 3, 4, 5]

    data_rev = oneArray([3, 1, 4, 1, 5])
    sort!(data_rev, rev = true)
    @test Array(data_rev) == [5, 4, 3, 1, 1]
    data = oneArray([3, 1, 4, 1, 5])
    @test Array(sortperm(data)) == sortperm([3, 1, 4, 1, 5])

    data_rev = oneArray([3, 1, 4, 1, 5])
    @test Array(sortperm(data_rev, rev = true)) == sortperm([3, 1, 4, 1, 5], rev = true)
end
