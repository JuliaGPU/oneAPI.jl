using Test
using oneAPI

@testset "findall" begin
    bools1d = oneArray([true, false, true, false, true])
    @test Array(findall(bools1d)) == findall(Bool[true, false, true, false, true])

    bools2d = oneArray(Bool[true false; false true; true false])
    @test Array(findall(bools2d)) == findall(Bool[true false; false true; true false])

    all_false = oneArray(fill(false, 4))
    @test Array(findall(all_false)) == Int[]

    all_true = oneArray(fill(true, 3, 2))
    @test Array(findall(all_true)) == findall(fill(true, 3, 2))

    data = oneArray(collect(1:6))
    mask = oneArray(Bool[true, false, true, false, false, true])
    @test Array(data[mask]) == collect(1:6)[findall(Bool[true, false, true, false, false, true])]
end
