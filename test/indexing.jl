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

    # Test with array larger than 1024 to trigger multiple groups
    large_size = 2048
    large_mask = oneArray(rand(Bool, large_size))
    large_result_gpu = Array(findall(large_mask))
    large_result_cpu = findall(Array(large_mask))
    @test large_result_gpu == large_result_cpu

    # Test with even larger array to ensure robustness
    very_large_size = 5000
    very_large_mask = oneArray(fill(true, very_large_size))  # all true for predictable result
    very_large_result_gpu = Array(findall(very_large_mask))
    very_large_result_cpu = findall(fill(true, very_large_size))
    @test very_large_result_gpu == very_large_result_cpu
end
