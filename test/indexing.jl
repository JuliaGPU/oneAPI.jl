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

@testset "CartesianIndices with mapreduce" begin
    # Test for bug fix: mapreduce with CartesianIndices and tuple reduction
    # Previously failed due to SPIR-V codegen issues with nested insertvalue instructions
    # when combining tuples of (bool, CartesianIndex) in reduction operations.
    # The fix involved properly handling nested struct insertions in SPIR-V codegen.

    # Test that we can zip CartesianIndices with array values in a mapreduce
    # This tests the fix for nested tuple operations in SPIR-V codegen

    # Simple test: sum of values while tracking indices
    x = oneArray(ones(Int, 2, 2))
    indices = CartesianIndices((2, 2))

    # Map to tuple of (value, index), then reduce by summing the values
    result = mapreduce(tuple, (t1, t2) -> (t1[1] + t2[1], t1[2]), x, indices;
                       init = (0, CartesianIndex(0, 0)))
    @test result[1] == 4  # sum of four 1s

    # Test with 1D array
    y = oneArray(ones(Int, 4))
    indices_1d = CartesianIndices((4,))
    result_1d = mapreduce(tuple, (t1, t2) -> (t1[1] + t2[1], t1[2]), y, indices_1d;
                          init = (0, CartesianIndex(0,)))
    @test result_1d[1] == 4

    # Test with boolean array and index comparison (closer to original failure case)
    # This pattern is similar to what findfirst would use internally
    z = oneArray([false, true, false, true])
    indices_z = CartesianIndices((4,))
    result_z = mapreduce(tuple,
                         (t1, t2) -> begin
                             (found1, idx1), (found2, idx2) = t1, t2
                             # Return the first found index (smallest index if both found)
                             if found1
                                 return (found1, idx1)
                             else
                                 return (found2, idx2)
                             end
                         end,
                         z, indices_z;
                         init = (false, CartesianIndex(0,)))
    @test result_z[1] == true  # Found a true value
    @test result_z[2] == CartesianIndex(2,)  # First true is at index 2
end
