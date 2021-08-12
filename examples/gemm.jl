using oneAPI, Test



A = oneArray(rand(Float32, 2, 3))

B = oneArray(rand(Float32, 3, 4))

C = oneAPI.zeros(Float32, size(A, 1), size(B, 2))

oneMKL.gemm!('N', 'N', 1f0, A, B, 0f0, C)

@test Array(C) ≈ Array(A) * Array(B)

println("Done")
