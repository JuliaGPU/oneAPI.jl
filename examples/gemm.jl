using oneAPI, Test

A = oneArray(rand(Float32, 2, 3))

B = oneArray(rand(Float32, 3, 4))

C = A * B

@test Array(C) ≈ Array(A) * Array(B)

println("Done")
