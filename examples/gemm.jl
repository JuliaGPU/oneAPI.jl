using oneAPI, Test


drv = driver()
dev = device()
ctx = context()
queue = global_queue(ctx, dev)
sycl_q = sycl_queue(queue)


A = oneArray(rand(Float32, 2, 3))

B = oneArray(rand(Float32, 3, 4))

C = oneAPI.zeros(Float32, size(A, 1), size(B, 2))

oneMKL.onemklSgemm(sycl_q, oneMKL.ONEMKL_TRANSPOSE_NONTRANS, oneMKL.ONEMKL_TRANSPOSE_NONTRANS,
                   size(A, 1), size(B, 2), size(A, 2),
                   1f0,
                   A, stride(A, 2),
                   B, stride(B, 2),
                   0f0,
                   C, stride(C, 2))

@test Array(C) ≈ Array(A) * Array(B)

println("Done")
