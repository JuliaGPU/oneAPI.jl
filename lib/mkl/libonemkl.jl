using CEnum

@cenum onemklTranspose::UInt32 begin
    ONEMKL_TRANSPOSE_NONTRANS = 0
    ONEMKL_TRANSPOSE_TRANS = 1
    ONEMLK_TRANSPOSE_CONJTRANS = 2
end

function onemklSgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    ccall((:onemklSgemm, liboneapilib), Cint,
          (syclQueue_t, onemklTranspose, onemklTranspose, Int64, Int64, Int64, Cfloat,
           ZePtr{Cfloat}, Int64, ZePtr{Cfloat}, Int64, Cfloat, ZePtr{Cfloat}, Int64),
          device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function onemklDgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    ccall((:onemklDgemm, liboneapilib), Cint,
          (syclQueue_t, onemklTranspose, onemklTranspose, Int64, Int64, Int64, Cdouble,
           ZePtr{Cdouble}, Int64, ZePtr{Cdouble}, Int64, Cdouble, ZePtr{Cdouble}, Int64),
          device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function onemklCgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    ccall((:onemklCgemm, liboneapilib), Cint,
          (syclQueue_t, onemklTranspose, onemklTranspose, Int64, Int64, Int64, ComplexF32,
           ZePtr{ComplexF32}, Int64, ZePtr{ComplexF32}, Int64, ComplexF32, ZePtr{ComplexF32},
           Int64),
          device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function onemklZgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    ccall((:onemklZgemm, liboneapilib), Cint,
          (syclQueue_t, onemklTranspose, onemklTranspose, Int64, Int64, Int64, ComplexF32,
           ZePtr{ComplexF32}, Int64, ZePtr{ComplexF32}, Int64, ComplexF32, ZePtr{ComplexF32},
           Int64),
          device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
