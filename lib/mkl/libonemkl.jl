using CEnum
using oneAPI.SYCL: syclQueue_t, syclContext_t, syclDevice_t

@cenum onemklTranspose::UInt32 begin
    ONEMKL_TRANSPOSE_NONTRANS = 0
    ONEMKL_TRANSPOSE_TRANS = 1
    ONEMLK_TRANSPOSE_CONJTRANS = 2
end

function onemklSgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklSgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::Cfloat, A::ZePtr{Cfloat}, lda::Int64,
                                    B::ZePtr{Cfloat}, ldb::Int64, beta::Cfloat,
                                    C::ZePtr{Cfloat}, ldc::Int64)::Cint
end

function onemklDgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklDgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::Cdouble, A::ZePtr{Cdouble}, lda::Int64,
                                    B::ZePtr{Cdouble}, ldb::Int64, beta::Cdouble,
                                    C::ZePtr{Cdouble}, ldc::Int64)::Cint
end

function onemklCgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklCgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::ComplexF32, A::ZePtr{ComplexF32}, lda::Int64,
                                    B::ZePtr{ComplexF32}, ldb::Int64, beta::ComplexF32,
                                    C::ZePtr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklZgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                    transB::onemklTranspose, m::Int64, n::Int64, k::Int64,
                                    alpha::ComplexF64, A::ZePtr{ComplexF64}, lda::Int64,
                                    B::ZePtr{ComplexF64}, ldb::Int64, beta::ComplexF64,
                                    C::ZePtr{ComplexF64}, ldc::Int64)::Cint
end

function onemklDnrm2(device_queue, ctx, dev, n, x, incx, result)
	@ccall liboneapi_support.onemklDnrm2(device_queue::syclQueue_t, ctx::syclContext_t, dev::syclDevice_t, n::Int64, x::ZePtr{Cdouble}, incx::Int64, result::RefOrZeRef{Cdouble})::Cvoid
end

function onemklSnrm2(device_queue, ctx, dev, n, x, incx, result)
	@ccall liboneapi_support.onemklSnrm2(device_queue::syclQueue_t, ctx::syclContext_t, dev::syclDevice_t, n::Int64, x::ZePtr{Cfloat}, incx::Int64, result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklCnrm2(device_queue, ctx, dev, n, x, incx, result)
	@ccall liboneapi_support.onemklCnrm2(device_queue::syclQueue_t, ctx::syclContext_t, dev::syclDevice_t, n::Int64, x::ZePtr{ComplexF32}, incx::Int64, result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklZnrm2(device_queue, ctx, dev, n, x, incx, result)
	@ccall liboneapi_support.onemklZnrm2(device_queue::syclQueue_t, ctx::syclContext_t, dev::syclDevice_t, n::Int64, x::ZePtr{ComplexF64}, incx::Int64, result::RefOrZeRef{Cdouble})::Cvoid
end


