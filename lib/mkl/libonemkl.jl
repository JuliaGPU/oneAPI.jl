using CEnum

@cenum onemklTranspose::UInt32 begin
    ONEMKL_TRANSPOSE_NONTRANS = 0
    ONEMKL_TRANSPOSE_TRANS = 1
    ONEMLK_TRANSPOSE_CONJTRANS = 2
end

@cenum onemklUplo::UInt32 begin
    ONEMKL_UPLO_UPPER = 0
    ONEMKL_UPLO_LOWER = 1
end

@cenum onemklDiag::UInt32 begin
    ONEMKL_DIAG_NONUNIT = 0
    ONEMKL_DIAG_UNIT = 1
end

function onemklSgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSgbmv(device_queue::syclQueue_t, trans::onemklTranspose, 
                                        m::Int64, n::Int64, kl::Int64, ku::Int64, alpha::Cfloat,
                                        a::ZePtr{Cfloat}, lda::Int64, x::ZePtr{Cfloat}, incx::Int64,
                                        beta::Cfloat, y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklDgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDgbmv(device_queue::syclQueue_t, trans::onemklTranspose, 
                                        m::Int64, n::Int64, kl::Int64, ku::Int64, alpha::Cdouble,
                                        a::ZePtr{Cdouble}, lda::Int64, x::ZePtr{Cdouble}, incx::Int64,
                                        beta::Cdouble, y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklCgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklCgbmv(device_queue::syclQueue_t, trans::onemklTranspose, 
                                        m::Int64, n::Int64, kl::Int64, ku::Int64, alpha::ComplexF32,
                                        a::ZePtr{ComplexF32}, lda::Int64, x::ZePtr{ComplexF32}, incx::Int64,
                                        beta::ComplexF32, y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklZgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZgbmv(device_queue::syclQueue_t, trans::onemklTranspose, 
                                        m::Int64, n::Int64, kl::Int64, ku::Int64, alpha::ComplexF64,
                                        a::ZePtr{ComplexF64}, lda::Int64, x::ZePtr{ComplexF64}, incx::Int64,
                                        beta::ComplexF64, y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
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

function onemklSdot(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklSdot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat},
                                        incy::Int64, result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklDdot(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklDdot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble},
                                        incy::Int64, result::RefOrZeRef{Cdouble})::Cvoid
end

function onemklCdotc(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklCdotc(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF32}, incx::Int64, y::ZePtr{ComplexF32},
                                        incy::Int64, result::RefOrZeRef{ComplexF32})::Cvoid
end

function onemklZdotc(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklZdotc(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF64}, incx::Int64, y::ZePtr{ComplexF64},
                                        incy::Int64, result::RefOrZeRef{ComplexF64})::Cvoid
end

function onemklCdotu(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklCdotu(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF32}, incx::Int64, y::ZePtr{ComplexF32},
                                        incy::Int64, result::RefOrZeRef{ComplexF32})::Cvoid
end

function onemklZdotu(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklZdotu(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF64}, incx::Int64, y::ZePtr{ComplexF64},
                                        incy::Int64, result::RefOrZeRef{ComplexF64})::Cvoid
end

function onemklSasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSasum(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cfloat}, incx::Int64,
                                        result::ZePtr{Cfloat})::Cvoid
end

function onemklDasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDasum(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cdouble}, incx::Int64,
                                        result::ZePtr{Cdouble})::Cvoid
end

function onemklCasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCasum(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF32}, incx::Int64,
                                        result::ZePtr{Cfloat})::Cvoid
end

function onemklZasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZasum(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF64}, incx::Int64,
                                        result::ZePtr{Cdouble})::Cvoid
end

function onemklSaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklSaxpy(device_queue::syclQueue_t, n::Int64, alpha::Cfloat,
                        x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklDaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklDaxpy(device_queue::syclQueue_t, n::Int64, alpha::Cdouble,
                        x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklCaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklCaxpy(device_queue::syclQueue_t, n::Int64, alpha::ComplexF32,
                        x::ZePtr{ComplexF32}, incx::Int64, y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklZaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklZaxpy(device_queue::syclQueue_t, n::Int64, alpha::ComplexF64,
                        x::ZePtr{ComplexF64}, incx::Int64, y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
end

function onemklDscal(device_queue, n, alpha, x, incx)
	@ccall liboneapi_support.onemklDscal(device_queue::syclQueue_t, n::Int64, 
                                        alpha::Cdouble, x::ZePtr{Cdouble}, incx::Int64)::Cvoid
end

function onemklSscal(device_queue, n, alpha, x, incx)
	@ccall liboneapi_support.onemklSscal(device_queue::syclQueue_t, n::Int64, 
                                        alpha::Cfloat, x::ZePtr{Cfloat}, incx::Int64)::Cvoid
end

function onemklZscal(device_queue, n, alpha, x, incx)
	@ccall liboneapi_support.onemklZscal(device_queue::syclQueue_t, n::Int64, 
                                        alpha::ComplexF64, x::ZePtr{ComplexF64}, incx::Int64)::Cvoid
end

function onemklZdscal(device_queue, n, alpha, x, incx)
	@ccall liboneapi_support.onemklZdscal(device_queue::syclQueue_t, n::Int64, 
                                        alpha::Cdouble, x::ZePtr{ComplexF64}, incx::Int64)::Cvoid
end

function onemklCscal(device_queue, n, alpha, x, incx)
	@ccall liboneapi_support.onemklCscal(device_queue::syclQueue_t, n::Int64, 
                                        alpha::ComplexF32, x::ZePtr{ComplexF32}, incx::Int64)::Cvoid
end

function onemklCsscal(device_queue, n, alpha, x, incx)
	@ccall liboneapi_support.onemklCsscal(device_queue::syclQueue_t, n::Int64, 
                                        alpha::Cfloat, x::ZePtr{ComplexF32}, incx::Int64)::Cvoid
end

function onemklSgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                        m::Int64, n::Int64, alpha::Cfloat, a::ZePtr{Cfloat},
                                        lda::Int64, x::ZePtr{Cfloat}, incx::Int64, beta::Cfloat,
                                        y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklDgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                        m::Int64, n::Int64, alpha::Cdouble, a::ZePtr{Cdouble},
                                        lda::Int64, x::ZePtr{Cdouble}, incx::Int64, beta::Cdouble,
                                        y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklCgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklCgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                        m::Int64, n::Int64, alpha::ComplexF32, a::ZePtr{ComplexF32},
                                        lda::Int64, x::ZePtr{ComplexF32}, incx::Int64, beta::ComplexF32,
                                        y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklZgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                        m::Int64, n::Int64, alpha::ComplexF64, a::ZePtr{ComplexF64},
                                        lda::Int64, x::ZePtr{ComplexF64}, incx::Int64, beta::ComplexF64,
                                        y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
end

function onemklSger(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklSger(device_queue::syclQueue_t, m::Int64, n::Int64,
                                        alpha::Cfloat, x::ZePtr{Cfloat}, incx::Int64,
                                        y::ZePtr{Cfloat}, incy::Int64, a::ZePtr{Cfloat},
                                        lda::Int64)::Cvoid
end

function onemklDger(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklDger(device_queue::syclQueue_t, m::Int64, n::Int64,
                                        alpha::Cdouble, x::ZePtr{Cdouble}, incx::Int64,
                                        y::ZePtr{Cdouble}, incy::Int64, a::ZePtr{Cdouble},
                                        lda::Int64)::Cvoid
end

function onemklCgerc(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklCgerc(device_queue::syclQueue_t, m::Int64, n::Int64,
                                        alpha::ComplexF32, x::ZePtr{ComplexF32}, incx::Int64,
                                        y::ZePtr{ComplexF32}, incy::Int64, a::ZePtr{ComplexF32},
                                        lda::Int64)::Cvoid
end

function onemklZgerc(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklZgerc(device_queue::syclQueue_t, m::Int64, n::Int64,
                                        alpha::ComplexF64, x::ZePtr{ComplexF64}, incx::Int64,
                                        y::ZePtr{ComplexF64}, incy::Int64, a::ZePtr{ComplexF64},
                                        lda::Int64)::Cvoid
end

function onemklChemv(device_queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklChemv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, alpha::ComplexF32, a::ZePtr{ComplexF32},
                                         lda::Int64, x::ZePtr{ComplexF32}, incx::Int64,
                                         beta::ComplexF32, y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklZhemv(device_queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZhemv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, alpha::ComplexF64, a::ZePtr{ComplexF64},
                                         lda::Int64, x::ZePtr{ComplexF64}, incx::Int64,
                                         beta::ComplexF64, y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
end

function onemklChbmv(device_queue, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklChbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, k::Int64, alpha::ComplexF32, a::ZePtr{ComplexF32},
                                         lda::Int64, x::ZePtr{ComplexF32}, incx::Int64, beta::ComplexF32,
                                         y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklZhbmv(device_queue, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZhbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, k::Int64, alpha::ComplexF64, a::ZePtr{ComplexF64},
                                         lda::Int64, x::ZePtr{ComplexF64}, incx::Int64, beta::ComplexF64,
                                         y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
end

function onemklCher(device_queue, uplo, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklCher(device_queue::syclQueue_t, uplo::onemklUplo,
                                        n::Int64, alpha::ComplexF32, x::ZePtr{ComplexF32},
                                        incx::Int64, a::ZePtr{ComplexF32}, lda::Int64)::Cvoid
end

function onemklZher(device_queue, uplo, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklZher(device_queue::syclQueue_t, uplo::onemklUplo,
                                        n::Int64, alpha::ComplexF64, x::ZePtr{ComplexF64},
                                        incx::Int64, a::ZePtr{ComplexF64}, lda::Int64)::Cvoid
end

function onemklCher2(device_queue, uplo, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklCher2(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, alpha::ComplexF32, x::ZePtr{ComplexF32},
                                         incx::Int64, y::ZePtr{ComplexF32}, incy::Int64,
                                         a::ZePtr{ComplexF32}, lda::Int64)::Cvoid
end

function onemklZher2(device_queue, uplo, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklZher2(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, alpha::ComplexF64, x::ZePtr{ComplexF64},
                                         incx::Int64, y::ZePtr{ComplexF64}, incy::Int64,
                                         a::ZePtr{ComplexF64}, lda::Int64)::Cvoid
end

function onemklSsbmv(device_queue, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSsbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, k::Int64, alpha::Cfloat, a::ZePtr{Cfloat},
                                         lda::Int64, x::ZePtr{Cfloat}, incx::Int64, beta::Cfloat,
                                         y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklDsbmv(device_queue, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDsbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, k::Int64, alpha::Cdouble, a::ZePtr{Cdouble},
                                         lda::Int64, x::ZePtr{Cdouble}, incx::Int64, beta::Cdouble,
                                         y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklSsymv(device_queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSsymv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, alpha::Cfloat, a::ZePtr{Cfloat},
                                         lda::Int64, x::ZePtr{Cfloat}, incx::Int64, beta::Cfloat,
                                         y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklDsymv(device_queue, uplo, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDsymv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         n::Int64, alpha::Cdouble, a::ZePtr{Cdouble},
                                         lda::Int64, x::ZePtr{Cdouble}, incx::Int64, beta::Cdouble,
                                         y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklSsyr(device_queue, uplo, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklSsyr(device_queue::syclQueue_t, uplo::onemklUplo,
                                        n::Int64, alpha::Cfloat, x::ZePtr{Cfloat}, 
                                        incx::Int64, a::ZePtr{Cfloat}, lda::Int64)::Cvoid
end

function onemklDsyr(device_queue, uplo, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklDsyr(device_queue::syclQueue_t, uplo::onemklUplo,
                                        n::Int64, alpha::Cdouble, x::ZePtr{Cdouble}, 
                                        incx::Int64, a::ZePtr{Cdouble}, lda::Int64)::Cvoid
end

function onemklStbmv(device_queue, uplo, trans, diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklStbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         k::Int64, a::ZePtr{Cfloat}, lda::Int64, x::ZePtr{Cfloat},
                                         incx::Int64)::Cvoid
end

function onemklDtbmv(device_queue, uplo, trans, diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         k::Int64, a::ZePtr{Cdouble}, lda::Int64, x::ZePtr{Cdouble},
                                         incx::Int64)::Cvoid
end

function onemklCtbmv(device_queue, uplo, trans, diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         k::Int64, a::ZePtr{ComplexF32}, lda::Int64, x::ZePtr{ComplexF32},
                                         incx::Int64)::Cvoid
end

function onemklZtbmv(device_queue, uplo, trans, diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtbmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         k::Int64, a::ZePtr{ComplexF64}, lda::Int64, x::ZePtr{ComplexF64},
                                         incx::Int64)::Cvoid
end

function onemklStrmv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklStrmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{Cfloat}, lda::Int64, x::ZePtr{Cfloat}, incx::Int64)::Cvoid
end

function onemklDtrmv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtrmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{Cdouble}, lda::Int64, x::ZePtr{Cdouble}, incx::Int64)::Cvoid
end

function onemklCtrmv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtrmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{ComplexF32}, lda::Int64, x::ZePtr{ComplexF32}, incx::Int64)::Cvoid
end

function onemklZtrmv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtrmv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{ComplexF64}, lda::Int64, x::ZePtr{ComplexF64}, incx::Int64)::Cvoid
end

function onemklStrsv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklStrsv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{Cfloat}, lda::Int64, x::ZePtr{Cfloat}, incx::Int64)::Cvoid
end

function onemklDtrsv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtrsv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{Cdouble}, lda::Int64, x::ZePtr{Cdouble}, incx::Int64)::Cvoid
end

function onemklCtrsv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtrsv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{ComplexF32}, lda::Int64, x::ZePtr{ComplexF32}, incx::Int64)::Cvoid
end

function onemklZtrsv(device_queue, uplo, trans, diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtrsv(device_queue::syclQueue_t, uplo::onemklUplo,
                                         trans::onemklTranspose, diag::onemklDiag, n::Int64,
                                         a::ZePtr{ComplexF64}, lda::Int64, x::ZePtr{ComplexF64}, incx::Int64)::Cvoid
end

function onemklDnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklDnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{Cdouble}, incx::Int64, 
                                result::RefOrZeRef{Cdouble})::Cvoid
end

function onemklSnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklSnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{Cfloat}, incx::Int64, 
                                result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklCnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklCnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{ComplexF32}, incx::Int64, 
                                result::RefOrZeRef{Cfloat})::Cvoid
end

function onemklZnrm2(device_queue, n, x, incx, result)
	@ccall liboneapi_support.onemklZnrm2(device_queue::syclQueue_t, 
                                n::Int64, x::ZePtr{ComplexF64}, incx::Int64, 
                                result::RefOrZeRef{Cdouble})::Cvoid
end


function onemklDcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklDcopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{Cdouble}, incx::Int64,
                                y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklScopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklScopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{Cfloat}, incx::Int64,
                                y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklZcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklZcopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{ComplexF64}, incx::Int64,
                                y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
end

function onemklCcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklCcopy(device_queue::syclQueue_t, n::Int64, 
                                x::ZePtr{ComplexF32}, incx::Int64,
                                y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklSamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cfloat}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklDamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cdouble}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklCamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF32}, incx::Int64,result::ZePtr{Int64})::Cvoid
end

function onemklZamax(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZamax(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF64}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklSamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cfloat}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklDamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{Cdouble}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklCamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF32}, incx::Int64,result::ZePtr{Int64})::Cvoid
end

function onemklZamin(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZamin(device_queue::syclQueue_t, n::Int64,
                             x::ZePtr{ComplexF64}, incx::Int64, result::ZePtr{Int64})::Cvoid
end

function onemklSswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklSswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{Cfloat}, incx::Cint,
                                    y::ZePtr{Cfloat}, incy::Cint)::Cvoid
end

function onemklDswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklDswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{Cdouble}, incx::Cint,
                                    y::ZePtr{Cdouble}, incy::Cint)::Cvoid
end

function onemklCswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklCswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{ComplexF32}, incx::Cint,
                                    y::ZePtr{ComplexF32}, incy::Cint)::Cvoid
end

function onemklZswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklZswap(device_queue::syclQueue_t, n::Cint,
                                    x::ZePtr{ComplexF64}, incx::Cint,
                                    y::ZePtr{ComplexF64}, incy::Cint)::Cvoid
end