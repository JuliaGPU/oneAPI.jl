using CEnum

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

function onemklSaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklSaxpy(device_queue::syclQueue_t, n::Int64, alpha::Cfloat, x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat}, incy::Int64)::Cvoid
end

function onemklDaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklDaxpy(device_queue::syclQueue_t, n::Int64, alpha::Cdouble, x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble}, incy::Int64)::Cvoid
end

function onemklCaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklCaxpy(device_queue::syclQueue_t, n::Int64, alpha::ComplexF32, x::ZePtr{ComplexF32}, incx::Int64, y::ZePtr{ComplexF32}, incy::Int64)::Cvoid
end

function onemklZaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklZaxpy(device_queue::syclQueue_t, n::Int64, alpha::ComplexF64, x::ZePtr{ComplexF64}, incx::Int64, y::ZePtr{ComplexF64}, incy::Int64)::Cvoid
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
