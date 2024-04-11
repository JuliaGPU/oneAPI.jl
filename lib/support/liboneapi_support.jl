using CEnum

mutable struct syclPlatform_st end

const syclPlatform_t = Ptr{syclPlatform_st}

function syclPlatformCreate(obj, driver)
    @ccall liboneapi_support.syclPlatformCreate(obj::Ptr{syclPlatform_t},
                                                driver::ze_driver_handle_t)::Cint
end

function syclPlatformDestroy(obj)
    @ccall liboneapi_support.syclPlatformDestroy(obj::syclPlatform_t)::Cint
end

mutable struct syclDevice_st end

const syclDevice_t = Ptr{syclDevice_st}

function syclDeviceCreate(obj, platform, device)
    @ccall liboneapi_support.syclDeviceCreate(obj::Ptr{syclDevice_t},
                                              platform::syclPlatform_t,
                                              device::ze_device_handle_t)::Cint
end

function syclDeviceDestroy(obj)
    @ccall liboneapi_support.syclDeviceDestroy(obj::syclDevice_t)::Cint
end

mutable struct syclContext_st end

const syclContext_t = Ptr{syclContext_st}

function syclContextCreate(obj, devices, ndevices, context, keep_ownership)
    @ccall liboneapi_support.syclContextCreate(obj::Ptr{syclContext_t},
                                               devices::Ptr{syclDevice_t},
                                               ndevices::Csize_t,
                                               context::ze_context_handle_t,
                                               keep_ownership::Cint)::Cint
end

function syclContextDestroy(obj)
    @ccall liboneapi_support.syclContextDestroy(obj::syclContext_t)::Cint
end

mutable struct syclQueue_st end

const syclQueue_t = Ptr{syclQueue_st}

function syclQueueCreate(obj, context, device, queue, keep_ownership)
    @ccall liboneapi_support.syclQueueCreate(obj::Ptr{syclQueue_t}, context::syclContext_t,
                                             device::syclDevice_t,
                                             queue::ze_command_queue_handle_t,
                                             keep_ownership::Cint)::Cint
end

function syclQueueDestroy(obj)
    @ccall liboneapi_support.syclQueueDestroy(obj::syclQueue_t)::Cint
end

mutable struct syclEvent_st end

const syclEvent_t = Ptr{syclEvent_st}

function syclEventCreate(obj, context, event, keep_ownership)
    @ccall liboneapi_support.syclEventCreate(obj::Ptr{syclEvent_t}, context::syclContext_t,
                                             event::ze_event_handle_t,
                                             keep_ownership::Cint)::Cint
end

function syclEventDestroy(obj)
    @ccall liboneapi_support.syclEventDestroy(obj::syclEvent_t)::Cint
end

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

@cenum onemklSide::UInt32 begin
    ONEMKL_SIDE_LEFT = 0
    ONEMKL_SIDE_RIGHT = 1
end

@cenum onemklOffset::UInt32 begin
    ONEMKL_OFFSET_ROW = 0
    ONEMKL_OFFSET_COL = 1
    ONEMKL_OFFSET_FIX = 2
end

@cenum onemklJob::UInt32 begin
    ONEMKL_JOB_N = 0
    ONEMKL_JOB_V = 1
    ONEMKL_JOB_U = 2
    ONEMKL_JOB_A = 3
    ONEMKL_JOB_S = 4
    ONEMKL_JOB_O = 5
end

@cenum onemklGenerate::UInt32 begin
    ONEMKL_GENERATE_Q = 0
    ONEMKL_GENERATE_P = 1
    ONEMKL_GENERATE_N = 2
    ONEMKL_GENERATE_V = 3
end

@cenum onemklCompz::UInt32 begin
    ONEMKL_COMPZ_N = 0
    ONEMKL_COMPZ_V = 1
    ONEMKL_COMPZ_I = 2
end

@cenum onemklDirect::UInt32 begin
    ONEMKL_DIRECT_F = 0
    ONEMKL_DIRECT_B = 1
end

@cenum onemklStorev::UInt32 begin
    ONEMKL_STOREV_C = 0
    ONEMKL_STOREV_R = 1
end

@cenum onemklRangev::UInt32 begin
    ONEMKL_RANGEV_A = 0
    ONEMKL_RANGEV_V = 1
    ONEMKL_RANGEV_I = 2
end

@cenum onemklOrder::UInt32 begin
    ONEMKL_ORDER_B = 0
    ONEMKL_ORDER_E = 1
end

@cenum onemklJobsvd::UInt32 begin
    ONEMKL_JOBSVD_N = 0
    ONEMKL_JOBSVD_A = 1
    ONEMKL_JOBSVD_O = 2
    ONEMKL_JOBSVD_S = 3
end

@cenum onemklLayout::UInt32 begin
    ONEMKL_LAYOUT_ROW = 0
    ONEMKL_LAYOUT_COL = 1
end

@cenum onemklIndex::UInt32 begin
    ONEMKL_INDEX_ZERO = 0
    ONEMKL_INDEX_ONE = 1
end

@cenum onemklProperty::UInt32 begin
    ONEMKL_PROPERTY_SYMMETRIC = 0
    ONEMKL_PROPERTY_SORTED = 1
end

@cenum onemklMatrixView::UInt32 begin
    ONEMKL_MATRIX_VIEW_GENERAL = 0
end

@cenum onemklMatmatRequest::UInt32 begin
    ONEMKL_MATMAT_REQUEST_GET_WORK_ESTIMATION_BUF_SIZE = 0
    ONEMKL_MATMAT_REQUEST_WORK_ESTIMATION = 1
    ONEMKL_MATMAT_REQUEST_GET_COMPUTE_STRUCTURE_BUF_SIZE = 2
    ONEMKL_MATMAT_REQUEST_COMPUTE_STRUCTURE = 3
    ONEMKL_MATMAT_REQUEST_FINALIZE_STRUCTURE = 4
    ONEMKL_MATMAT_REQUEST_GET_COMPUTE_BUF_SIZE = 5
    ONEMKL_MATMAT_REQUEST_COMPUTE = 6
    ONEMKL_MATMAT_REQUEST_GET_NNZ = 7
    ONEMKL_MATMAT_REQUEST_FINALIZE = 8
end

mutable struct matrix_handle end

const matrix_handle_t = Ptr{matrix_handle}

mutable struct matmat_descr end

const matmat_descr_t = Ptr{matmat_descr}

function onemklHgemmBatched(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size)
    @ccall liboneapi_support.onemklHgemmBatched(device_queue::syclQueue_t,
                                                transa::onemklTranspose,
                                                transb::onemklTranspose, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, k::ZePtr{Int64},
                                                alpha::ZePtr{Float16},
                                                a::ZePtr{Ptr{Float16}}, lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{Float16}}, ldb::ZePtr{Int64},
                                                beta::ZePtr{Float16},
                                                c::ZePtr{Ptr{Float16}}, ldc::ZePtr{Int64},
                                                group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklSgemmBatched(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size)
    @ccall liboneapi_support.onemklSgemmBatched(device_queue::syclQueue_t,
                                                transa::onemklTranspose,
                                                transb::onemklTranspose, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, k::ZePtr{Int64},
                                                alpha::ZePtr{Cfloat}, a::ZePtr{Ptr{Cfloat}},
                                                lda::ZePtr{Int64}, b::ZePtr{Ptr{Cfloat}},
                                                ldb::ZePtr{Int64}, beta::ZePtr{Cfloat},
                                                c::ZePtr{Ptr{Cfloat}}, ldc::ZePtr{Int64},
                                                group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklDgemmBatched(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size)
    @ccall liboneapi_support.onemklDgemmBatched(device_queue::syclQueue_t,
                                                transa::onemklTranspose,
                                                transb::onemklTranspose, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, k::ZePtr{Int64},
                                                alpha::ZePtr{Cdouble},
                                                a::ZePtr{Ptr{Cdouble}}, lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{Cdouble}}, ldb::ZePtr{Int64},
                                                beta::ZePtr{Cdouble},
                                                c::ZePtr{Ptr{Cdouble}}, ldc::ZePtr{Int64},
                                                group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklCgemmBatched(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size)
    @ccall liboneapi_support.onemklCgemmBatched(device_queue::syclQueue_t,
                                                transa::onemklTranspose,
                                                transb::onemklTranspose, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, k::ZePtr{Int64},
                                                alpha::ZePtr{ComplexF32},
                                                a::ZePtr{Ptr{ComplexF32}},
                                                lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{ComplexF32}},
                                                ldb::ZePtr{Int64}, beta::ZePtr{ComplexF32},
                                                c::ZePtr{Ptr{ComplexF32}},
                                                ldc::ZePtr{Int64}, group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklZgemmBatched(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                            beta, c, ldc, group_count, group_size)
    @ccall liboneapi_support.onemklZgemmBatched(device_queue::syclQueue_t,
                                                transa::onemklTranspose,
                                                transb::onemklTranspose, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, k::ZePtr{Int64},
                                                alpha::ZePtr{ComplexF64},
                                                a::ZePtr{Ptr{ComplexF64}},
                                                lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{ComplexF64}},
                                                ldb::ZePtr{Int64}, beta::ZePtr{ComplexF64},
                                                c::ZePtr{Ptr{ComplexF64}},
                                                ldc::ZePtr{Int64}, group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklStrsmBatched(device_queue, left_right, upper_lower, transa, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size)
    @ccall liboneapi_support.onemklStrsmBatched(device_queue::syclQueue_t,
                                                left_right::onemklSide,
                                                upper_lower::onemklUplo,
                                                transa::onemklTranspose,
                                                unit_diag::onemklDiag, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, alpha::ZePtr{Cfloat},
                                                a::ZePtr{Ptr{Cfloat}}, lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{Cfloat}}, ldb::ZePtr{Int64},
                                                group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklDtrsmBatched(device_queue, left_right, upper_lower, transa, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size)
    @ccall liboneapi_support.onemklDtrsmBatched(device_queue::syclQueue_t,
                                                left_right::onemklSide,
                                                upper_lower::onemklUplo,
                                                transa::onemklTranspose,
                                                unit_diag::onemklDiag, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, alpha::ZePtr{Cdouble},
                                                a::ZePtr{Ptr{Cdouble}}, lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{Cdouble}}, ldb::ZePtr{Int64},
                                                group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklCtrsmBatched(device_queue, left_right, upper_lower, transa, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size)
    @ccall liboneapi_support.onemklCtrsmBatched(device_queue::syclQueue_t,
                                                left_right::onemklSide,
                                                upper_lower::onemklUplo,
                                                transa::onemklTranspose,
                                                unit_diag::onemklDiag, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, alpha::ZePtr{ComplexF32},
                                                a::ZePtr{Ptr{ComplexF32}},
                                                lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{ComplexF32}},
                                                ldb::ZePtr{Int64}, group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklZtrsmBatched(device_queue, left_right, upper_lower, transa, unit_diag, m, n,
                            alpha, a, lda, b, ldb, group_count, group_size)
    @ccall liboneapi_support.onemklZtrsmBatched(device_queue::syclQueue_t,
                                                left_right::onemklSide,
                                                upper_lower::onemklUplo,
                                                transa::onemklTranspose,
                                                unit_diag::onemklDiag, m::ZePtr{Int64},
                                                n::ZePtr{Int64}, alpha::ZePtr{ComplexF64},
                                                a::ZePtr{Ptr{ComplexF64}},
                                                lda::ZePtr{Int64},
                                                b::ZePtr{Ptr{ComplexF64}},
                                                ldb::ZePtr{Int64}, group_count::Int64,
                                                group_size::ZePtr{Int64})::Cint
end

function onemklHgemmBatchStrided(device_queue, transa, transb, m, n, k, alpha, a, lda,
                                 stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                 batch_size)
    @ccall liboneapi_support.onemklHgemmBatchStrided(device_queue::syclQueue_t,
                                                     transa::onemklTranspose,
                                                     transb::onemklTranspose, m::Int64,
                                                     n::Int64, k::Int64,
                                                     alpha::Ref{Float16}, a::ZePtr{Float16},
                                                     lda::Int64, stridea::Int64,
                                                     b::ZePtr{Float16}, ldb::Int64,
                                                     strideb::Int64, beta::Ref{Float16},
                                                     c::ZePtr{Float16}, ldc::Int64,
                                                     stridec::Int64,
                                                     batch_size::Int64)::Cint
end

function onemklSgemmBatchStrided(device_queue, transa, transb, m, n, k, alpha, a, lda,
                                 stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                 batch_size)
    @ccall liboneapi_support.onemklSgemmBatchStrided(device_queue::syclQueue_t,
                                                     transa::onemklTranspose,
                                                     transb::onemklTranspose, m::Int64,
                                                     n::Int64, k::Int64, alpha::Ref{Cfloat},
                                                     a::ZePtr{Cfloat}, lda::Int64,
                                                     stridea::Int64, b::ZePtr{Cfloat},
                                                     ldb::Int64, strideb::Int64,
                                                     beta::Ref{Cfloat}, c::ZePtr{Cfloat},
                                                     ldc::Int64, stridec::Int64,
                                                     batch_size::Int64)::Cint
end

function onemklDgemmBatchStrided(device_queue, transa, transb, m, n, k, alpha, a, lda,
                                 stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                 batch_size)
    @ccall liboneapi_support.onemklDgemmBatchStrided(device_queue::syclQueue_t,
                                                     transa::onemklTranspose,
                                                     transb::onemklTranspose, m::Int64,
                                                     n::Int64, k::Int64,
                                                     alpha::Ref{Cdouble}, a::ZePtr{Cdouble},
                                                     lda::Int64, stridea::Int64,
                                                     b::ZePtr{Cdouble}, ldb::Int64,
                                                     strideb::Int64, beta::Ref{Cdouble},
                                                     c::ZePtr{Cdouble}, ldc::Int64,
                                                     stridec::Int64,
                                                     batch_size::Int64)::Cint
end

function onemklCgemmBatchStrided(device_queue, transa, transb, m, n, k, alpha, a, lda,
                                 stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                 batch_size)
    @ccall liboneapi_support.onemklCgemmBatchStrided(device_queue::syclQueue_t,
                                                     transa::onemklTranspose,
                                                     transb::onemklTranspose, m::Int64,
                                                     n::Int64, k::Int64,
                                                     alpha::Ref{ComplexF32},
                                                     a::ZePtr{ComplexF32}, lda::Int64,
                                                     stridea::Int64, b::ZePtr{ComplexF32},
                                                     ldb::Int64, strideb::Int64,
                                                     beta::Ref{ComplexF32},
                                                     c::ZePtr{ComplexF32}, ldc::Int64,
                                                     stridec::Int64,
                                                     batch_size::Int64)::Cint
end

function onemklZgemmBatchStrided(device_queue, transa, transb, m, n, k, alpha, a, lda,
                                 stridea, b, ldb, strideb, beta, c, ldc, stridec,
                                 batch_size)
    @ccall liboneapi_support.onemklZgemmBatchStrided(device_queue::syclQueue_t,
                                                     transa::onemklTranspose,
                                                     transb::onemklTranspose, m::Int64,
                                                     n::Int64, k::Int64,
                                                     alpha::Ref{ComplexF64},
                                                     a::ZePtr{ComplexF64}, lda::Int64,
                                                     stridea::Int64, b::ZePtr{ComplexF64},
                                                     ldb::Int64, strideb::Int64,
                                                     beta::Ref{ComplexF64},
                                                     c::ZePtr{ComplexF64}, ldc::Int64,
                                                     stridec::Int64,
                                                     batch_size::Int64)::Cint
end

function onemklHgemm(device_queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                     ldc)
    @ccall liboneapi_support.onemklHgemm(device_queue::syclQueue_t, transA::onemklTranspose,
                                         transB::onemklTranspose, m::Int64, n::Int64,
                                         k::Int64, alpha::Ref{Float16}, A::ZePtr{Float16},
                                         lda::Int64, B::ZePtr{Float16}, ldb::Int64,
                                         beta::Ref{Float16}, C::ZePtr{Float16},
                                         ldc::Int64)::Cint
end

function onemklHaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklHaxpy(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{Float16}, x::ZePtr{Float16},
                                         incx::Int64, y::ZePtr{Float16}, incy::Int64)::Cint
end

function onemklHscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklHscal(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{Float16}, x::ZePtr{Float16},
                                         incx::Int64)::Cint
end

function onemklHnrm2(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklHnrm2(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Float16}, incx::Int64,
                                         result::RefOrZeRef{Float16})::Cint
end

function onemklHdot(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklHdot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Float16}, incx::Int64, y::ZePtr{Float16},
                                        incy::Int64, result::RefOrZeRef{Float16})::Cint
end

function onemklSgemm(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc)
    @ccall liboneapi_support.onemklSgemm(device_queue::syclQueue_t, transa::onemklTranspose,
                                         transb::onemklTranspose, m::Int64, n::Int64,
                                         k::Int64, alpha::Ref{Cfloat}, a::ZePtr{Cfloat},
                                         lda::Int64, b::ZePtr{Cfloat}, ldb::Int64,
                                         beta::Ref{Cfloat}, c::ZePtr{Cfloat},
                                         ldc::Int64)::Cint
end

function onemklDgemm(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc)
    @ccall liboneapi_support.onemklDgemm(device_queue::syclQueue_t, transa::onemklTranspose,
                                         transb::onemklTranspose, m::Int64, n::Int64,
                                         k::Int64, alpha::Ref{Cdouble}, a::ZePtr{Cdouble},
                                         lda::Int64, b::ZePtr{Cdouble}, ldb::Int64,
                                         beta::Ref{Cdouble}, c::ZePtr{Cdouble},
                                         ldc::Int64)::Cint
end

function onemklCgemm(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc)
    @ccall liboneapi_support.onemklCgemm(device_queue::syclQueue_t, transa::onemklTranspose,
                                         transb::onemklTranspose, m::Int64, n::Int64,
                                         k::Int64, alpha::Ref{ComplexF32},
                                         a::ZePtr{ComplexF32}, lda::Int64,
                                         b::ZePtr{ComplexF32}, ldb::Int64,
                                         beta::Ref{ComplexF32}, c::ZePtr{ComplexF32},
                                         ldc::Int64)::Cint
end

function onemklZgemm(device_queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc)
    @ccall liboneapi_support.onemklZgemm(device_queue::syclQueue_t, transa::onemklTranspose,
                                         transb::onemklTranspose, m::Int64, n::Int64,
                                         k::Int64, alpha::Ref{ComplexF64},
                                         a::ZePtr{ComplexF64}, lda::Int64,
                                         b::ZePtr{ComplexF64}, ldb::Int64,
                                         beta::Ref{ComplexF64}, c::ZePtr{ComplexF64},
                                         ldc::Int64)::Cint
end

function onemklSsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc)
    @ccall liboneapi_support.onemklSsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64,
                                         alpha::Ref{Cfloat}, a::ZePtr{Cfloat}, lda::Int64,
                                         b::ZePtr{Cfloat}, ldb::Int64, beta::Ref{Cfloat},
                                         c::ZePtr{Cfloat}, ldc::Int64)::Cint
end

function onemklDsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc)
    @ccall liboneapi_support.onemklDsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64,
                                         alpha::Ref{Cdouble}, a::ZePtr{Cdouble}, lda::Int64,
                                         b::ZePtr{Cdouble}, ldb::Int64, beta::Ref{Cdouble},
                                         c::ZePtr{Cdouble}, ldc::Int64)::Cint
end

function onemklCsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc)
    @ccall liboneapi_support.onemklCsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, b::ZePtr{ComplexF32}, ldb::Int64,
                                         beta::Ref{ComplexF32}, c::ZePtr{ComplexF32},
                                         ldc::Int64)::Cint
end

function onemklZsymm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc)
    @ccall liboneapi_support.onemklZsymm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, b::ZePtr{ComplexF64}, ldb::Int64,
                                         beta::Ref{ComplexF64}, c::ZePtr{ComplexF64},
                                         ldc::Int64)::Cint
end

function onemklChemm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc)
    @ccall liboneapi_support.onemklChemm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, b::ZePtr{ComplexF32}, ldb::Int64,
                                         beta::Ref{ComplexF32}, c::ZePtr{ComplexF32},
                                         ldc::Int64)::Cint
end

function onemklZhemm(device_queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc)
    @ccall liboneapi_support.onemklZhemm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, b::ZePtr{ComplexF64}, ldb::Int64,
                                         beta::Ref{ComplexF64}, c::ZePtr{ComplexF64},
                                         ldc::Int64)::Cint
end

function onemklSsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklSsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64,
                                         alpha::Ref{Cfloat}, a::ZePtr{Cfloat}, lda::Int64,
                                         beta::Ref{Cfloat}, c::ZePtr{Cfloat},
                                         ldc::Int64)::Cint
end

function onemklDsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklDsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64,
                                         alpha::Ref{Cdouble}, a::ZePtr{Cdouble}, lda::Int64,
                                         beta::Ref{Cdouble}, c::ZePtr{Cdouble},
                                         ldc::Int64)::Cint
end

function onemklCsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklCsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, beta::Ref{ComplexF32},
                                         c::ZePtr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZsyrk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklZsyrk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, beta::Ref{ComplexF64},
                                         c::ZePtr{ComplexF64}, ldc::Int64)::Cint
end

function onemklCherk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklCherk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, beta::Ref{ComplexF32},
                                         c::ZePtr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZherk(device_queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc)
    @ccall liboneapi_support.onemklZherk(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, n::Int64, k::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, beta::Ref{ComplexF64},
                                         c::ZePtr{ComplexF64}, ldc::Int64)::Cint
end

function onemklSsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                      c, ldc)
    @ccall liboneapi_support.onemklSsyr2k(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, trans::onemklTranspose,
                                          n::Int64, k::Int64, alpha::Ref{Cfloat},
                                          a::ZePtr{Cfloat}, lda::Int64, b::ZePtr{Cfloat},
                                          ldb::Int64, beta::Ref{Cfloat}, c::ZePtr{Cfloat},
                                          ldc::Int64)::Cint
end

function onemklDsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                      c, ldc)
    @ccall liboneapi_support.onemklDsyr2k(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, trans::onemklTranspose,
                                          n::Int64, k::Int64, alpha::Ref{Cdouble},
                                          a::ZePtr{Cdouble}, lda::Int64, b::ZePtr{Cdouble},
                                          ldb::Int64, beta::Ref{Cdouble}, c::ZePtr{Cdouble},
                                          ldc::Int64)::Cint
end

function onemklCsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                      c, ldc)
    @ccall liboneapi_support.onemklCsyr2k(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, trans::onemklTranspose,
                                          n::Int64, k::Int64, alpha::Ref{ComplexF32},
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          b::ZePtr{ComplexF32}, ldb::Int64,
                                          beta::Ref{ComplexF32}, c::ZePtr{ComplexF32},
                                          ldc::Int64)::Cint
end

function onemklZsyr2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                      c, ldc)
    @ccall liboneapi_support.onemklZsyr2k(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, trans::onemklTranspose,
                                          n::Int64, k::Int64, alpha::Ref{ComplexF64},
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          b::ZePtr{ComplexF64}, ldb::Int64,
                                          beta::Ref{ComplexF64}, c::ZePtr{ComplexF64},
                                          ldc::Int64)::Cint
end

function onemklCher2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                      c, ldc)
    @ccall liboneapi_support.onemklCher2k(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, trans::onemklTranspose,
                                          n::Int64, k::Int64, alpha::Ref{ComplexF32},
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          b::ZePtr{ComplexF32}, ldb::Int64,
                                          beta::Ref{ComplexF32}, c::ZePtr{ComplexF32},
                                          ldc::Int64)::Cint
end

function onemklZher2k(device_queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta,
                      c, ldc)
    @ccall liboneapi_support.onemklZher2k(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, trans::onemklTranspose,
                                          n::Int64, k::Int64, alpha::Ref{ComplexF64},
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          b::ZePtr{ComplexF64}, ldb::Int64,
                                          beta::Ref{ComplexF64}, c::ZePtr{ComplexF64},
                                          ldc::Int64)::Cint
end

function onemklStrmm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklStrmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{Cfloat}, a::ZePtr{Cfloat}, lda::Int64,
                                         b::ZePtr{Cfloat}, ldb::Int64)::Cint
end

function onemklDtrmm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklDtrmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{Cdouble}, a::ZePtr{Cdouble}, lda::Int64,
                                         b::ZePtr{Cdouble}, ldb::Int64)::Cint
end

function onemklCtrmm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklCtrmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, b::ZePtr{ComplexF32}, ldb::Int64)::Cint
end

function onemklZtrmm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklZtrmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, b::ZePtr{ComplexF64}, ldb::Int64)::Cint
end

function onemklStrsm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklStrsm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{Cfloat}, a::ZePtr{Cfloat}, lda::Int64,
                                         b::ZePtr{Cfloat}, ldb::Int64)::Cint
end

function onemklDtrsm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklDtrsm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{Cdouble}, a::ZePtr{Cdouble}, lda::Int64,
                                         b::ZePtr{Cdouble}, ldb::Int64)::Cint
end

function onemklCtrsm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklCtrsm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, b::ZePtr{ComplexF32}, ldb::Int64)::Cint
end

function onemklZtrsm(device_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb)
    @ccall liboneapi_support.onemklZtrsm(device_queue::syclQueue_t, left_right::onemklSide,
                                         upper_lower::onemklUplo, trans::onemklTranspose,
                                         unit_diag::onemklDiag, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, b::ZePtr{ComplexF64}, ldb::Int64)::Cint
end

function onemklSdgmm(device_queue, left_right, m, n, a, lda, x, incx, c, ldc)
    @ccall liboneapi_support.onemklSdgmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         m::Int64, n::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64, c::ZePtr{Cfloat},
                                         ldc::Int64)::Cint
end

function onemklDdgmm(device_queue, left_right, m, n, a, lda, x, incx, c, ldc)
    @ccall liboneapi_support.onemklDdgmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         m::Int64, n::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64, c::ZePtr{Cdouble},
                                         ldc::Int64)::Cint
end

function onemklCdgmm(device_queue, left_right, m, n, a, lda, x, incx, c, ldc)
    @ccall liboneapi_support.onemklCdgmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         m::Int64, n::Int64, a::ZePtr{ComplexF32},
                                         lda::Int64, x::ZePtr{ComplexF32}, incx::Int64,
                                         c::ZePtr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZdgmm(device_queue, left_right, m, n, a, lda, x, incx, c, ldc)
    @ccall liboneapi_support.onemklZdgmm(device_queue::syclQueue_t, left_right::onemklSide,
                                         m::Int64, n::Int64, a::ZePtr{ComplexF64},
                                         lda::Int64, x::ZePtr{ComplexF64}, incx::Int64,
                                         c::ZePtr{ComplexF64}, ldc::Int64)::Cint
end

function onemklSgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, alpha::Ref{Cfloat},
                                         a::ZePtr{Cfloat}, lda::Int64, x::ZePtr{Cfloat},
                                         incx::Int64, beta::Ref{Cfloat}, y::ZePtr{Cfloat},
                                         incy::Int64)::Cint
end

function onemklDgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, alpha::Ref{Cdouble},
                                         a::ZePtr{Cdouble}, lda::Int64, x::ZePtr{Cdouble},
                                         incx::Int64, beta::Ref{Cdouble}, y::ZePtr{Cdouble},
                                         incy::Int64)::Cint
end

function onemklCgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklCgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, alpha::Ref{ComplexF32},
                                         a::ZePtr{ComplexF32}, lda::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         beta::Ref{ComplexF32}, y::ZePtr{ComplexF32},
                                         incy::Int64)::Cint
end

function onemklZgemv(device_queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZgemv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, alpha::Ref{ComplexF64},
                                         a::ZePtr{ComplexF64}, lda::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         beta::Ref{ComplexF64}, y::ZePtr{ComplexF64},
                                         incy::Int64)::Cint
end

function onemklSgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                     incy)
    @ccall liboneapi_support.onemklSgbmv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, kl::Int64, ku::Int64,
                                         alpha::Ref{Cfloat}, a::ZePtr{Cfloat}, lda::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64, beta::Ref{Cfloat},
                                         y::ZePtr{Cfloat}, incy::Int64)::Cint
end

function onemklDgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                     incy)
    @ccall liboneapi_support.onemklDgbmv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, kl::Int64, ku::Int64,
                                         alpha::Ref{Cdouble}, a::ZePtr{Cdouble}, lda::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64, beta::Ref{Cdouble},
                                         y::ZePtr{Cdouble}, incy::Int64)::Cint
end

function onemklCgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                     incy)
    @ccall liboneapi_support.onemklCgbmv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, kl::Int64, ku::Int64,
                                         alpha::Ref{ComplexF32}, a::ZePtr{ComplexF32},
                                         lda::Int64, x::ZePtr{ComplexF32}, incx::Int64,
                                         beta::Ref{ComplexF32}, y::ZePtr{ComplexF32},
                                         incy::Int64)::Cint
end

function onemklZgbmv(device_queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                     incy)
    @ccall liboneapi_support.onemklZgbmv(device_queue::syclQueue_t, trans::onemklTranspose,
                                         m::Int64, n::Int64, kl::Int64, ku::Int64,
                                         alpha::Ref{ComplexF64}, a::ZePtr{ComplexF64},
                                         lda::Int64, x::ZePtr{ComplexF64}, incx::Int64,
                                         beta::Ref{ComplexF64}, y::ZePtr{ComplexF64},
                                         incy::Int64)::Cint
end

function onemklSger(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklSger(device_queue::syclQueue_t, m::Int64, n::Int64,
                                        alpha::Ref{Cfloat}, x::ZePtr{Cfloat}, incx::Int64,
                                        y::ZePtr{Cfloat}, incy::Int64, a::ZePtr{Cfloat},
                                        lda::Int64)::Cint
end

function onemklDger(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklDger(device_queue::syclQueue_t, m::Int64, n::Int64,
                                        alpha::Ref{Cdouble}, x::ZePtr{Cdouble}, incx::Int64,
                                        y::ZePtr{Cdouble}, incy::Int64, a::ZePtr{Cdouble},
                                        lda::Int64)::Cint
end

function onemklCgerc(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklCgerc(device_queue::syclQueue_t, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF32}, x::ZePtr{ComplexF32},
                                         incx::Int64, y::ZePtr{ComplexF32}, incy::Int64,
                                         a::ZePtr{ComplexF32}, lda::Int64)::Cint
end

function onemklZgerc(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklZgerc(device_queue::syclQueue_t, m::Int64, n::Int64,
                                         alpha::Ref{ComplexF64}, x::ZePtr{ComplexF64},
                                         incx::Int64, y::ZePtr{ComplexF64}, incy::Int64,
                                         a::ZePtr{ComplexF64}, lda::Int64)::Cint
end

function onemklCgeru(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklCgeru(device_queue::syclQueue_t, m::Int64, n::Int64,
                                         alpha::ComplexF32, x::ZePtr{ComplexF32},
                                         incx::Int64, y::ZePtr{ComplexF32}, incy::Int64,
                                         a::ZePtr{ComplexF32}, lda::Int64)::Cint
end

function onemklZgeru(device_queue, m, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklZgeru(device_queue::syclQueue_t, m::Int64, n::Int64,
                                         alpha::ComplexF32, x::ZePtr{ComplexF64},
                                         incx::Int64, y::ZePtr{ComplexF64}, incy::Int64,
                                         a::ZePtr{ComplexF64}, lda::Int64)::Cint
end

function onemklChbmv(device_queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklChbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, k::Int64, alpha::Ref{ComplexF32},
                                         a::ZePtr{ComplexF32}, lda::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         beta::Ref{ComplexF32}, y::ZePtr{ComplexF32},
                                         incy::Int64)::Cint
end

function onemklZhbmv(device_queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZhbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, k::Int64, alpha::Ref{ComplexF64},
                                         a::ZePtr{ComplexF64}, lda::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         beta::Ref{ComplexF64}, y::ZePtr{ComplexF64},
                                         incy::Int64)::Cint
end

function onemklChemv(device_queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklChemv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{ComplexF32},
                                         a::ZePtr{ComplexF32}, lda::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         beta::Ref{ComplexF32}, y::ZePtr{ComplexF32},
                                         incy::Int64)::Cint
end

function onemklZhemv(device_queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZhemv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{ComplexF64},
                                         a::ZePtr{ComplexF64}, lda::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         beta::Ref{ComplexF64}, y::ZePtr{ComplexF64},
                                         incy::Int64)::Cint
end

function onemklCher(device_queue, upper_lower, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklCher(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Ref{ComplexF32},
                                        x::ZePtr{ComplexF32}, incx::Int64,
                                        a::ZePtr{ComplexF32}, lda::Int64)::Cint
end

function onemklZher(device_queue, upper_lower, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklZher(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Ref{ComplexF64},
                                        x::ZePtr{ComplexF64}, incx::Int64,
                                        a::ZePtr{ComplexF64}, lda::Int64)::Cint
end

function onemklCher2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklCher2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{ComplexF32},
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         y::ZePtr{ComplexF32}, incy::Int64,
                                         a::ZePtr{ComplexF32}, lda::Int64)::Cint
end

function onemklZher2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklZher2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{ComplexF64},
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         y::ZePtr{ComplexF64}, incy::Int64,
                                         a::ZePtr{ComplexF64}, lda::Int64)::Cint
end

function onemklChpmv(device_queue, upper_lower, n, alpha, a, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklChpmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::ComplexF32, a::Ptr{ComplexF32},
                                         x::Ptr{ComplexF32}, incx::Int64, beta::ComplexF32,
                                         y::Ptr{ComplexF32}, incy::Int64)::Cint
end

function onemklZhpmv(device_queue, upper_lower, n, alpha, a, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZhpmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::ComplexF32, a::Ptr{ComplexF32},
                                         x::Ptr{ComplexF32}, incx::Int64, beta::ComplexF32,
                                         y::Ptr{ComplexF32}, incy::Int64)::Cint
end

function onemklChpr(device_queue, upper_lower, n, alpha, x, incx, a)
    @ccall liboneapi_support.onemklChpr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Cfloat, x::Ptr{ComplexF32},
                                        incx::Int64, a::Ptr{ComplexF32})::Cint
end

function onemklZhpr(device_queue, upper_lower, n, alpha, x, incx, a)
    @ccall liboneapi_support.onemklZhpr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Cdouble, x::Ptr{ComplexF32},
                                        incx::Int64, a::Ptr{ComplexF32})::Cint
end

function onemklChpr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a)
    @ccall liboneapi_support.onemklChpr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::ComplexF32, x::Ptr{ComplexF32},
                                         incx::Int64, y::Ptr{ComplexF32}, incy::Int64,
                                         a::Ptr{ComplexF32})::Cint
end

function onemklZhpr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a)
    @ccall liboneapi_support.onemklZhpr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::ComplexF32, x::Ptr{ComplexF32},
                                         incx::Int64, y::Ptr{ComplexF32}, incy::Int64,
                                         a::Ptr{ComplexF32})::Cint
end

function onemklSsbmv(device_queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSsbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, k::Int64, alpha::Ref{Cfloat},
                                         a::ZePtr{Cfloat}, lda::Int64, x::ZePtr{Cfloat},
                                         incx::Int64, beta::Ref{Cfloat}, y::ZePtr{Cfloat},
                                         incy::Int64)::Cint
end

function onemklDsbmv(device_queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDsbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, k::Int64, alpha::Ref{Cdouble},
                                         a::ZePtr{Cdouble}, lda::Int64, x::ZePtr{Cdouble},
                                         incx::Int64, beta::Ref{Cdouble}, y::ZePtr{Cdouble},
                                         incy::Int64)::Cint
end

function onemklSsymv(device_queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSsymv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{Cfloat}, a::ZePtr{Cfloat},
                                         lda::Int64, x::ZePtr{Cfloat}, incx::Int64,
                                         beta::Ref{Cfloat}, y::ZePtr{Cfloat},
                                         incy::Int64)::Cint
end

function onemklDsymv(device_queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDsymv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{Cdouble}, a::ZePtr{Cdouble},
                                         lda::Int64, x::ZePtr{Cdouble}, incx::Int64,
                                         beta::Ref{Cdouble}, y::ZePtr{Cdouble},
                                         incy::Int64)::Cint
end

function onemklCsymv(device_queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklCsymv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{ComplexF32},
                                         a::ZePtr{ComplexF32}, lda::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         beta::Ref{ComplexF32}, y::ZePtr{ComplexF32},
                                         incy::Int64)::Cint
end

function onemklZsymv(device_queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZsymv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Ref{ComplexF64},
                                         a::ZePtr{ComplexF64}, lda::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         beta::Ref{ComplexF64}, y::ZePtr{ComplexF64},
                                         incy::Int64)::Cint
end

function onemklSsyr(device_queue, upper_lower, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklSsyr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Ref{Cfloat}, x::ZePtr{Cfloat},
                                        incx::Int64, a::ZePtr{Cfloat}, lda::Int64)::Cint
end

function onemklDsyr(device_queue, upper_lower, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklDsyr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Ref{Cdouble}, x::ZePtr{Cdouble},
                                        incx::Int64, a::ZePtr{Cdouble}, lda::Int64)::Cint
end

function onemklCsyr(device_queue, upper_lower, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklCsyr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Ref{ComplexF32},
                                        x::ZePtr{ComplexF32}, incx::Int64,
                                        a::ZePtr{ComplexF32}, lda::Int64)::Cint
end

function onemklZsyr(device_queue, upper_lower, n, alpha, x, incx, a, lda)
    @ccall liboneapi_support.onemklZsyr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Ref{ComplexF64},
                                        x::ZePtr{ComplexF64}, incx::Int64,
                                        a::ZePtr{ComplexF64}, lda::Int64)::Cint
end

function onemklSsyr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklSsyr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Cfloat, x::Ptr{Cfloat},
                                         incx::Int64, y::Ptr{Cfloat}, incy::Int64,
                                         a::Ptr{Cfloat}, lda::Int64)::Cint
end

function onemklDsyr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklDsyr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Cdouble, x::Ptr{Cdouble},
                                         incx::Int64, y::Ptr{Cdouble}, incy::Int64,
                                         a::Ptr{Cdouble}, lda::Int64)::Cint
end

function onemklCsyr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklCsyr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::ComplexF32, x::Ptr{ComplexF32},
                                         incx::Int64, y::Ptr{ComplexF32}, incy::Int64,
                                         a::Ptr{ComplexF32}, lda::Int64)::Cint
end

function onemklZsyr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a, lda)
    @ccall liboneapi_support.onemklZsyr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::ComplexF32, x::Ptr{ComplexF32},
                                         incx::Int64, y::Ptr{ComplexF32}, incy::Int64,
                                         a::Ptr{ComplexF32}, lda::Int64)::Cint
end

function onemklSspmv(device_queue, upper_lower, n, alpha, a, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSspmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Cfloat, a::Ptr{Cfloat},
                                         x::Ptr{Cfloat}, incx::Int64, beta::Cfloat,
                                         y::Ptr{Cfloat}, incy::Int64)::Cint
end

function onemklDspmv(device_queue, upper_lower, n, alpha, a, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDspmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Cdouble, a::Ptr{Cdouble},
                                         x::Ptr{Cdouble}, incx::Int64, beta::Cdouble,
                                         y::Ptr{Cdouble}, incy::Int64)::Cint
end

function onemklSspr(device_queue, upper_lower, n, alpha, x, incx, a)
    @ccall liboneapi_support.onemklSspr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Cfloat, x::Ptr{Cfloat},
                                        incx::Int64, a::Ptr{Cfloat})::Cint
end

function onemklDspr(device_queue, upper_lower, n, alpha, x, incx, a)
    @ccall liboneapi_support.onemklDspr(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                        n::Int64, alpha::Cdouble, x::Ptr{Cdouble},
                                        incx::Int64, a::Ptr{Cdouble})::Cint
end

function onemklSspr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a)
    @ccall liboneapi_support.onemklSspr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Cfloat, x::Ptr{Cfloat},
                                         incx::Int64, y::Ptr{Cfloat}, incy::Int64,
                                         a::Ptr{Cfloat})::Cint
end

function onemklDspr2(device_queue, upper_lower, n, alpha, x, incx, y, incy, a)
    @ccall liboneapi_support.onemklDspr2(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         n::Int64, alpha::Cdouble, x::Ptr{Cdouble},
                                         incx::Int64, y::Ptr{Cdouble}, incy::Int64,
                                         a::Ptr{Cdouble})::Cint
end

function onemklStbmv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklStbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64)::Cint
end

function onemklDtbmv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64)::Cint
end

function onemklCtbmv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::ZePtr{ComplexF32},
                                         lda::Int64, x::ZePtr{ComplexF32},
                                         incx::Int64)::Cint
end

function onemklZtbmv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtbmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::ZePtr{ComplexF64},
                                         lda::Int64, x::ZePtr{ComplexF64},
                                         incx::Int64)::Cint
end

function onemklStbsv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklStbsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::Ptr{Cfloat}, lda::Int64,
                                         x::Ptr{Cfloat}, incx::Int64)::Cint
end

function onemklDtbsv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtbsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::Ptr{Cdouble}, lda::Int64,
                                         x::Ptr{Cdouble}, incx::Int64)::Cint
end

function onemklCtbsv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtbsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                         x::Ptr{ComplexF32}, incx::Int64)::Cint
end

function onemklZtbsv(device_queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtbsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, k::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                         x::Ptr{ComplexF32}, incx::Int64)::Cint
end

function onemklStpmv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklStpmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{Cfloat}, x::Ptr{Cfloat},
                                         incx::Int64)::Cint
end

function onemklDtpmv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklDtpmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{Cdouble}, x::Ptr{Cdouble},
                                         incx::Int64)::Cint
end

function onemklCtpmv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklCtpmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{ComplexF32}, x::Ptr{ComplexF32},
                                         incx::Int64)::Cint
end

function onemklZtpmv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklZtpmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{ComplexF32}, x::Ptr{ComplexF32},
                                         incx::Int64)::Cint
end

function onemklStpsv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklStpsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{Cfloat}, x::Ptr{Cfloat},
                                         incx::Int64)::Cint
end

function onemklDtpsv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklDtpsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{Cdouble}, x::Ptr{Cdouble},
                                         incx::Int64)::Cint
end

function onemklCtpsv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklCtpsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{ComplexF32}, x::Ptr{ComplexF32},
                                         incx::Int64)::Cint
end

function onemklZtpsv(device_queue, upper_lower, trans, unit_diag, n, a, x, incx)
    @ccall liboneapi_support.onemklZtpsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::Ptr{ComplexF32}, x::Ptr{ComplexF32},
                                         incx::Int64)::Cint
end

function onemklStrmv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklStrmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64)::Cint
end

function onemklDtrmv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtrmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64)::Cint
end

function onemklCtrmv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtrmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64)::Cint
end

function onemklZtrmv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtrmv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64)::Cint
end

function onemklStrsv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklStrsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64)::Cint
end

function onemklDtrsv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklDtrsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64)::Cint
end

function onemklCtrsv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklCtrsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64)::Cint
end

function onemklZtrsv(device_queue, upper_lower, trans, unit_diag, n, a, lda, x, incx)
    @ccall liboneapi_support.onemklZtrsv(device_queue::syclQueue_t, upper_lower::onemklUplo,
                                         trans::onemklTranspose, unit_diag::onemklDiag,
                                         n::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64)::Cint
end

function onemklCdotc(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklCdotc(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         y::ZePtr{ComplexF32}, incy::Int64,
                                         result::RefOrZeRef{ComplexF32})::Cint
end

function onemklZdotc(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklZdotc(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         y::ZePtr{ComplexF64}, incy::Int64,
                                         result::RefOrZeRef{ComplexF64})::Cint
end

function onemklCdotu(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklCdotu(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         y::ZePtr{ComplexF32}, incy::Int64,
                                         result::RefOrZeRef{ComplexF32})::Cint
end

function onemklZdotu(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklZdotu(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         y::ZePtr{ComplexF64}, incy::Int64,
                                         result::RefOrZeRef{ComplexF64})::Cint
end

function onemklSiamax(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklSiamax(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{Cfloat}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklDiamax(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklDiamax(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{Cdouble}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklCiamax(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklCiamax(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{ComplexF32}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklZiamax(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklZiamax(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{ComplexF64}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklSiamin(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklSiamin(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{Cfloat}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklDiamin(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklDiamin(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{Cdouble}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklCiamin(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklCiamin(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{ComplexF32}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklZiamin(device_queue, n, x, incx, result, base)
    @ccall liboneapi_support.onemklZiamin(device_queue::syclQueue_t, n::Int64,
                                          x::ZePtr{ComplexF64}, incx::Int64,
                                          result::ZePtr{Int64}, base::onemklIndex)::Cint
end

function onemklSasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSasum(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64,
                                         result::ZePtr{Cfloat})::Cint
end

function onemklDasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDasum(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64,
                                         result::ZePtr{Cdouble})::Cint
end

function onemklCasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCasum(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         result::ZePtr{Cfloat})::Cint
end

function onemklZasum(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZasum(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         result::ZePtr{Float64})::Cint
end

function onemklSaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklSaxpy(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{Cfloat}, x::ZePtr{Cfloat}, incx::Int64,
                                         y::ZePtr{Cfloat}, incy::Int64)::Cint
end

function onemklDaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklDaxpy(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{Cdouble}, x::ZePtr{Cdouble},
                                         incx::Int64, y::ZePtr{Cdouble}, incy::Int64)::Cint
end

function onemklCaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklCaxpy(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{ComplexF32}, x::ZePtr{ComplexF32},
                                         incx::Int64, y::ZePtr{ComplexF32},
                                         incy::Int64)::Cint
end

function onemklZaxpy(device_queue, n, alpha, x, incx, y, incy)
    @ccall liboneapi_support.onemklZaxpy(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{ComplexF64}, x::ZePtr{ComplexF64},
                                         incx::Int64, y::ZePtr{ComplexF64},
                                         incy::Int64)::Cint
end

function onemklSaxpby(device_queue, n, alpha, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklSaxpby(device_queue::syclQueue_t, n::Int64,
                                          alpha::Ref{Cfloat}, x::ZePtr{Cfloat}, incx::Int64,
                                          beta::Ref{Cfloat}, y::ZePtr{Cfloat},
                                          incy::Int64)::Cint
end

function onemklDaxpby(device_queue, n, alpha, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklDaxpby(device_queue::syclQueue_t, n::Int64,
                                          alpha::Ref{Cdouble}, x::ZePtr{Cdouble},
                                          incx::Int64, beta::Ref{Cdouble},
                                          y::ZePtr{Cdouble}, incy::Int64)::Cint
end

function onemklCaxpby(device_queue, n, alpha, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklCaxpby(device_queue::syclQueue_t, n::Int64,
                                          alpha::Ref{ComplexF32}, x::ZePtr{ComplexF32},
                                          incx::Int64, beta::Ref{ComplexF32},
                                          y::ZePtr{ComplexF32}, incy::Int64)::Cint
end

function onemklZaxpby(device_queue, n, alpha, x, incx, beta, y, incy)
    @ccall liboneapi_support.onemklZaxpby(device_queue::syclQueue_t, n::Int64,
                                          alpha::Ref{ComplexF64}, x::ZePtr{ComplexF64},
                                          incx::Int64, beta::Ref{ComplexF64},
                                          y::ZePtr{ComplexF64}, incy::Int64)::Cint
end

function onemklScopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklScopy(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat},
                                         incy::Int64)::Cint
end

function onemklDcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklDcopy(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble},
                                         incy::Int64)::Cint
end

function onemklCcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklCcopy(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         y::ZePtr{ComplexF32}, incy::Int64)::Cint
end

function onemklZcopy(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklZcopy(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         y::ZePtr{ComplexF64}, incy::Int64)::Cint
end

function onemklSdot(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklSdot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat},
                                        incy::Int64, result::RefOrZeRef{Cfloat})::Cint
end

function onemklDdot(device_queue, n, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklDdot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble},
                                        incy::Int64, result::RefOrZeRef{Cdouble})::Cint
end

function onemklSsdsdot(device_queue, n, sb, x, incx, y, incy, result)
    @ccall liboneapi_support.onemklSsdsdot(device_queue::syclQueue_t, n::Int64, sb::Cfloat,
                                           x::Ptr{Cfloat}, incx::Int64, y::Ptr{Cfloat},
                                           incy::Int64, result::Ptr{Cfloat})::Cint
end

function onemklSnrm2(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklSnrm2(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64,
                                         result::RefOrZeRef{Cfloat})::Cint
end

function onemklDnrm2(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklDnrm2(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64,
                                         result::RefOrZeRef{Cdouble})::Cint
end

function onemklCnrm2(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklCnrm2(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         result::RefOrZeRef{Cfloat})::Cint
end

function onemklZnrm2(device_queue, n, x, incx, result)
    @ccall liboneapi_support.onemklZnrm2(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         result::RefOrZeRef{Cdouble})::Cint
end

function onemklSrot(device_queue, n, x, incx, y, incy, c, s)
    @ccall liboneapi_support.onemklSrot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat},
                                        incy::Int64, c::Cfloat, s::Cfloat)::Cint
end

function onemklDrot(device_queue, n, x, incx, y, incy, c, s)
    @ccall liboneapi_support.onemklDrot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble},
                                        incy::Int64, c::Cdouble, s::Cdouble)::Cint
end

function onemklCSrot(device_queue, n, x, incx, y, incy, c, s)
    @ccall liboneapi_support.onemklCSrot(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         y::ZePtr{ComplexF32}, incy::Int64, c::Cfloat,
                                         s::Cfloat)::Cint
end

function onemklCrot(device_queue, n, x, incx, y, incy, c, s)
    @ccall liboneapi_support.onemklCrot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF32}, incx::Int64,
                                        y::ZePtr{ComplexF32}, incy::Int64, c::Cfloat,
                                        s::ComplexF32)::Cint
end

function onemklZDrot(device_queue, n, x, incx, y, incy, c, s)
    @ccall liboneapi_support.onemklZDrot(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         y::ZePtr{ComplexF64}, incy::Int64, c::Cdouble,
                                         s::Cdouble)::Cint
end

function onemklZrot(device_queue, n, x, incx, y, incy, c, s)
    @ccall liboneapi_support.onemklZrot(device_queue::syclQueue_t, n::Int64,
                                        x::ZePtr{ComplexF64}, incx::Int64,
                                        y::ZePtr{ComplexF64}, incy::Int64, c::Cdouble,
                                        s::ComplexF32)::Cint
end

function onemklSrotg(device_queue, a, b, c, s)
    @ccall liboneapi_support.onemklSrotg(device_queue::syclQueue_t, a::Ptr{Cfloat},
                                         b::Ptr{Cfloat}, c::Ptr{Cfloat},
                                         s::Ptr{Cfloat})::Cint
end

function onemklDrotg(device_queue, a, b, c, s)
    @ccall liboneapi_support.onemklDrotg(device_queue::syclQueue_t, a::Ptr{Cdouble},
                                         b::Ptr{Cdouble}, c::Ptr{Cdouble},
                                         s::Ptr{Cdouble})::Cint
end

function onemklCrotg(device_queue, a, b, c, s)
    @ccall liboneapi_support.onemklCrotg(device_queue::syclQueue_t, a::Ptr{ComplexF32},
                                         b::Ptr{ComplexF32}, c::Ptr{Cfloat},
                                         s::Ptr{ComplexF32})::Cint
end

function onemklZrotg(device_queue, a, b, c, s)
    @ccall liboneapi_support.onemklZrotg(device_queue::syclQueue_t, a::Ptr{ComplexF32},
                                         b::Ptr{ComplexF32}, c::Ptr{Cdouble},
                                         s::Ptr{ComplexF32})::Cint
end

function onemklSrotm(device_queue, n, x, incx, y, incy, param)
    @ccall liboneapi_support.onemklSrotm(device_queue::syclQueue_t, n::Int64,
                                         x::Ptr{Cfloat}, incx::Int64, y::Ptr{Cfloat},
                                         incy::Int64, param::Ptr{Cfloat})::Cint
end

function onemklDrotm(device_queue, n, x, incx, y, incy, param)
    @ccall liboneapi_support.onemklDrotm(device_queue::syclQueue_t, n::Int64,
                                         x::Ptr{Cdouble}, incx::Int64, y::Ptr{Cdouble},
                                         incy::Int64, param::Ptr{Cdouble})::Cint
end

function onemklSrotmg(device_queue, d1, d2, x1, y1, param)
    @ccall liboneapi_support.onemklSrotmg(device_queue::syclQueue_t, d1::Ptr{Cfloat},
                                          d2::Ptr{Cfloat}, x1::Ptr{Cfloat}, y1::Cfloat,
                                          param::Ptr{Cfloat})::Cint
end

function onemklDrotmg(device_queue, d1, d2, x1, y1, param)
    @ccall liboneapi_support.onemklDrotmg(device_queue::syclQueue_t, d1::Ptr{Cdouble},
                                          d2::Ptr{Cdouble}, x1::Ptr{Cdouble}, y1::Cdouble,
                                          param::Ptr{Cdouble})::Cint
end

function onemklSscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklSscal(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{Cfloat}, x::ZePtr{Cfloat},
                                         incx::Int64)::Cint
end

function onemklDscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklDscal(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{Cdouble}, x::ZePtr{Cdouble},
                                         incx::Int64)::Cint
end

function onemklCSscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklCSscal(device_queue::syclQueue_t, n::Int64,
                                          alpha::Cfloat, x::ZePtr{ComplexF32},
                                          incx::Int64)::Cint
end

function onemklZDscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklZDscal(device_queue::syclQueue_t, n::Int64,
                                          alpha::Cdouble, x::ZePtr{ComplexF64},
                                          incx::Int64)::Cint
end

function onemklCscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklCscal(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{ComplexF32}, x::ZePtr{ComplexF32},
                                         incx::Int64)::Cint
end

function onemklZscal(device_queue, n, alpha, x, incx)
    @ccall liboneapi_support.onemklZscal(device_queue::syclQueue_t, n::Int64,
                                         alpha::Ref{ComplexF64}, x::ZePtr{ComplexF64},
                                         incx::Int64)::Cint
end

function onemklSswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklSswap(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cfloat}, incx::Int64, y::ZePtr{Cfloat},
                                         incy::Int64)::Cint
end

function onemklDswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklDswap(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{Cdouble}, incx::Int64, y::ZePtr{Cdouble},
                                         incy::Int64)::Cint
end

function onemklCswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklCswap(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF32}, incx::Int64,
                                         y::ZePtr{ComplexF32}, incy::Int64)::Cint
end

function onemklZswap(device_queue, n, x, incx, y, incy)
    @ccall liboneapi_support.onemklZswap(device_queue::syclQueue_t, n::Int64,
                                         x::ZePtr{ComplexF64}, incx::Int64,
                                         y::ZePtr{ComplexF64}, incy::Int64)::Cint
end

function onemklSgemm_batch(device_queue, transa, transb, m, n, k, alpha, a, lda, stride_a,
                           b, ldb, stride_b, beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklSgemm_batch(device_queue::syclQueue_t,
                                               transa::onemklTranspose,
                                               transb::onemklTranspose, m::Int64, n::Int64,
                                               k::Int64, alpha::Cfloat, a::Ptr{Cfloat},
                                               lda::Int64, stride_a::Int64, b::Ptr{Cfloat},
                                               ldb::Int64, stride_b::Int64, beta::Cfloat,
                                               c::Ptr{Cfloat}, ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklDgemm_batch(device_queue, transa, transb, m, n, k, alpha, a, lda, stride_a,
                           b, ldb, stride_b, beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklDgemm_batch(device_queue::syclQueue_t,
                                               transa::onemklTranspose,
                                               transb::onemklTranspose, m::Int64, n::Int64,
                                               k::Int64, alpha::Cdouble, a::Ptr{Cdouble},
                                               lda::Int64, stride_a::Int64, b::Ptr{Cdouble},
                                               ldb::Int64, stride_b::Int64, beta::Cdouble,
                                               c::Ptr{Cdouble}, ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklCgemm_batch(device_queue, transa, transb, m, n, k, alpha, a, lda, stride_a,
                           b, ldb, stride_b, beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklCgemm_batch(device_queue::syclQueue_t,
                                               transa::onemklTranspose,
                                               transb::onemklTranspose, m::Int64, n::Int64,
                                               k::Int64, alpha::ComplexF32,
                                               a::Ptr{ComplexF32}, lda::Int64,
                                               stride_a::Int64, b::Ptr{ComplexF32},
                                               ldb::Int64, stride_b::Int64,
                                               beta::ComplexF32, c::Ptr{ComplexF32},
                                               ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklZgemm_batch(device_queue, transa, transb, m, n, k, alpha, a, lda, stride_a,
                           b, ldb, stride_b, beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklZgemm_batch(device_queue::syclQueue_t,
                                               transa::onemklTranspose,
                                               transb::onemklTranspose, m::Int64, n::Int64,
                                               k::Int64, alpha::ComplexF32,
                                               a::Ptr{ComplexF32}, lda::Int64,
                                               stride_a::Int64, b::Ptr{ComplexF32},
                                               ldb::Int64, stride_b::Int64,
                                               beta::ComplexF32, c::Ptr{ComplexF32},
                                               ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklSsyrk_batch(device_queue, upper_lower, trans, n, k, alpha, a, lda, stride_a,
                           beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklSsyrk_batch(device_queue::syclQueue_t,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose, n::Int64, k::Int64,
                                               alpha::Cfloat, a::Ptr{Cfloat}, lda::Int64,
                                               stride_a::Int64, beta::Cfloat,
                                               c::Ptr{Cfloat}, ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklDsyrk_batch(device_queue, upper_lower, trans, n, k, alpha, a, lda, stride_a,
                           beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklDsyrk_batch(device_queue::syclQueue_t,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose, n::Int64, k::Int64,
                                               alpha::Cdouble, a::Ptr{Cdouble}, lda::Int64,
                                               stride_a::Int64, beta::Cdouble,
                                               c::Ptr{Cdouble}, ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklCsyrk_batch(device_queue, upper_lower, trans, n, k, alpha, a, lda, stride_a,
                           beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklCsyrk_batch(device_queue::syclQueue_t,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose, n::Int64, k::Int64,
                                               alpha::ComplexF32, a::Ptr{ComplexF32},
                                               lda::Int64, stride_a::Int64,
                                               beta::ComplexF32, c::Ptr{ComplexF32},
                                               ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklZsyrk_batch(device_queue, upper_lower, trans, n, k, alpha, a, lda, stride_a,
                           beta, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklZsyrk_batch(device_queue::syclQueue_t,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose, n::Int64, k::Int64,
                                               alpha::ComplexF32, a::Ptr{ComplexF32},
                                               lda::Int64, stride_a::Int64,
                                               beta::ComplexF32, c::Ptr{ComplexF32},
                                               ldc::Int64, stride_c::Int64,
                                               batch_size::Int64)::Cint
end

function onemklStrsm_batch(device_queue, left_right, upper_lower, trans, unit_diag, m, n,
                           alpha, a, lda, stride_a, b, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklStrsm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose,
                                               unit_diag::onemklDiag, m::Int64, n::Int64,
                                               alpha::Cfloat, a::Ptr{Cfloat}, lda::Int64,
                                               stride_a::Int64, b::Ptr{Cfloat}, ldb::Int64,
                                               stride_b::Int64, batch_size::Int64)::Cint
end

function onemklDtrsm_batch(device_queue, left_right, upper_lower, trans, unit_diag, m, n,
                           alpha, a, lda, stride_a, b, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklDtrsm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose,
                                               unit_diag::onemklDiag, m::Int64, n::Int64,
                                               alpha::Cdouble, a::Ptr{Cdouble}, lda::Int64,
                                               stride_a::Int64, b::Ptr{Cdouble}, ldb::Int64,
                                               stride_b::Int64, batch_size::Int64)::Cint
end

function onemklCtrsm_batch(device_queue, left_right, upper_lower, trans, unit_diag, m, n,
                           alpha, a, lda, stride_a, b, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklCtrsm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose,
                                               unit_diag::onemklDiag, m::Int64, n::Int64,
                                               alpha::ComplexF32, a::Ptr{ComplexF32},
                                               lda::Int64, stride_a::Int64,
                                               b::Ptr{ComplexF32}, ldb::Int64,
                                               stride_b::Int64, batch_size::Int64)::Cint
end

function onemklZtrsm_batch(device_queue, left_right, upper_lower, trans, unit_diag, m, n,
                           alpha, a, lda, stride_a, b, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklZtrsm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide,
                                               upper_lower::onemklUplo,
                                               trans::onemklTranspose,
                                               unit_diag::onemklDiag, m::Int64, n::Int64,
                                               alpha::ComplexF32, a::Ptr{ComplexF32},
                                               lda::Int64, stride_a::Int64,
                                               b::Ptr{ComplexF32}, ldb::Int64,
                                               stride_b::Int64, batch_size::Int64)::Cint
end

function onemklSgemv_batch(device_queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                           stridex, beta, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklSgemv_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               alpha::Cfloat, a::Ptr{Cfloat}, lda::Int64,
                                               stridea::Int64, x::Ptr{Cfloat}, incx::Int64,
                                               stridex::Int64, beta::Cfloat, y::Ptr{Cfloat},
                                               incy::Int64, stridey::Int64,
                                               batch_size::Int64)::Cint
end

function onemklDgemv_batch(device_queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                           stridex, beta, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklDgemv_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               alpha::Cdouble, a::Ptr{Cdouble}, lda::Int64,
                                               stridea::Int64, x::Ptr{Cdouble}, incx::Int64,
                                               stridex::Int64, beta::Cdouble,
                                               y::Ptr{Cdouble}, incy::Int64, stridey::Int64,
                                               batch_size::Int64)::Cint
end

function onemklCgemv_batch(device_queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                           stridex, beta, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklCgemv_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               alpha::ComplexF32, a::Ptr{ComplexF32},
                                               lda::Int64, stridea::Int64,
                                               x::Ptr{ComplexF32}, incx::Int64,
                                               stridex::Int64, beta::ComplexF32,
                                               y::Ptr{ComplexF32}, incy::Int64,
                                               stridey::Int64, batch_size::Int64)::Cint
end

function onemklZgemv_batch(device_queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                           stridex, beta, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklZgemv_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               alpha::ComplexF32, a::Ptr{ComplexF32},
                                               lda::Int64, stridea::Int64,
                                               x::Ptr{ComplexF32}, incx::Int64,
                                               stridex::Int64, beta::ComplexF32,
                                               y::Ptr{ComplexF32}, incy::Int64,
                                               stridey::Int64, batch_size::Int64)::Cint
end

function onemklSdgmm_batch(device_queue, left_right, m, n, a, lda, stridea, x, incx,
                           stridex, c, ldc, stridec, batch_size)
    @ccall liboneapi_support.onemklSdgmm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide, m::Int64, n::Int64,
                                               a::Ptr{Cfloat}, lda::Int64, stridea::Int64,
                                               x::Ptr{Cfloat}, incx::Int64, stridex::Int64,
                                               c::Ptr{Cfloat}, ldc::Int64, stridec::Int64,
                                               batch_size::Int64)::Cint
end

function onemklDdgmm_batch(device_queue, left_right, m, n, a, lda, stridea, x, incx,
                           stridex, c, ldc, stridec, batch_size)
    @ccall liboneapi_support.onemklDdgmm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide, m::Int64, n::Int64,
                                               a::Ptr{Cdouble}, lda::Int64, stridea::Int64,
                                               x::Ptr{Cdouble}, incx::Int64, stridex::Int64,
                                               c::Ptr{Cdouble}, ldc::Int64, stridec::Int64,
                                               batch_size::Int64)::Cint
end

function onemklCdgmm_batch(device_queue, left_right, m, n, a, lda, stridea, x, incx,
                           stridex, c, ldc, stridec, batch_size)
    @ccall liboneapi_support.onemklCdgmm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide, m::Int64, n::Int64,
                                               a::Ptr{ComplexF32}, lda::Int64,
                                               stridea::Int64, x::Ptr{ComplexF32},
                                               incx::Int64, stridex::Int64,
                                               c::Ptr{ComplexF32}, ldc::Int64,
                                               stridec::Int64, batch_size::Int64)::Cint
end

function onemklZdgmm_batch(device_queue, left_right, m, n, a, lda, stridea, x, incx,
                           stridex, c, ldc, stridec, batch_size)
    @ccall liboneapi_support.onemklZdgmm_batch(device_queue::syclQueue_t,
                                               left_right::onemklSide, m::Int64, n::Int64,
                                               a::Ptr{ComplexF32}, lda::Int64,
                                               stridea::Int64, x::Ptr{ComplexF32},
                                               incx::Int64, stridex::Int64,
                                               c::Ptr{ComplexF32}, ldc::Int64,
                                               stridec::Int64, batch_size::Int64)::Cint
end

function onemklSaxpy_batch(device_queue, n, alpha, x, incx, stridex, y, incy, stridey,
                           batch_size)
    @ccall liboneapi_support.onemklSaxpy_batch(device_queue::syclQueue_t, n::Int64,
                                               alpha::Cfloat, x::Ptr{Cfloat}, incx::Int64,
                                               stridex::Int64, y::Ptr{Cfloat}, incy::Int64,
                                               stridey::Int64, batch_size::Int64)::Cint
end

function onemklDaxpy_batch(device_queue, n, alpha, x, incx, stridex, y, incy, stridey,
                           batch_size)
    @ccall liboneapi_support.onemklDaxpy_batch(device_queue::syclQueue_t, n::Int64,
                                               alpha::Cdouble, x::Ptr{Cdouble}, incx::Int64,
                                               stridex::Int64, y::Ptr{Cdouble}, incy::Int64,
                                               stridey::Int64, batch_size::Int64)::Cint
end

function onemklCaxpy_batch(device_queue, n, alpha, x, incx, stridex, y, incy, stridey,
                           batch_size)
    @ccall liboneapi_support.onemklCaxpy_batch(device_queue::syclQueue_t, n::Int64,
                                               alpha::ComplexF32, x::Ptr{ComplexF32},
                                               incx::Int64, stridex::Int64,
                                               y::Ptr{ComplexF32}, incy::Int64,
                                               stridey::Int64, batch_size::Int64)::Cint
end

function onemklZaxpy_batch(device_queue, n, alpha, x, incx, stridex, y, incy, stridey,
                           batch_size)
    @ccall liboneapi_support.onemklZaxpy_batch(device_queue::syclQueue_t, n::Int64,
                                               alpha::ComplexF32, x::Ptr{ComplexF32},
                                               incx::Int64, stridex::Int64,
                                               y::Ptr{ComplexF32}, incy::Int64,
                                               stridey::Int64, batch_size::Int64)::Cint
end

function onemklScopy_batch(device_queue, n, x, incx, stridex, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklScopy_batch(device_queue::syclQueue_t, n::Int64,
                                               x::Ptr{Cfloat}, incx::Int64, stridex::Int64,
                                               y::Ptr{Cfloat}, incy::Int64, stridey::Int64,
                                               batch_size::Int64)::Cint
end

function onemklDcopy_batch(device_queue, n, x, incx, stridex, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklDcopy_batch(device_queue::syclQueue_t, n::Int64,
                                               x::Ptr{Cdouble}, incx::Int64, stridex::Int64,
                                               y::Ptr{Cdouble}, incy::Int64, stridey::Int64,
                                               batch_size::Int64)::Cint
end

function onemklCcopy_batch(device_queue, n, x, incx, stridex, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklCcopy_batch(device_queue::syclQueue_t, n::Int64,
                                               x::Ptr{ComplexF32}, incx::Int64,
                                               stridex::Int64, y::Ptr{ComplexF32},
                                               incy::Int64, stridey::Int64,
                                               batch_size::Int64)::Cint
end

function onemklZcopy_batch(device_queue, n, x, incx, stridex, y, incy, stridey, batch_size)
    @ccall liboneapi_support.onemklZcopy_batch(device_queue::syclQueue_t, n::Int64,
                                               x::Ptr{ComplexF32}, incx::Int64,
                                               stridex::Int64, y::Ptr{ComplexF32},
                                               incy::Int64, stridey::Int64,
                                               batch_size::Int64)::Cint
end

function onemklSgemmt(device_queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                      ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklSgemmt(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, transa::onemklTranspose,
                                          transb::onemklTranspose, n::Int64, k::Int64,
                                          alpha::Cfloat, a::Ptr{Cfloat}, lda::Int64,
                                          b::Ptr{Cfloat}, ldb::Int64, beta::Cfloat,
                                          c::Ptr{Cfloat}, ldc::Int64)::Cint
end

function onemklDgemmt(device_queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                      ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklDgemmt(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, transa::onemklTranspose,
                                          transb::onemklTranspose, n::Int64, k::Int64,
                                          alpha::Cdouble, a::Ptr{Cdouble}, lda::Int64,
                                          b::Ptr{Cdouble}, ldb::Int64, beta::Cdouble,
                                          c::Ptr{Cdouble}, ldc::Int64)::Cint
end

function onemklCgemmt(device_queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                      ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklCgemmt(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, transa::onemklTranspose,
                                          transb::onemklTranspose, n::Int64, k::Int64,
                                          alpha::ComplexF32, a::Ptr{ComplexF32}, lda::Int64,
                                          b::Ptr{ComplexF32}, ldb::Int64, beta::ComplexF32,
                                          c::Ptr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZgemmt(device_queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                      ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklZgemmt(device_queue::syclQueue_t,
                                          upper_lower::onemklUplo, transa::onemklTranspose,
                                          transb::onemklTranspose, n::Int64, k::Int64,
                                          alpha::ComplexF32, a::Ptr{ComplexF32}, lda::Int64,
                                          b::Ptr{ComplexF32}, ldb::Int64, beta::ComplexF32,
                                          c::Ptr{ComplexF32}, ldc::Int64)::Cint
end

function onemklSimatcopy(device_queue, trans, m, n, alpha, ab, lda, ldb)
    @ccall liboneapi_support.onemklSimatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::Cfloat, ab::Ptr{Cfloat}, lda::Int64,
                                             ldb::Int64)::Cint
end

function onemklDimatcopy(device_queue, trans, m, n, alpha, ab, lda, ldb)
    @ccall liboneapi_support.onemklDimatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::Cdouble, ab::Ptr{Cdouble}, lda::Int64,
                                             ldb::Int64)::Cint
end

function onemklCimatcopy(device_queue, trans, m, n, alpha, ab, lda, ldb)
    @ccall liboneapi_support.onemklCimatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::ComplexF32, ab::Ptr{ComplexF32},
                                             lda::Int64, ldb::Int64)::Cint
end

function onemklZimatcopy(device_queue, trans, m, n, alpha, ab, lda, ldb)
    @ccall liboneapi_support.onemklZimatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::ComplexF32, ab::Ptr{ComplexF32},
                                             lda::Int64, ldb::Int64)::Cint
end

function onemklSomatcopy(device_queue, trans, m, n, alpha, a, lda, b, ldb)
    @ccall liboneapi_support.onemklSomatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::Cfloat, a::Ptr{Cfloat}, lda::Int64,
                                             b::Ptr{Cfloat}, ldb::Int64)::Cint
end

function onemklDomatcopy(device_queue, trans, m, n, alpha, a, lda, b, ldb)
    @ccall liboneapi_support.onemklDomatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::Cdouble, a::Ptr{Cdouble}, lda::Int64,
                                             b::Ptr{Cdouble}, ldb::Int64)::Cint
end

function onemklComatcopy(device_queue, trans, m, n, alpha, a, lda, b, ldb)
    @ccall liboneapi_support.onemklComatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::ComplexF32, a::Ptr{ComplexF32},
                                             lda::Int64, b::Ptr{ComplexF32},
                                             ldb::Int64)::Cint
end

function onemklZomatcopy(device_queue, trans, m, n, alpha, a, lda, b, ldb)
    @ccall liboneapi_support.onemklZomatcopy(device_queue::syclQueue_t,
                                             trans::onemklTranspose, m::Int64, n::Int64,
                                             alpha::ComplexF32, a::Ptr{ComplexF32},
                                             lda::Int64, b::Ptr{ComplexF32},
                                             ldb::Int64)::Cint
end

function onemklSomatadd(device_queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c,
                        ldc)
    @ccall liboneapi_support.onemklSomatadd(device_queue::syclQueue_t,
                                            transa::onemklTranspose,
                                            transb::onemklTranspose, m::Int64, n::Int64,
                                            alpha::Cfloat, a::Ptr{Cfloat}, lda::Int64,
                                            beta::Cfloat, b::Ptr{Cfloat}, ldb::Int64,
                                            c::Ptr{Cfloat}, ldc::Int64)::Cint
end

function onemklDomatadd(device_queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c,
                        ldc)
    @ccall liboneapi_support.onemklDomatadd(device_queue::syclQueue_t,
                                            transa::onemklTranspose,
                                            transb::onemklTranspose, m::Int64, n::Int64,
                                            alpha::Cdouble, a::Ptr{Cdouble}, lda::Int64,
                                            beta::Cdouble, b::Ptr{Cdouble}, ldb::Int64,
                                            c::Ptr{Cdouble}, ldc::Int64)::Cint
end

function onemklComatadd(device_queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c,
                        ldc)
    @ccall liboneapi_support.onemklComatadd(device_queue::syclQueue_t,
                                            transa::onemklTranspose,
                                            transb::onemklTranspose, m::Int64, n::Int64,
                                            alpha::ComplexF32, a::Ptr{ComplexF32},
                                            lda::Int64, beta::ComplexF32,
                                            b::Ptr{ComplexF32}, ldb::Int64,
                                            c::Ptr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZomatadd(device_queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c,
                        ldc)
    @ccall liboneapi_support.onemklZomatadd(device_queue::syclQueue_t,
                                            transa::onemklTranspose,
                                            transb::onemklTranspose, m::Int64, n::Int64,
                                            alpha::ComplexF32, a::Ptr{ComplexF32},
                                            lda::Int64, beta::ComplexF32,
                                            b::Ptr{ComplexF32}, ldb::Int64,
                                            c::Ptr{ComplexF32}, ldc::Int64)::Cint
end

function onemklSimatcopy_batch(device_queue, trans, m, n, alpha, ab, lda, ldb, stride,
                               batch_size)
    @ccall liboneapi_support.onemklSimatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::Cfloat, ab::Ptr{Cfloat},
                                                   lda::Int64, ldb::Int64, stride::Int64,
                                                   batch_size::Int64)::Cint
end

function onemklDimatcopy_batch(device_queue, trans, m, n, alpha, ab, lda, ldb, stride,
                               batch_size)
    @ccall liboneapi_support.onemklDimatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::Cdouble,
                                                   ab::Ptr{Cdouble}, lda::Int64, ldb::Int64,
                                                   stride::Int64, batch_size::Int64)::Cint
end

function onemklCimatcopy_batch(device_queue, trans, m, n, alpha, ab, lda, ldb, stride,
                               batch_size)
    @ccall liboneapi_support.onemklCimatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::ComplexF32,
                                                   ab::Ptr{ComplexF32}, lda::Int64,
                                                   ldb::Int64, stride::Int64,
                                                   batch_size::Int64)::Cint
end

function onemklZimatcopy_batch(device_queue, trans, m, n, alpha, ab, lda, ldb, stride,
                               batch_size)
    @ccall liboneapi_support.onemklZimatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::ComplexF32,
                                                   ab::Ptr{ComplexF32}, lda::Int64,
                                                   ldb::Int64, stride::Int64,
                                                   batch_size::Int64)::Cint
end

function onemklSomatcopy_batch(device_queue, trans, m, n, alpha, a, lda, stride_a, b, ldb,
                               stride_b, batch_size)
    @ccall liboneapi_support.onemklSomatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::Cfloat, a::Ptr{Cfloat},
                                                   lda::Int64, stride_a::Int64,
                                                   b::Ptr{Cfloat}, ldb::Int64,
                                                   stride_b::Int64, batch_size::Int64)::Cint
end

function onemklDomatcopy_batch(device_queue, trans, m, n, alpha, a, lda, stride_a, b, ldb,
                               stride_b, batch_size)
    @ccall liboneapi_support.onemklDomatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::Cdouble,
                                                   a::Ptr{Cdouble}, lda::Int64,
                                                   stride_a::Int64, b::Ptr{Cdouble},
                                                   ldb::Int64, stride_b::Int64,
                                                   batch_size::Int64)::Cint
end

function onemklComatcopy_batch(device_queue, trans, m, n, alpha, a, lda, stride_a, b, ldb,
                               stride_b, batch_size)
    @ccall liboneapi_support.onemklComatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::ComplexF32,
                                                   a::Ptr{ComplexF32}, lda::Int64,
                                                   stride_a::Int64, b::Ptr{ComplexF32},
                                                   ldb::Int64, stride_b::Int64,
                                                   batch_size::Int64)::Cint
end

function onemklZomatcopy_batch(device_queue, trans, m, n, alpha, a, lda, stride_a, b, ldb,
                               stride_b, batch_size)
    @ccall liboneapi_support.onemklZomatcopy_batch(device_queue::syclQueue_t,
                                                   trans::onemklTranspose, m::Int64,
                                                   n::Int64, alpha::ComplexF32,
                                                   a::Ptr{ComplexF32}, lda::Int64,
                                                   stride_a::Int64, b::Ptr{ComplexF32},
                                                   ldb::Int64, stride_b::Int64,
                                                   batch_size::Int64)::Cint
end

function onemklSomatadd_batch(device_queue, transa, transb, m, n, alpha, a, lda, stride_a,
                              beta, b, ldb, stride_b, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklSomatadd_batch(device_queue::syclQueue_t,
                                                  transa::onemklTranspose,
                                                  transb::onemklTranspose, m::Int64,
                                                  n::Int64, alpha::Cfloat, a::Ptr{Cfloat},
                                                  lda::Int64, stride_a::Int64, beta::Cfloat,
                                                  b::Ptr{Cfloat}, ldb::Int64,
                                                  stride_b::Int64, c::Ptr{Cfloat},
                                                  ldc::Int64, stride_c::Int64,
                                                  batch_size::Int64)::Cint
end

function onemklDomatadd_batch(device_queue, transa, transb, m, n, alpha, a, lda, stride_a,
                              beta, b, ldb, stride_b, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklDomatadd_batch(device_queue::syclQueue_t,
                                                  transa::onemklTranspose,
                                                  transb::onemklTranspose, m::Int64,
                                                  n::Int64, alpha::Cdouble, a::Ptr{Cdouble},
                                                  lda::Int64, stride_a::Int64,
                                                  beta::Cdouble, b::Ptr{Cdouble},
                                                  ldb::Int64, stride_b::Int64,
                                                  c::Ptr{Cdouble}, ldc::Int64,
                                                  stride_c::Int64, batch_size::Int64)::Cint
end

function onemklComatadd_batch(device_queue, transa, transb, m, n, alpha, a, lda, stride_a,
                              beta, b, ldb, stride_b, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklComatadd_batch(device_queue::syclQueue_t,
                                                  transa::onemklTranspose,
                                                  transb::onemklTranspose, m::Int64,
                                                  n::Int64, alpha::ComplexF32,
                                                  a::Ptr{ComplexF32}, lda::Int64,
                                                  stride_a::Int64, beta::ComplexF32,
                                                  b::Ptr{ComplexF32}, ldb::Int64,
                                                  stride_b::Int64, c::Ptr{ComplexF32},
                                                  ldc::Int64, stride_c::Int64,
                                                  batch_size::Int64)::Cint
end

function onemklZomatadd_batch(device_queue, transa, transb, m, n, alpha, a, lda, stride_a,
                              beta, b, ldb, stride_b, c, ldc, stride_c, batch_size)
    @ccall liboneapi_support.onemklZomatadd_batch(device_queue::syclQueue_t,
                                                  transa::onemklTranspose,
                                                  transb::onemklTranspose, m::Int64,
                                                  n::Int64, alpha::ComplexF32,
                                                  a::Ptr{ComplexF32}, lda::Int64,
                                                  stride_a::Int64, beta::ComplexF32,
                                                  b::Ptr{ComplexF32}, ldb::Int64,
                                                  stride_b::Int64, c::Ptr{ComplexF32},
                                                  ldc::Int64, stride_c::Int64,
                                                  batch_size::Int64)::Cint
end

function onemklSpotrf(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSpotrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDpotrf(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDpotrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklCpotrf(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCpotrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZpotrf(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZpotrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSpotrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklSpotrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDpotrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklDpotrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCpotrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklCpotrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZpotrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklZpotrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklSpotrs(device_queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklSpotrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, nrhs::Int64, a::ZePtr{Cfloat},
                                          lda::Int64, b::ZePtr{Cfloat}, ldb::Int64,
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDpotrs(device_queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDpotrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, nrhs::Int64, a::ZePtr{Cdouble},
                                          lda::Int64, b::ZePtr{Cdouble}, ldb::Int64,
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklCpotrs(device_queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklCpotrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, nrhs::Int64, a::ZePtr{ComplexF32},
                                          lda::Int64, b::ZePtr{ComplexF32}, ldb::Int64,
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZpotrs(device_queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZpotrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, nrhs::Int64, a::ZePtr{ComplexF64},
                                          lda::Int64, b::ZePtr{ComplexF64}, ldb::Int64,
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSpotrs_scratchpad_size(device_queue, uplo, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklSpotrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklDpotrs_scratchpad_size(device_queue, uplo, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklDpotrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklCpotrs_scratchpad_size(device_queue, uplo, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklCpotrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklZpotrs_scratchpad_size(device_queue, uplo, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklZpotrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklSpotri(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSpotri(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDpotri(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDpotri(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklCpotri(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCpotri(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZpotri(device_queue, uplo, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZpotri(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSpotri_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklSpotri_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDpotri_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklDpotri_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCpotri_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklCpotri_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZpotri_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklZpotri_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklSgebrd_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklSgebrd_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDgebrd_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklDgebrd_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgebrd_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklCgebrd_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZgebrd_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklZgebrd_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgebrd(device_queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklCgebrd(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          d::ZePtr{Float32}, e::ZePtr{ComplexF32},
                                          tauq::ZePtr{ComplexF32}, taup::ZePtr{ComplexF32},
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgebrd(device_queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDgebrd(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{Float64}, lda::Int64, d::ZePtr{Float64},
                                          e::ZePtr{Float64}, tauq::ZePtr{Float64},
                                          taup::ZePtr{Float64}, scratchpad::ZePtr{Float64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgebrd(device_queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklSgebrd(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{Float32}, lda::Int64, d::ZePtr{Float32},
                                          e::ZePtr{Float32}, tauq::ZePtr{Float32},
                                          taup::ZePtr{Float32}, scratchpad::ZePtr{Float32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgebrd(device_queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZgebrd(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          d::ZePtr{Float64}, e::ZePtr{ComplexF64},
                                          tauq::ZePtr{ComplexF64}, taup::ZePtr{ComplexF64},
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgeqrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklSgeqrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDgeqrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklDgeqrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgeqrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklCgeqrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZgeqrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklZgeqrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgeqrf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgeqrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          tau::ZePtr{ComplexF32},
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgeqrf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgeqrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{Cdouble}, lda::Int64,
                                          tau::ZePtr{Cdouble}, scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgeqrf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgeqrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{Cfloat}, lda::Int64, tau::ZePtr{Cfloat},
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgeqrf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgeqrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          tau::ZePtr{ComplexF64},
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgesvd_scratchpad_size(device_queue, jobu, jobvt, m, n, lda, ldu, ldvt)
    @ccall liboneapi_support.onemklSgesvd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobu::onemklJobsvd,
                                                          jobvt::onemklJobsvd, m::Int64,
                                                          n::Int64, lda::Int64, ldu::Int64,
                                                          ldvt::Int64)::Int64
end

function onemklDgesvd_scratchpad_size(device_queue, jobu, jobvt, m, n, lda, ldu, ldvt)
    @ccall liboneapi_support.onemklDgesvd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobu::onemklJobsvd,
                                                          jobvt::onemklJobsvd, m::Int64,
                                                          n::Int64, lda::Int64, ldu::Int64,
                                                          ldvt::Int64)::Int64
end

function onemklCgesvd_scratchpad_size(device_queue, jobu, jobvt, m, n, lda, ldu, ldvt)
    @ccall liboneapi_support.onemklCgesvd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobu::onemklJobsvd,
                                                          jobvt::onemklJobsvd, m::Int64,
                                                          n::Int64, lda::Int64, ldu::Int64,
                                                          ldvt::Int64)::Int64
end

function onemklZgesvd_scratchpad_size(device_queue, jobu, jobvt, m, n, lda, ldu, ldvt)
    @ccall liboneapi_support.onemklZgesvd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobu::onemklJobsvd,
                                                          jobvt::onemklJobsvd, m::Int64,
                                                          n::Int64, lda::Int64, ldu::Int64,
                                                          ldvt::Int64)::Int64
end

function onemklCgesvd(device_queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgesvd(device_queue::syclQueue_t, jobu::onemklJobsvd,
                                          jobvt::onemklJobsvd, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          s::ZePtr{Float32}, u::ZePtr{ComplexF32},
                                          ldu::Int64, vt::ZePtr{ComplexF32}, ldvt::Int64,
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgesvd(device_queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgesvd(device_queue::syclQueue_t, jobu::onemklJobsvd,
                                          jobvt::onemklJobsvd, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          s::ZePtr{Float64}, u::ZePtr{ComplexF64},
                                          ldu::Int64, vt::ZePtr{ComplexF64}, ldvt::Int64,
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgesvd(device_queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgesvd(device_queue::syclQueue_t, jobu::onemklJobsvd,
                                          jobvt::onemklJobsvd, m::Int64, n::Int64,
                                          a::ZePtr{Float64}, lda::Int64, s::ZePtr{Float64},
                                          u::ZePtr{Float64}, ldu::Int64, vt::ZePtr{Float64},
                                          ldvt::Int64, scratchpad::ZePtr{Float64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgesvd(device_queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgesvd(device_queue::syclQueue_t, jobu::onemklJobsvd,
                                          jobvt::onemklJobsvd, m::Int64, n::Int64,
                                          a::ZePtr{Float32}, lda::Int64, s::ZePtr{Float32},
                                          u::ZePtr{Float32}, ldu::Int64, vt::ZePtr{Float32},
                                          ldvt::Int64, scratchpad::ZePtr{Float32},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklSgetrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDgetrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklDgetrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgetrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklCgetrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZgetrf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklZgetrf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgetrf(device_queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgetrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgetrf(device_queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgetrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{Cdouble}, lda::Int64, ipiv::ZePtr{Int64},
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetrf(device_queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgetrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{Cfloat}, lda::Int64, ipiv::ZePtr{Int64},
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgetrf(device_queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgetrf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklSgetrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDgetrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklDgetrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCgetrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklCgetrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZgetrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklZgetrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCgetrf_batch(device_queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgetrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{ComplexF32}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{ComplexF32},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklDgetrf_batch(device_queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgetrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{Cdouble}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{Cdouble},
                                                scratchpad::Ptr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgetrf_batch(device_queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgetrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{Cfloat}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{Cfloat},
                                                scratchpad::Ptr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklZgetrf_batch(device_queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgetrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{ComplexF64}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{ComplexF64},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgetrfnp_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklSgetrfnp_scratchpad_size(device_queue::syclQueue_t,
                                                            m::Int64, n::Int64,
                                                            lda::Int64)::Int64
end

function onemklDgetrfnp_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklDgetrfnp_scratchpad_size(device_queue::syclQueue_t,
                                                            m::Int64, n::Int64,
                                                            lda::Int64)::Int64
end

function onemklCgetrfnp_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklCgetrfnp_scratchpad_size(device_queue::syclQueue_t,
                                                            m::Int64, n::Int64,
                                                            lda::Int64)::Int64
end

function onemklZgetrfnp_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklZgetrfnp_scratchpad_size(device_queue::syclQueue_t,
                                                            m::Int64, n::Int64,
                                                            lda::Int64)::Int64
end

function onemklCgetrfnp(device_queue, m, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgetrfnp(device_queue::syclQueue_t, m::Int64, n::Int64,
                                            a::Ptr{ComplexF32}, lda::Int64,
                                            scratchpad::Ptr{ComplexF32},
                                            scratchpad_size::Int64)::Cint
end

function onemklDgetrfnp(device_queue, m, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgetrfnp(device_queue::syclQueue_t, m::Int64, n::Int64,
                                            a::Ptr{Cdouble}, lda::Int64,
                                            scratchpad::Ptr{Cdouble},
                                            scratchpad_size::Int64)::Cint
end

function onemklSgetrfnp(device_queue, m, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgetrfnp(device_queue::syclQueue_t, m::Int64, n::Int64,
                                            a::Ptr{Cfloat}, lda::Int64,
                                            scratchpad::Ptr{Cfloat},
                                            scratchpad_size::Int64)::Cint
end

function onemklZgetrfnp(device_queue, m, n, a, lda, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgetrfnp(device_queue::syclQueue_t, m::Int64, n::Int64,
                                            a::Ptr{ComplexF32}, lda::Int64,
                                            scratchpad::Ptr{ComplexF32},
                                            scratchpad_size::Int64)::Cint
end

function onemklSgetrfnp_batch_scratchpad_size(device_queue, m, n, lda, stride_a, batch_size)
    @ccall liboneapi_support.onemklSgetrfnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  m::Int64, n::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklDgetrfnp_batch_scratchpad_size(device_queue, m, n, lda, stride_a, batch_size)
    @ccall liboneapi_support.onemklDgetrfnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  m::Int64, n::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklCgetrfnp_batch_scratchpad_size(device_queue, m, n, lda, stride_a, batch_size)
    @ccall liboneapi_support.onemklCgetrfnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  m::Int64, n::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklZgetrfnp_batch_scratchpad_size(device_queue, m, n, lda, stride_a, batch_size)
    @ccall liboneapi_support.onemklZgetrfnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  m::Int64, n::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklCgetrfnp_batch(device_queue, m, n, a, lda, stride_a, batch_size, scratchpad,
                              scratchpad_size)
    @ccall liboneapi_support.onemklCgetrfnp_batch(device_queue::syclQueue_t, m::Int64,
                                                  n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                                  stride_a::Int64, batch_size::Int64,
                                                  scratchpad::Ptr{ComplexF32},
                                                  scratchpad_size::Int64)::Cint
end

function onemklDgetrfnp_batch(device_queue, m, n, a, lda, stride_a, batch_size, scratchpad,
                              scratchpad_size)
    @ccall liboneapi_support.onemklDgetrfnp_batch(device_queue::syclQueue_t, m::Int64,
                                                  n::Int64, a::Ptr{Cdouble}, lda::Int64,
                                                  stride_a::Int64, batch_size::Int64,
                                                  scratchpad::Ptr{Cdouble},
                                                  scratchpad_size::Int64)::Cint
end

function onemklSgetrfnp_batch(device_queue, m, n, a, lda, stride_a, batch_size, scratchpad,
                              scratchpad_size)
    @ccall liboneapi_support.onemklSgetrfnp_batch(device_queue::syclQueue_t, m::Int64,
                                                  n::Int64, a::Ptr{Cfloat}, lda::Int64,
                                                  stride_a::Int64, batch_size::Int64,
                                                  scratchpad::Ptr{Cfloat},
                                                  scratchpad_size::Int64)::Cint
end

function onemklZgetrfnp_batch(device_queue, m, n, a, lda, stride_a, batch_size, scratchpad,
                              scratchpad_size)
    @ccall liboneapi_support.onemklZgetrfnp_batch(device_queue::syclQueue_t, m::Int64,
                                                  n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                                  stride_a::Int64, batch_size::Int64,
                                                  scratchpad::Ptr{ComplexF32},
                                                  scratchpad_size::Int64)::Cint
end

function onemklSgetri_scratchpad_size(device_queue, n, lda)
    @ccall liboneapi_support.onemklSgetri_scratchpad_size(device_queue::syclQueue_t,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklDgetri_scratchpad_size(device_queue, n, lda)
    @ccall liboneapi_support.onemklDgetri_scratchpad_size(device_queue::syclQueue_t,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklCgetri_scratchpad_size(device_queue, n, lda)
    @ccall liboneapi_support.onemklCgetri_scratchpad_size(device_queue::syclQueue_t,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklZgetri_scratchpad_size(device_queue, n, lda)
    @ccall liboneapi_support.onemklZgetri_scratchpad_size(device_queue::syclQueue_t,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklCgetri(device_queue, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgetri(device_queue::syclQueue_t, n::Int64,
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgetri(device_queue, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgetri(device_queue::syclQueue_t, n::Int64,
                                          a::ZePtr{Cdouble}, lda::Int64, ipiv::ZePtr{Int64},
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetri(device_queue, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgetri(device_queue::syclQueue_t, n::Int64,
                                          a::ZePtr{Cfloat}, lda::Int64, ipiv::ZePtr{Int64},
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgetri(device_queue, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgetri(device_queue::syclQueue_t, n::Int64,
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetrs_scratchpad_size(device_queue, trans, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklSgetrs_scratchpad_size(device_queue::syclQueue_t,
                                                          trans::onemklTranspose, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklDgetrs_scratchpad_size(device_queue, trans, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklDgetrs_scratchpad_size(device_queue::syclQueue_t,
                                                          trans::onemklTranspose, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklCgetrs_scratchpad_size(device_queue, trans, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklCgetrs_scratchpad_size(device_queue::syclQueue_t,
                                                          trans::onemklTranspose, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklZgetrs_scratchpad_size(device_queue, trans, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklZgetrs_scratchpad_size(device_queue::syclQueue_t,
                                                          trans::onemklTranspose, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklCgetrs(device_queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklCgetrs(device_queue::syclQueue_t, trans::onemklTranspose,
                                          n::Int64, nrhs::Int64, a::ZePtr{ComplexF32},
                                          lda::Int64, ipiv::ZePtr{Int64},
                                          b::ZePtr{ComplexF32}, ldb::Int64,
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgetrs(device_queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDgetrs(device_queue::syclQueue_t, trans::onemklTranspose,
                                          n::Int64, nrhs::Int64, a::ZePtr{Cdouble},
                                          lda::Int64, ipiv::ZePtr{Int64}, b::ZePtr{Cdouble},
                                          ldb::Int64, scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetrs(device_queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklSgetrs(device_queue::syclQueue_t, trans::onemklTranspose,
                                          n::Int64, nrhs::Int64, a::ZePtr{Cfloat},
                                          lda::Int64, ipiv::ZePtr{Int64}, b::ZePtr{Cfloat},
                                          ldb::Int64, scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgetrs(device_queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZgetrs(device_queue::syclQueue_t, trans::onemklTranspose,
                                          n::Int64, nrhs::Int64, a::ZePtr{ComplexF64},
                                          lda::Int64, ipiv::ZePtr{Int64},
                                          b::ZePtr{ComplexF64}, ldb::Int64,
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgetrs_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                            stride_ipiv, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklSgetrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                trans::onemklTranspose,
                                                                n::Int64, nrhs::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                ldb::Int64, stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDgetrs_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                            stride_ipiv, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklDgetrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                trans::onemklTranspose,
                                                                n::Int64, nrhs::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                ldb::Int64, stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCgetrs_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                            stride_ipiv, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklCgetrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                trans::onemklTranspose,
                                                                n::Int64, nrhs::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                ldb::Int64, stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZgetrs_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                            stride_ipiv, ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklZgetrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                trans::onemklTranspose,
                                                                n::Int64, nrhs::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                ldb::Int64, stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCgetrs_batch(device_queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                            stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklCgetrs_batch(device_queue::syclQueue_t,
                                                trans::onemklTranspose, n::Int64,
                                                nrhs::Int64, a::ZePtr{Ptr{ComplexF32}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::ZePtr{Ptr{ComplexF32}},
                                                stride_ipiv::Int64, b::Ptr{ComplexF32},
                                                ldb::Int64, stride_b::ZePtr{ComplexF32},
                                                batch_size::Int64,
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklDgetrs_batch(device_queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                            stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklDgetrs_batch(device_queue::syclQueue_t,
                                                trans::onemklTranspose, n::Int64,
                                                nrhs::Int64, a::ZePtr{Ptr{Cdouble}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::ZePtr{Ptr{Cdouble}},
                                                stride_ipiv::Int64, b::Ptr{Cdouble},
                                                ldb::Int64, stride_b::ZePtr{Cdouble},
                                                batch_size::Int64, scratchpad::Ptr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgetrs_batch(device_queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                            stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklSgetrs_batch(device_queue::syclQueue_t,
                                                trans::onemklTranspose, n::Int64,
                                                nrhs::Int64, a::ZePtr{Ptr{Cfloat}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::ZePtr{Ptr{Cfloat}},
                                                stride_ipiv::Int64, b::Ptr{Cfloat},
                                                ldb::Int64, stride_b::ZePtr{Cfloat},
                                                batch_size::Int64, scratchpad::Ptr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklZgetrs_batch(device_queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                            stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklZgetrs_batch(device_queue::syclQueue_t,
                                                trans::onemklTranspose, n::Int64,
                                                nrhs::Int64, a::ZePtr{Ptr{ComplexF64}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::ZePtr{Ptr{ComplexF64}},
                                                stride_ipiv::Int64, b::Ptr{ComplexF32},
                                                ldb::Int64, stride_b::ZePtr{ComplexF64},
                                                batch_size::Int64,
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgetrsnp_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                              ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklSgetrsnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  trans::onemklTranspose,
                                                                  n::Int64, nrhs::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  ldb::Int64,
                                                                  stride_b::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklDgetrsnp_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                              ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklDgetrsnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  trans::onemklTranspose,
                                                                  n::Int64, nrhs::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  ldb::Int64,
                                                                  stride_b::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklCgetrsnp_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                              ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklCgetrsnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  trans::onemklTranspose,
                                                                  n::Int64, nrhs::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  ldb::Int64,
                                                                  stride_b::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklZgetrsnp_batch_scratchpad_size(device_queue, trans, n, nrhs, lda, stride_a,
                                              ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklZgetrsnp_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                  trans::onemklTranspose,
                                                                  n::Int64, nrhs::Int64,
                                                                  lda::Int64,
                                                                  stride_a::Int64,
                                                                  ldb::Int64,
                                                                  stride_b::Int64,
                                                                  batch_size::Int64)::Int64
end

function onemklCgetrsnp_batch(device_queue, trans, n, nrhs, a, lda, stride_a, b, ldb,
                              stride_b, batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgetrsnp_batch(device_queue::syclQueue_t,
                                                  trans::onemklTranspose, n::Int64,
                                                  nrhs::Int64, a::Ptr{ComplexF32},
                                                  lda::Int64, stride_a::Int64,
                                                  b::Ptr{ComplexF32}, ldb::Int64,
                                                  stride_b::Int64, batch_size::Int64,
                                                  scratchpad::Ptr{ComplexF32},
                                                  scratchpad_size::Int64)::Cint
end

function onemklDgetrsnp_batch(device_queue, trans, n, nrhs, a, lda, stride_a, b, ldb,
                              stride_b, batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgetrsnp_batch(device_queue::syclQueue_t,
                                                  trans::onemklTranspose, n::Int64,
                                                  nrhs::Int64, a::Ptr{Cdouble}, lda::Int64,
                                                  stride_a::Int64, b::Ptr{Cdouble},
                                                  ldb::Int64, stride_b::Int64,
                                                  batch_size::Int64,
                                                  scratchpad::Ptr{Cdouble},
                                                  scratchpad_size::Int64)::Cint
end

function onemklSgetrsnp_batch(device_queue, trans, n, nrhs, a, lda, stride_a, b, ldb,
                              stride_b, batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgetrsnp_batch(device_queue::syclQueue_t,
                                                  trans::onemklTranspose, n::Int64,
                                                  nrhs::Int64, a::Ptr{Cfloat}, lda::Int64,
                                                  stride_a::Int64, b::Ptr{Cfloat},
                                                  ldb::Int64, stride_b::Int64,
                                                  batch_size::Int64,
                                                  scratchpad::Ptr{Cfloat},
                                                  scratchpad_size::Int64)::Cint
end

function onemklZgetrsnp_batch(device_queue, trans, n, nrhs, a, lda, stride_a, b, ldb,
                              stride_b, batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgetrsnp_batch(device_queue::syclQueue_t,
                                                  trans::onemklTranspose, n::Int64,
                                                  nrhs::Int64, a::Ptr{ComplexF32},
                                                  lda::Int64, stride_a::Int64,
                                                  b::Ptr{ComplexF32}, ldb::Int64,
                                                  stride_b::Int64, batch_size::Int64,
                                                  scratchpad::Ptr{ComplexF32},
                                                  scratchpad_size::Int64)::Cint
end

function onemklCheev_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklCheev_scratchpad_size(device_queue::syclQueue_t,
                                                         jobz::onemklCompz,
                                                         uplo::onemklUplo, n::Int64,
                                                         lda::Int64)::Int64
end

function onemklZheev_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklZheev_scratchpad_size(device_queue::syclQueue_t,
                                                         jobz::onemklCompz,
                                                         uplo::onemklUplo, n::Int64,
                                                         lda::Int64)::Int64
end

function onemklCheev(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCheev(device_queue::syclQueue_t, jobz::onemklCompz,
                                         uplo::onemklUplo, n::Int64, a::Ptr{ComplexF32},
                                         lda::Int64, w::Ptr{Cfloat},
                                         scratchpad::Ptr{ComplexF32},
                                         scratchpad_size::Int64)::Cint
end

function onemklZheev(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZheev(device_queue::syclQueue_t, jobz::onemklCompz,
                                         uplo::onemklUplo, n::Int64, a::Ptr{ComplexF32},
                                         lda::Int64, w::Ptr{Cdouble},
                                         scratchpad::Ptr{ComplexF32},
                                         scratchpad_size::Int64)::Cint
end

function onemklCheevd_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklCheevd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobz::onemklJob, uplo::onemklUplo,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklZheevd_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklZheevd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobz::onemklJob, uplo::onemklUplo,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklCheevd(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCheevd(device_queue::syclQueue_t, jobz::onemklJob,
                                          uplo::onemklUplo, n::Int64, a::ZePtr{ComplexF32},
                                          lda::Int64, w::ZePtr{Float32},
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZheevd(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZheevd(device_queue::syclQueue_t, jobz::onemklJob,
                                          uplo::onemklUplo, n::Int64, a::ZePtr{ComplexF64},
                                          lda::Int64, w::ZePtr{Float64},
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklChegvd_scratchpad_size(device_queue, itype, jobz, uplo, n, lda, ldb)
    @ccall liboneapi_support.onemklChegvd_scratchpad_size(device_queue::syclQueue_t,
                                                          itype::Int64, jobz::onemklJob,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, ldb::Int64)::Int64
end

function onemklZhegvd_scratchpad_size(device_queue, itype, jobz, uplo, n, lda, ldb)
    @ccall liboneapi_support.onemklZhegvd_scratchpad_size(device_queue::syclQueue_t,
                                                          itype::Int64, jobz::onemklJob,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, ldb::Int64)::Int64
end

function onemklChegvd(device_queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklChegvd(device_queue::syclQueue_t, itype::Int64,
                                          jobz::onemklJob, uplo::onemklUplo, n::Int64,
                                          a::ZePtr{ComplexF32}, lda::Int64,
                                          b::ZePtr{ComplexF32}, ldb::Int64,
                                          w::ZePtr{Float32}, scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZhegvd(device_queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZhegvd(device_queue::syclQueue_t, itype::Int64,
                                          jobz::onemklJob, uplo::onemklUplo, n::Int64,
                                          a::ZePtr{ComplexF64}, lda::Int64,
                                          b::ZePtr{ComplexF64}, ldb::Int64,
                                          w::ZePtr{Float64}, scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklChetrd_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklChetrd_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZhetrd_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklZhetrd_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklChetrd(device_queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklChetrd(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          d::Ptr{Cfloat}, e::Ptr{Cfloat},
                                          tau::Ptr{ComplexF32}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZhetrd(device_queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZhetrd(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          d::Ptr{Cdouble}, e::Ptr{Cdouble},
                                          tau::Ptr{ComplexF32}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklChetrf(device_queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklChetrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          ipiv::Ptr{Int64}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZhetrf(device_queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZhetrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          ipiv::Ptr{Int64}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklChetrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklChetrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZhetrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklZhetrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklSorgbr(device_queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSorgbr(device_queue::syclQueue_t, vec::onemklGenerate,
                                          m::Int64, n::Int64, k::Int64, a::Ptr{Cfloat},
                                          lda::Int64, tau::Ptr{Cfloat},
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDorgbr(device_queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDorgbr(device_queue::syclQueue_t, vec::onemklGenerate,
                                          m::Int64, n::Int64, k::Int64, a::Ptr{Cdouble},
                                          lda::Int64, tau::Ptr{Cdouble},
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSorgbr_scratchpad_size(device_queue, vect, m, n, k, lda)
    @ccall liboneapi_support.onemklSorgbr_scratchpad_size(device_queue::syclQueue_t,
                                                          vect::onemklGenerate, m::Int64,
                                                          n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklDorgbr_scratchpad_size(device_queue, vect, m, n, k, lda)
    @ccall liboneapi_support.onemklDorgbr_scratchpad_size(device_queue::syclQueue_t,
                                                          vect::onemklGenerate, m::Int64,
                                                          n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklSorgqr_scratchpad_size(device_queue, m, n, k, lda)
    @ccall liboneapi_support.onemklSorgqr_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklDorgqr_scratchpad_size(device_queue, m, n, k, lda)
    @ccall liboneapi_support.onemklDorgqr_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklDorgqr(device_queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDorgqr(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                          tau::ZePtr{Cdouble}, scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSorgqr(device_queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSorgqr(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                          tau::ZePtr{Cfloat}, scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklSormqr_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklSormqr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklDormqr_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklDormqr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklDormqr(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDormqr(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                          tau::ZePtr{Cdouble}, c::ZePtr{Cdouble},
                                          ldc::Int64, scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSormqr(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklSormqr(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                          tau::ZePtr{Cfloat}, c::ZePtr{Cfloat}, ldc::Int64,
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsteqr_scratchpad_size(device_queue, compz, n, ldz)
    @ccall liboneapi_support.onemklSsteqr_scratchpad_size(device_queue::syclQueue_t,
                                                          compz::onemklCompz, n::Int64,
                                                          ldz::Int64)::Int64
end

function onemklDsteqr_scratchpad_size(device_queue, compz, n, ldz)
    @ccall liboneapi_support.onemklDsteqr_scratchpad_size(device_queue::syclQueue_t,
                                                          compz::onemklCompz, n::Int64,
                                                          ldz::Int64)::Int64
end

function onemklCsteqr_scratchpad_size(device_queue, compz, n, ldz)
    @ccall liboneapi_support.onemklCsteqr_scratchpad_size(device_queue::syclQueue_t,
                                                          compz::onemklCompz, n::Int64,
                                                          ldz::Int64)::Int64
end

function onemklZsteqr_scratchpad_size(device_queue, compz, n, ldz)
    @ccall liboneapi_support.onemklZsteqr_scratchpad_size(device_queue::syclQueue_t,
                                                          compz::onemklCompz, n::Int64,
                                                          ldz::Int64)::Int64
end

function onemklCsteqr(device_queue, compz, n, d, e, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCsteqr(device_queue::syclQueue_t, compz::onemklCompz,
                                          n::Int64, d::Ptr{Cfloat}, e::Ptr{Cfloat},
                                          z::Ptr{ComplexF32}, ldz::Int64,
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDsteqr(device_queue, compz, n, d, e, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsteqr(device_queue::syclQueue_t, compz::onemklCompz,
                                          n::Int64, d::Ptr{Cdouble}, e::Ptr{Cdouble},
                                          z::Ptr{Cdouble}, ldz::Int64,
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsteqr(device_queue, compz, n, d, e, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsteqr(device_queue::syclQueue_t, compz::onemklCompz,
                                          n::Int64, d::Ptr{Cfloat}, e::Ptr{Cfloat},
                                          z::Ptr{Cfloat}, ldz::Int64,
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklZsteqr(device_queue, compz, n, d, e, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZsteqr(device_queue::syclQueue_t, compz::onemklCompz,
                                          n::Int64, d::Ptr{Cdouble}, e::Ptr{Cdouble},
                                          z::Ptr{ComplexF32}, ldz::Int64,
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsyev_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklSsyev_scratchpad_size(device_queue::syclQueue_t,
                                                         jobz::onemklCompz,
                                                         uplo::onemklUplo, n::Int64,
                                                         lda::Int64)::Int64
end

function onemklDsyev_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklDsyev_scratchpad_size(device_queue::syclQueue_t,
                                                         jobz::onemklCompz,
                                                         uplo::onemklUplo, n::Int64,
                                                         lda::Int64)::Int64
end

function onemklDsyev(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsyev(device_queue::syclQueue_t, jobz::onemklCompz,
                                         uplo::onemklUplo, n::Int64, a::Ptr{Cdouble},
                                         lda::Int64, w::Ptr{Cdouble},
                                         scratchpad::Ptr{Cdouble},
                                         scratchpad_size::Int64)::Cint
end

function onemklSsyev(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsyev(device_queue::syclQueue_t, jobz::onemklCompz,
                                         uplo::onemklUplo, n::Int64, a::Ptr{Cfloat},
                                         lda::Int64, w::Ptr{Cfloat},
                                         scratchpad::Ptr{Cfloat},
                                         scratchpad_size::Int64)::Cint
end

function onemklSsyevd_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklSsyevd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobz::onemklJob, uplo::onemklUplo,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklDsyevd_scratchpad_size(device_queue, jobz, uplo, n, lda)
    @ccall liboneapi_support.onemklDsyevd_scratchpad_size(device_queue::syclQueue_t,
                                                          jobz::onemklJob, uplo::onemklUplo,
                                                          n::Int64, lda::Int64)::Int64
end

function onemklDsyevd(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsyevd(device_queue::syclQueue_t, jobz::onemklJob,
                                          uplo::onemklUplo, n::Int64, a::ZePtr{Cdouble},
                                          lda::Int64, w::ZePtr{Cdouble},
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsyevd(device_queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsyevd(device_queue::syclQueue_t, jobz::onemklJob,
                                          uplo::onemklUplo, n::Int64, a::ZePtr{Cfloat},
                                          lda::Int64, w::ZePtr{Cfloat},
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsyevx_scratchpad_size(device_queue, jobz, range, uplo, n, lda, vl, vu, il,
                                      iu, abstol, ldz)
    @ccall liboneapi_support.onemklSsyevx_scratchpad_size(device_queue::syclQueue_t,
                                                          jobz::onemklCompz,
                                                          range::onemklRangev,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, vl::Cfloat,
                                                          vu::Cfloat, il::Int64, iu::Int64,
                                                          abstol::Cfloat, ldz::Int64)::Int64
end

function onemklDsyevx_scratchpad_size(device_queue, jobz, range, uplo, n, lda, vl, vu, il,
                                      iu, abstol, ldz)
    @ccall liboneapi_support.onemklDsyevx_scratchpad_size(device_queue::syclQueue_t,
                                                          jobz::onemklCompz,
                                                          range::onemklRangev,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, vl::Cdouble,
                                                          vu::Cdouble, il::Int64, iu::Int64,
                                                          abstol::Cdouble,
                                                          ldz::Int64)::Int64
end

function onemklDsyevx(device_queue, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m,
                      w, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsyevx(device_queue::syclQueue_t, jobz::onemklCompz,
                                          range::onemklRangev, uplo::onemklUplo, n::Int64,
                                          a::Ptr{Cdouble}, lda::Int64, vl::Cdouble,
                                          vu::Cdouble, il::Int64, iu::Int64,
                                          abstol::Cdouble, m::Ptr{Int64}, w::Ptr{Cdouble},
                                          z::Ptr{Cdouble}, ldz::Int64,
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsyevx(device_queue, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m,
                      w, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsyevx(device_queue::syclQueue_t, jobz::onemklCompz,
                                          range::onemklRangev, uplo::onemklUplo, n::Int64,
                                          a::Ptr{Cfloat}, lda::Int64, vl::Cfloat,
                                          vu::Cfloat, il::Int64, iu::Int64, abstol::Cfloat,
                                          m::Ptr{Int64}, w::Ptr{Cfloat}, z::Ptr{Cfloat},
                                          ldz::Int64, scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsygvd_scratchpad_size(device_queue, itype, jobz, uplo, n, lda, ldb)
    @ccall liboneapi_support.onemklSsygvd_scratchpad_size(device_queue::syclQueue_t,
                                                          itype::Int64, jobz::onemklJob,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, ldb::Int64)::Int64
end

function onemklDsygvd_scratchpad_size(device_queue, itype, jobz, uplo, n, lda, ldb)
    @ccall liboneapi_support.onemklDsygvd_scratchpad_size(device_queue::syclQueue_t,
                                                          itype::Int64, jobz::onemklJob,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, ldb::Int64)::Int64
end

function onemklDsygvd(device_queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDsygvd(device_queue::syclQueue_t, itype::Int64,
                                          jobz::onemklJob, uplo::onemklUplo, n::Int64,
                                          a::ZePtr{Cdouble}, lda::Int64, b::ZePtr{Cdouble},
                                          ldb::Int64, w::ZePtr{Cdouble},
                                          scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsygvd(device_queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklSsygvd(device_queue::syclQueue_t, itype::Int64,
                                          jobz::onemklJob, uplo::onemklUplo, n::Int64,
                                          a::ZePtr{Cfloat}, lda::Int64, b::ZePtr{Cfloat},
                                          ldb::Int64, w::ZePtr{Cfloat},
                                          scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsygvx_scratchpad_size(device_queue, itype, jobz, range, uplo, n, lda, ldb,
                                      vl, vu, il, iu, abstol, ldz)
    @ccall liboneapi_support.onemklSsygvx_scratchpad_size(device_queue::syclQueue_t,
                                                          itype::Int64, jobz::onemklCompz,
                                                          range::onemklRangev,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, ldb::Int64,
                                                          vl::Cfloat, vu::Cfloat, il::Int64,
                                                          iu::Int64, abstol::Cfloat,
                                                          ldz::Int64)::Int64
end

function onemklDsygvx_scratchpad_size(device_queue, itype, jobz, range, uplo, n, lda, ldb,
                                      vl, vu, il, iu, abstol, ldz)
    @ccall liboneapi_support.onemklDsygvx_scratchpad_size(device_queue::syclQueue_t,
                                                          itype::Int64, jobz::onemklCompz,
                                                          range::onemklRangev,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64, ldb::Int64,
                                                          vl::Cdouble, vu::Cdouble,
                                                          il::Int64, iu::Int64,
                                                          abstol::Cdouble,
                                                          ldz::Int64)::Int64
end

function onemklDsygvx(device_queue, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il,
                      iu, abstol, m, w, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsygvx(device_queue::syclQueue_t, itype::Int64,
                                          jobz::onemklCompz, range::onemklRangev,
                                          uplo::onemklUplo, n::Int64, a::Ptr{Cdouble},
                                          lda::Int64, b::Ptr{Cdouble}, ldb::Int64,
                                          vl::Cdouble, vu::Cdouble, il::Int64, iu::Int64,
                                          abstol::Cdouble, m::Ptr{Int64}, w::Ptr{Cdouble},
                                          z::Ptr{Cdouble}, ldz::Int64,
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsygvx(device_queue, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il,
                      iu, abstol, m, w, z, ldz, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsygvx(device_queue::syclQueue_t, itype::Int64,
                                          jobz::onemklCompz, range::onemklRangev,
                                          uplo::onemklUplo, n::Int64, a::Ptr{Cfloat},
                                          lda::Int64, b::Ptr{Cfloat}, ldb::Int64,
                                          vl::Cfloat, vu::Cfloat, il::Int64, iu::Int64,
                                          abstol::Cfloat, m::Ptr{Int64}, w::Ptr{Cfloat},
                                          z::Ptr{Cfloat}, ldz::Int64,
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsytrd_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklSsytrd_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDsytrd_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklDsytrd_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDsytrd(device_queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsytrd(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{Cdouble}, lda::Int64,
                                          d::Ptr{Cdouble}, e::Ptr{Cdouble},
                                          tau::Ptr{Cdouble}, scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsytrd(device_queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsytrd(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{Cfloat}, lda::Int64,
                                          d::Ptr{Cfloat}, e::Ptr{Cfloat}, tau::Ptr{Cfloat},
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklStrtrs_scratchpad_size(device_queue, uplo, trans, diag, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklStrtrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose,
                                                          diag::onemklDiag, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklDtrtrs_scratchpad_size(device_queue, uplo, trans, diag, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklDtrtrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose,
                                                          diag::onemklDiag, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklCtrtrs_scratchpad_size(device_queue, uplo, trans, diag, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklCtrtrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose,
                                                          diag::onemklDiag, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklZtrtrs_scratchpad_size(device_queue, uplo, trans, diag, n, nrhs, lda, ldb)
    @ccall liboneapi_support.onemklZtrtrs_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose,
                                                          diag::onemklDiag, n::Int64,
                                                          nrhs::Int64, lda::Int64,
                                                          ldb::Int64)::Int64
end

function onemklCtrtrs(device_queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklCtrtrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          trans::onemklTranspose, diag::onemklDiag,
                                          n::Int64, nrhs::Int64, a::Ptr{ComplexF32},
                                          lda::Int64, b::Ptr{ComplexF32}, ldb::Int64,
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklDtrtrs(device_queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDtrtrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          trans::onemklTranspose, diag::onemklDiag,
                                          n::Int64, nrhs::Int64, a::Ptr{Cdouble},
                                          lda::Int64, b::Ptr{Cdouble}, ldb::Int64,
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklStrtrs(device_queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklStrtrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          trans::onemklTranspose, diag::onemklDiag,
                                          n::Int64, nrhs::Int64, a::Ptr{Cfloat}, lda::Int64,
                                          b::Ptr{Cfloat}, ldb::Int64,
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklZtrtrs(device_queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZtrtrs(device_queue::syclQueue_t, uplo::onemklUplo,
                                          trans::onemklTranspose, diag::onemklDiag,
                                          n::Int64, nrhs::Int64, a::Ptr{ComplexF32},
                                          lda::Int64, b::Ptr{ComplexF32}, ldb::Int64,
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklCungbr(device_queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCungbr(device_queue::syclQueue_t, vec::onemklGenerate,
                                          m::Int64, n::Int64, k::Int64, a::Ptr{ComplexF32},
                                          lda::Int64, tau::Ptr{ComplexF32},
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZungbr(device_queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZungbr(device_queue::syclQueue_t, vec::onemklGenerate,
                                          m::Int64, n::Int64, k::Int64, a::Ptr{ComplexF32},
                                          lda::Int64, tau::Ptr{ComplexF32},
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklCungbr_scratchpad_size(device_queue, vect, m, n, k, lda)
    @ccall liboneapi_support.onemklCungbr_scratchpad_size(device_queue::syclQueue_t,
                                                          vect::onemklGenerate, m::Int64,
                                                          n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklZungbr_scratchpad_size(device_queue, vect, m, n, k, lda)
    @ccall liboneapi_support.onemklZungbr_scratchpad_size(device_queue::syclQueue_t,
                                                          vect::onemklGenerate, m::Int64,
                                                          n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklCungqr_scratchpad_size(device_queue, m, n, k, lda)
    @ccall liboneapi_support.onemklCungqr_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklZungqr_scratchpad_size(device_queue, m, n, k, lda)
    @ccall liboneapi_support.onemklZungqr_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64, k::Int64,
                                                          lda::Int64)::Int64
end

function onemklCungqr(device_queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCungqr(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                          tau::ZePtr{ComplexF32},
                                          scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZungqr(device_queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZungqr(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                          tau::ZePtr{ComplexF64},
                                          scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklCunmqr_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklCunmqr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklZunmqr_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklZunmqr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklCunmqr(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklCunmqr(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                          tau::ZePtr{ComplexF32}, c::ZePtr{ComplexF32},
                                          ldc::Int64, scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZunmqr(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZunmqr(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                          tau::ZePtr{ComplexF64}, c::ZePtr{ComplexF64},
                                          ldc::Int64, scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgerqf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgerqf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::Ptr{Cfloat}, lda::Int64, tau::Ptr{Cfloat},
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDgerqf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgerqf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::Ptr{Cdouble}, lda::Int64, tau::Ptr{Cdouble},
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklCgerqf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgerqf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::Ptr{ComplexF32}, lda::Int64,
                                          tau::Ptr{ComplexF32}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZgerqf(device_queue, m, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgerqf(device_queue::syclQueue_t, m::Int64, n::Int64,
                                          a::Ptr{ComplexF32}, lda::Int64,
                                          tau::Ptr{ComplexF32}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklSgerqf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklSgerqf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDgerqf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklDgerqf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCgerqf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklCgerqf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZgerqf_scratchpad_size(device_queue, m, n, lda)
    @ccall liboneapi_support.onemklZgerqf_scratchpad_size(device_queue::syclQueue_t,
                                                          m::Int64, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklSormrq(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklSormrq(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::Ptr{Cfloat}, lda::Int64,
                                          tau::Ptr{Cfloat}, c::Ptr{Cfloat}, ldc::Int64,
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDormrq(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklDormrq(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::Ptr{Cdouble}, lda::Int64,
                                          tau::Ptr{Cdouble}, c::Ptr{Cdouble}, ldc::Int64,
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSormrq_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklSormrq_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklDormrq_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklDormrq_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklCunmrq(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklCunmrq(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          tau::Ptr{ComplexF32}, c::Ptr{ComplexF32},
                                          ldc::Int64, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZunmrq(device_queue, side, trans, m, n, k, a, lda, tau, c, ldc, scratchpad,
                      scratchpad_size)
    @ccall liboneapi_support.onemklZunmrq(device_queue::syclQueue_t, side::onemklSide,
                                          trans::onemklTranspose, m::Int64, n::Int64,
                                          k::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          tau::Ptr{ComplexF32}, c::Ptr{ComplexF32},
                                          ldc::Int64, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklCunmrq_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklCunmrq_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklZunmrq_scratchpad_size(device_queue, side, trans, m, n, k, lda, ldc)
    @ccall liboneapi_support.onemklZunmrq_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, k::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklSsytrf(device_queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSsytrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{Cfloat}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDsytrf(device_queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDsytrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{Cdouble}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklCsytrf(device_queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCsytrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{ComplexF32}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZsytrf(device_queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZsytrf(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::ZePtr{ComplexF64}, lda::Int64,
                                          ipiv::ZePtr{Int64}, scratchpad::ZePtr{ComplexF64},
                                          scratchpad_size::Int64)::Cint
end

function onemklSsytrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklSsytrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDsytrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklDsytrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCsytrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklCsytrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZsytrf_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklZsytrf_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklSorgtr(device_queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSorgtr(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{Cfloat}, lda::Int64,
                                          tau::Ptr{Cfloat}, scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDorgtr(device_queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDorgtr(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{Cdouble}, lda::Int64,
                                          tau::Ptr{Cdouble}, scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSorgtr_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklSorgtr_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklDorgtr_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklDorgtr_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklCungtr(device_queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCungtr(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          tau::Ptr{ComplexF32}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZungtr(device_queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZungtr(device_queue::syclQueue_t, uplo::onemklUplo,
                                          n::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                          tau::Ptr{ComplexF32}, scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklCungtr_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklCungtr_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklZungtr_scratchpad_size(device_queue, uplo, n, lda)
    @ccall liboneapi_support.onemklZungtr_scratchpad_size(device_queue::syclQueue_t,
                                                          uplo::onemklUplo, n::Int64,
                                                          lda::Int64)::Int64
end

function onemklSormtr(device_queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSormtr(device_queue::syclQueue_t, side::onemklSide,
                                          uplo::onemklUplo, trans::onemklTranspose,
                                          m::Int64, n::Int64, a::Ptr{Cfloat}, lda::Int64,
                                          tau::Ptr{Cfloat}, c::Ptr{Cfloat}, ldc::Int64,
                                          scratchpad::Ptr{Cfloat},
                                          scratchpad_size::Int64)::Cint
end

function onemklDormtr(device_queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDormtr(device_queue::syclQueue_t, side::onemklSide,
                                          uplo::onemklUplo, trans::onemklTranspose,
                                          m::Int64, n::Int64, a::Ptr{Cdouble}, lda::Int64,
                                          tau::Ptr{Cdouble}, c::Ptr{Cdouble}, ldc::Int64,
                                          scratchpad::Ptr{Cdouble},
                                          scratchpad_size::Int64)::Cint
end

function onemklSormtr_scratchpad_size(device_queue, side, uplo, trans, m, n, lda, ldc)
    @ccall liboneapi_support.onemklSormtr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklDormtr_scratchpad_size(device_queue, side, uplo, trans, m, n, lda, ldc)
    @ccall liboneapi_support.onemklDormtr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklCunmtr(device_queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCunmtr(device_queue::syclQueue_t, side::onemklSide,
                                          uplo::onemklUplo, trans::onemklTranspose,
                                          m::Int64, n::Int64, a::Ptr{ComplexF32},
                                          lda::Int64, tau::Ptr{ComplexF32},
                                          c::Ptr{ComplexF32}, ldc::Int64,
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklZunmtr(device_queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                      scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZunmtr(device_queue::syclQueue_t, side::onemklSide,
                                          uplo::onemklUplo, trans::onemklTranspose,
                                          m::Int64, n::Int64, a::Ptr{ComplexF32},
                                          lda::Int64, tau::Ptr{ComplexF32},
                                          c::Ptr{ComplexF32}, ldc::Int64,
                                          scratchpad::Ptr{ComplexF32},
                                          scratchpad_size::Int64)::Cint
end

function onemklCunmtr_scratchpad_size(device_queue, side, uplo, trans, m, n, lda, ldc)
    @ccall liboneapi_support.onemklCunmtr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklZunmtr_scratchpad_size(device_queue, side, uplo, trans, m, n, lda, ldc)
    @ccall liboneapi_support.onemklZunmtr_scratchpad_size(device_queue::syclQueue_t,
                                                          side::onemklSide,
                                                          uplo::onemklUplo,
                                                          trans::onemklTranspose, m::Int64,
                                                          n::Int64, lda::Int64,
                                                          ldc::Int64)::Int64
end

function onemklSpotrf_batch(device_queue, uplo, n, a, lda, stride_a, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklSpotrf_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, a::ZePtr{Ptr{Cfloat}}, lda::Int64,
                                                stride_a::Int64, batch_size::Int64,
                                                scratchpad::ZePtr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklDpotrf_batch(device_queue, uplo, n, a, lda, stride_a, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklDpotrf_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, a::ZePtr{Ptr{Cdouble}},
                                                lda::Int64, stride_a::Int64,
                                                batch_size::Int64,
                                                scratchpad::ZePtr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklCpotrf_batch(device_queue, uplo, n, a, lda, stride_a, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklCpotrf_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, a::ZePtr{Ptr{ComplexF32}},
                                                lda::Int64, stride_a::Int64,
                                                batch_size::Int64,
                                                scratchpad::ZePtr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklZpotrf_batch(device_queue, uplo, n, a, lda, stride_a, batch_size, scratchpad,
                            scratchpad_size)
    @ccall liboneapi_support.onemklZpotrf_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, a::ZePtr{Ptr{ComplexF64}},
                                                lda::Int64, stride_a::Int64,
                                                batch_size::Int64,
                                                scratchpad::ZePtr{ComplexF64},
                                                scratchpad_size::Int64)::Cint
end

function onemklSpotrs_batch(device_queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSpotrs_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, nrhs::Int64,
                                                a::ZePtr{Ptr{Cfloat}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Cfloat}},
                                                b::Ptr{Cfloat}, ldb::Int64, stride_b::Int64,
                                                batch_size::ZePtr{Cfloat},
                                                scratchpad::Ptr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklDpotrs_batch(device_queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDpotrs_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, nrhs::Int64,
                                                a::ZePtr{Ptr{Cdouble}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Cdouble}},
                                                b::Ptr{Cdouble}, ldb::Int64,
                                                stride_b::Int64, batch_size::ZePtr{Cdouble},
                                                scratchpad::Ptr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklCpotrs_batch(device_queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCpotrs_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, nrhs::Int64,
                                                a::ZePtr{Ptr{ComplexF32}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{ComplexF32}},
                                                b::Ptr{ComplexF32}, ldb::Int64,
                                                stride_b::Int64,
                                                batch_size::ZePtr{ComplexF32},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklZpotrs_batch(device_queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZpotrs_batch(device_queue::syclQueue_t, uplo::onemklUplo,
                                                n::Int64, nrhs::Int64,
                                                a::ZePtr{Ptr{ComplexF64}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{ComplexF64}},
                                                b::Ptr{ComplexF32}, ldb::Int64,
                                                stride_b::Int64,
                                                batch_size::ZePtr{ComplexF64},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgeqrf_batch(device_queue, m, n, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgeqrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{Cfloat}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Cfloat}},
                                                tau::Ptr{Cfloat}, stride_tau::Int64,
                                                batch_size::ZePtr{Cfloat},
                                                scratchpad::Ptr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklDgeqrf_batch(device_queue, m, n, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgeqrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{Cdouble}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Cdouble}},
                                                tau::Ptr{Cdouble}, stride_tau::Int64,
                                                batch_size::ZePtr{Cdouble},
                                                scratchpad::Ptr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklCgeqrf_batch(device_queue, m, n, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgeqrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{ComplexF32}},
                                                lda::Int64,
                                                stride_a::ZePtr{Ptr{ComplexF32}},
                                                tau::Ptr{ComplexF32}, stride_tau::Int64,
                                                batch_size::ZePtr{ComplexF32},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklZgeqrf_batch(device_queue, m, n, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgeqrf_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, a::ZePtr{Ptr{ComplexF64}},
                                                lda::Int64,
                                                stride_a::ZePtr{Ptr{ComplexF64}},
                                                tau::Ptr{ComplexF32}, stride_tau::Int64,
                                                batch_size::ZePtr{ComplexF64},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklSorgqr_batch(device_queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSorgqr_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, k::Int64, a::ZePtr{Ptr{Cfloat}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Cfloat}},
                                                tau::Ptr{Cfloat}, stride_tau::Int64,
                                                batch_size::ZePtr{Cfloat},
                                                scratchpad::Ptr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklDorgqr_batch(device_queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDorgqr_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, k::Int64, a::ZePtr{Ptr{Cdouble}},
                                                lda::Int64, stride_a::ZePtr{Ptr{Cdouble}},
                                                tau::Ptr{Cdouble}, stride_tau::Int64,
                                                batch_size::ZePtr{Cdouble},
                                                scratchpad::Ptr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklCungqr_batch(device_queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCungqr_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, k::Int64,
                                                a::ZePtr{Ptr{ComplexF32}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{ComplexF32}},
                                                tau::Ptr{ComplexF32}, stride_tau::Int64,
                                                batch_size::ZePtr{ComplexF32},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklZungqr_batch(device_queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZungqr_batch(device_queue::syclQueue_t, m::Int64,
                                                n::Int64, k::Int64,
                                                a::ZePtr{Ptr{ComplexF64}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{ComplexF64}},
                                                tau::Ptr{ComplexF32}, stride_tau::Int64,
                                                batch_size::ZePtr{ComplexF64},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgetri_batch(device_queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgetri_batch(device_queue::syclQueue_t, n::Int64,
                                                a::ZePtr{Ptr{Cfloat}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{Cfloat},
                                                scratchpad::Ptr{Cfloat},
                                                scratchpad_size::Int64)::Cint
end

function onemklDgetri_batch(device_queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgetri_batch(device_queue::syclQueue_t, n::Int64,
                                                a::ZePtr{Ptr{Cdouble}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{Cdouble},
                                                scratchpad::Ptr{Cdouble},
                                                scratchpad_size::Int64)::Cint
end

function onemklCgetri_batch(device_queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgetri_batch(device_queue::syclQueue_t, n::Int64,
                                                a::ZePtr{Ptr{ComplexF32}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{ComplexF32},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklZgetri_batch(device_queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                            batch_size, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgetri_batch(device_queue::syclQueue_t, n::Int64,
                                                a::ZePtr{Ptr{ComplexF64}}, lda::Int64,
                                                stride_a::ZePtr{Ptr{Int64}},
                                                ipiv::Ptr{Int64}, stride_ipiv::Int64,
                                                batch_size::ZePtr{ComplexF64},
                                                scratchpad::Ptr{ComplexF32},
                                                scratchpad_size::Int64)::Cint
end

function onemklSgels_batch(device_queue, trans, m, n, nrhs, a, lda, stridea, b, ldb,
                           strideb, batchsize, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklSgels_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               nrhs::Int64, a::Ptr{Cfloat}, lda::Int64,
                                               stridea::Int64, b::Ptr{Cfloat}, ldb::Int64,
                                               strideb::Int64, batchsize::Int64,
                                               scratchpad::Ptr{Cfloat},
                                               scratchpad_size::Int64)::Cint
end

function onemklDgels_batch(device_queue, trans, m, n, nrhs, a, lda, stridea, b, ldb,
                           strideb, batchsize, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklDgels_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               nrhs::Int64, a::Ptr{Cdouble}, lda::Int64,
                                               stridea::Int64, b::Ptr{Cdouble}, ldb::Int64,
                                               strideb::Int64, batchsize::Int64,
                                               scratchpad::Ptr{Cdouble},
                                               scratchpad_size::Int64)::Cint
end

function onemklCgels_batch(device_queue, trans, m, n, nrhs, a, lda, stridea, b, ldb,
                           strideb, batchsize, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklCgels_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               nrhs::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                               stridea::Int64, b::Ptr{ComplexF32},
                                               ldb::Int64, strideb::Int64, batchsize::Int64,
                                               scratchpad::Ptr{ComplexF32},
                                               scratchpad_size::Int64)::Cint
end

function onemklZgels_batch(device_queue, trans, m, n, nrhs, a, lda, stridea, b, ldb,
                           strideb, batchsize, scratchpad, scratchpad_size)
    @ccall liboneapi_support.onemklZgels_batch(device_queue::syclQueue_t,
                                               trans::onemklTranspose, m::Int64, n::Int64,
                                               nrhs::Int64, a::Ptr{ComplexF32}, lda::Int64,
                                               stridea::Int64, b::Ptr{ComplexF32},
                                               ldb::Int64, strideb::Int64, batchsize::Int64,
                                               scratchpad::Ptr{ComplexF32},
                                               scratchpad_size::Int64)::Cint
end

function onemklSpotrf_batch_scratchpad_size(device_queue, uplo, n, lda, stride_a,
                                            batch_size)
    @ccall liboneapi_support.onemklSpotrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDpotrf_batch_scratchpad_size(device_queue, uplo, n, lda, stride_a,
                                            batch_size)
    @ccall liboneapi_support.onemklDpotrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCpotrf_batch_scratchpad_size(device_queue, uplo, n, lda, stride_a,
                                            batch_size)
    @ccall liboneapi_support.onemklCpotrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZpotrf_batch_scratchpad_size(device_queue, uplo, n, lda, stride_a,
                                            batch_size)
    @ccall liboneapi_support.onemklZpotrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklSpotrs_batch_scratchpad_size(device_queue, uplo, n, nrhs, lda, stride_a, ldb,
                                            stride_b, batch_size)
    @ccall liboneapi_support.onemklSpotrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                nrhs::Int64, lda::Int64,
                                                                stride_a::Int64, ldb::Int64,
                                                                stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDpotrs_batch_scratchpad_size(device_queue, uplo, n, nrhs, lda, stride_a, ldb,
                                            stride_b, batch_size)
    @ccall liboneapi_support.onemklDpotrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                nrhs::Int64, lda::Int64,
                                                                stride_a::Int64, ldb::Int64,
                                                                stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCpotrs_batch_scratchpad_size(device_queue, uplo, n, nrhs, lda, stride_a, ldb,
                                            stride_b, batch_size)
    @ccall liboneapi_support.onemklCpotrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                nrhs::Int64, lda::Int64,
                                                                stride_a::Int64, ldb::Int64,
                                                                stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZpotrs_batch_scratchpad_size(device_queue, uplo, n, nrhs, lda, stride_a, ldb,
                                            stride_b, batch_size)
    @ccall liboneapi_support.onemklZpotrs_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                uplo::onemklUplo, n::Int64,
                                                                nrhs::Int64, lda::Int64,
                                                                stride_a::Int64, ldb::Int64,
                                                                stride_b::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklSgeqrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_tau,
                                            batch_size)
    @ccall liboneapi_support.onemklSgeqrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDgeqrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_tau,
                                            batch_size)
    @ccall liboneapi_support.onemklDgeqrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCgeqrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_tau,
                                            batch_size)
    @ccall liboneapi_support.onemklCgeqrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZgeqrf_batch_scratchpad_size(device_queue, m, n, lda, stride_a, stride_tau,
                                            batch_size)
    @ccall liboneapi_support.onemklZgeqrf_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                lda::Int64, stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklSorgqr_batch_scratchpad_size(device_queue, m, n, k, lda, stride_a,
                                            stride_tau, batch_size)
    @ccall liboneapi_support.onemklSorgqr_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                k::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDorgqr_batch_scratchpad_size(device_queue, m, n, k, lda, stride_a,
                                            stride_tau, batch_size)
    @ccall liboneapi_support.onemklDorgqr_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                k::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCungqr_batch_scratchpad_size(device_queue, m, n, k, lda, stride_a,
                                            stride_tau, batch_size)
    @ccall liboneapi_support.onemklCungqr_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                k::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZungqr_batch_scratchpad_size(device_queue, m, n, k, lda, stride_a,
                                            stride_tau, batch_size)
    @ccall liboneapi_support.onemklZungqr_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                m::Int64, n::Int64,
                                                                k::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_tau::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklSgetri_batch_scratchpad_size(device_queue, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklSgetri_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                n::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklDgetri_batch_scratchpad_size(device_queue, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklDgetri_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                n::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklCgetri_batch_scratchpad_size(device_queue, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklCgetri_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                n::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklZgetri_batch_scratchpad_size(device_queue, n, lda, stride_a, stride_ipiv,
                                            batch_size)
    @ccall liboneapi_support.onemklZgetri_batch_scratchpad_size(device_queue::syclQueue_t,
                                                                n::Int64, lda::Int64,
                                                                stride_a::Int64,
                                                                stride_ipiv::Int64,
                                                                batch_size::Int64)::Int64
end

function onemklSgels_batch_scratchpad_size(device_queue, trans, m, n, nrhs, lda, stride_a,
                                           ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklSgels_batch_scratchpad_size(device_queue::syclQueue_t,
                                                               trans::onemklTranspose,
                                                               m::Int64, n::Int64,
                                                               nrhs::Int64, lda::Int64,
                                                               stride_a::Int64, ldb::Int64,
                                                               stride_b::Int64,
                                                               batch_size::Int64)::Int64
end

function onemklDgels_batch_scratchpad_size(device_queue, trans, m, n, nrhs, lda, stride_a,
                                           ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklDgels_batch_scratchpad_size(device_queue::syclQueue_t,
                                                               trans::onemklTranspose,
                                                               m::Int64, n::Int64,
                                                               nrhs::Int64, lda::Int64,
                                                               stride_a::Int64, ldb::Int64,
                                                               stride_b::Int64,
                                                               batch_size::Int64)::Int64
end

function onemklCgels_batch_scratchpad_size(device_queue, trans, m, n, nrhs, lda, stride_a,
                                           ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklCgels_batch_scratchpad_size(device_queue::syclQueue_t,
                                                               trans::onemklTranspose,
                                                               m::Int64, n::Int64,
                                                               nrhs::Int64, lda::Int64,
                                                               stride_a::Int64, ldb::Int64,
                                                               stride_b::Int64,
                                                               batch_size::Int64)::Int64
end

function onemklZgels_batch_scratchpad_size(device_queue, trans, m, n, nrhs, lda, stride_a,
                                           ldb, stride_b, batch_size)
    @ccall liboneapi_support.onemklZgels_batch_scratchpad_size(device_queue::syclQueue_t,
                                                               trans::onemklTranspose,
                                                               m::Int64, n::Int64,
                                                               nrhs::Int64, lda::Int64,
                                                               stride_a::Int64, ldb::Int64,
                                                               stride_b::Int64,
                                                               batch_size::Int64)::Int64
end

function onemklXsparse_init_matrix_handle(handle)
    @ccall liboneapi_support.onemklXsparse_init_matrix_handle(handle::Ptr{matrix_handle_t})::Cint
end

function onemklSsparse_set_csr_data(device_queue, handle, num_rows, num_cols, index,
                                    row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklSsparse_set_csr_data(device_queue::syclQueue_t,
                                                        handle::matrix_handle_t,
                                                        num_rows::Int32, num_cols::Int32,
                                                        index::onemklIndex,
                                                        row_ptr::ZePtr{Int32},
                                                        col_ind::ZePtr{Int32},
                                                        val::ZePtr{Cfloat})::Cint
end

function onemklSsparse_set_csr_data_64(device_queue, handle, num_rows, num_cols, index,
                                       row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklSsparse_set_csr_data_64(device_queue::syclQueue_t,
                                                           handle::matrix_handle_t,
                                                           num_rows::Int64, num_cols::Int64,
                                                           index::onemklIndex,
                                                           row_ptr::ZePtr{Int64},
                                                           col_ind::ZePtr{Int64},
                                                           val::ZePtr{Cfloat})::Cint
end

function onemklDsparse_set_csr_data(device_queue, handle, num_rows, num_cols, index,
                                    row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklDsparse_set_csr_data(device_queue::syclQueue_t,
                                                        handle::matrix_handle_t,
                                                        num_rows::Int32, num_cols::Int32,
                                                        index::onemklIndex,
                                                        row_ptr::ZePtr{Int32},
                                                        col_ind::ZePtr{Int32},
                                                        val::ZePtr{Cdouble})::Cint
end

function onemklDsparse_set_csr_data_64(device_queue, handle, num_rows, num_cols, index,
                                       row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklDsparse_set_csr_data_64(device_queue::syclQueue_t,
                                                           handle::matrix_handle_t,
                                                           num_rows::Int64, num_cols::Int64,
                                                           index::onemklIndex,
                                                           row_ptr::ZePtr{Int64},
                                                           col_ind::ZePtr{Int64},
                                                           val::ZePtr{Cdouble})::Cint
end

function onemklCsparse_set_csr_data(device_queue, handle, num_rows, num_cols, index,
                                    row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklCsparse_set_csr_data(device_queue::syclQueue_t,
                                                        handle::matrix_handle_t,
                                                        num_rows::Int32, num_cols::Int32,
                                                        index::onemklIndex,
                                                        row_ptr::ZePtr{Int32},
                                                        col_ind::ZePtr{Int32},
                                                        val::ZePtr{ComplexF32})::Cint
end

function onemklCsparse_set_csr_data_64(device_queue, handle, num_rows, num_cols, index,
                                       row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklCsparse_set_csr_data_64(device_queue::syclQueue_t,
                                                           handle::matrix_handle_t,
                                                           num_rows::Int64, num_cols::Int64,
                                                           index::onemklIndex,
                                                           row_ptr::ZePtr{Int64},
                                                           col_ind::ZePtr{Int64},
                                                           val::ZePtr{ComplexF32})::Cint
end

function onemklZsparse_set_csr_data(device_queue, handle, num_rows, num_cols, index,
                                    row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklZsparse_set_csr_data(device_queue::syclQueue_t,
                                                        handle::matrix_handle_t,
                                                        num_rows::Int32, num_cols::Int32,
                                                        index::onemklIndex,
                                                        row_ptr::ZePtr{Int32},
                                                        col_ind::ZePtr{Int32},
                                                        val::ZePtr{ComplexF64})::Cint
end

function onemklZsparse_set_csr_data_64(device_queue, handle, num_rows, num_cols, index,
                                       row_ptr, col_ind, val)
    @ccall liboneapi_support.onemklZsparse_set_csr_data_64(device_queue::syclQueue_t,
                                                           handle::matrix_handle_t,
                                                           num_rows::Int64, num_cols::Int64,
                                                           index::onemklIndex,
                                                           row_ptr::ZePtr{Int64},
                                                           col_ind::ZePtr{Int64},
                                                           val::ZePtr{ComplexF64})::Cint
end

function onemklSsparse_gemv(device_queue, transpose_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklSsparse_gemv(device_queue::syclQueue_t,
                                                transpose_flag::onemklTranspose,
                                                alpha::Ref{Float32},
                                                handle::matrix_handle_t, x::ZePtr{Float32},
                                                beta::Ref{Float32}, y::ZePtr{Float32})::Cint
end

function onemklDsparse_gemv(device_queue, transpose_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklDsparse_gemv(device_queue::syclQueue_t,
                                                transpose_flag::onemklTranspose,
                                                alpha::Ref{Float64},
                                                handle::matrix_handle_t, x::ZePtr{Float64},
                                                beta::Ref{Float64}, y::ZePtr{Float64})::Cint
end

function onemklCsparse_gemv(device_queue, transpose_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklCsparse_gemv(device_queue::syclQueue_t,
                                                transpose_flag::onemklTranspose,
                                                alpha::Ref{ComplexF32},
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF32}, beta::Ref{ComplexF32},
                                                y::ZePtr{ComplexF32})::Cint
end

function onemklZsparse_gemv(device_queue, transpose_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklZsparse_gemv(device_queue::syclQueue_t,
                                                transpose_flag::onemklTranspose,
                                                alpha::Ref{ComplexF64},
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF64}, beta::Ref{ComplexF64},
                                                y::ZePtr{ComplexF64})::Cint
end

function onemklSsparse_symv(device_queue, uplo_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklSsparse_symv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo, alpha::Ref{Float32},
                                                handle::matrix_handle_t, x::ZePtr{Float32},
                                                beta::Ref{Float32}, y::ZePtr{Float32})::Cint
end

function onemklDsparse_symv(device_queue, uplo_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklDsparse_symv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo, alpha::Ref{Float64},
                                                handle::matrix_handle_t, x::ZePtr{Float64},
                                                beta::Ref{Float64}, y::ZePtr{Float64})::Cint
end

function onemklCsparse_symv(device_queue, uplo_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklCsparse_symv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                alpha::Ref{ComplexF32},
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF32}, beta::Ref{ComplexF32},
                                                y::ZePtr{ComplexF32})::Cint
end

function onemklZsparse_symv(device_queue, uplo_flag, alpha, handle, x, beta, y)
    @ccall liboneapi_support.onemklZsparse_symv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                alpha::Ref{ComplexF64},
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF64}, beta::Ref{ComplexF64},
                                                y::ZePtr{ComplexF64})::Cint
end

function onemklSsparse_trmv(device_queue, uplo_flag, transpose_flag, diag_val, alpha,
                            handle, x, beta, y)
    @ccall liboneapi_support.onemklSsparse_trmv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag, alpha::Ref{Float32},
                                                handle::matrix_handle_t, x::ZePtr{Float32},
                                                beta::Ref{Float32}, y::ZePtr{Float32})::Cint
end

function onemklDsparse_trmv(device_queue, uplo_flag, transpose_flag, diag_val, alpha,
                            handle, x, beta, y)
    @ccall liboneapi_support.onemklDsparse_trmv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag, alpha::Ref{Float64},
                                                handle::matrix_handle_t, x::ZePtr{Float64},
                                                beta::Ref{Float64}, y::ZePtr{Float64})::Cint
end

function onemklCsparse_trmv(device_queue, uplo_flag, transpose_flag, diag_val, alpha,
                            handle, x, beta, y)
    @ccall liboneapi_support.onemklCsparse_trmv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag,
                                                alpha::Ref{ComplexF32},
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF32}, beta::Ref{ComplexF32},
                                                y::ZePtr{ComplexF32})::Cint
end

function onemklZsparse_trmv(device_queue, uplo_flag, transpose_flag, diag_val, alpha,
                            handle, x, beta, y)
    @ccall liboneapi_support.onemklZsparse_trmv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag,
                                                alpha::Ref{ComplexF64},
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF64}, beta::Ref{ComplexF64},
                                                y::ZePtr{ComplexF64})::Cint
end

function onemklSsparse_trsv(device_queue, uplo_flag, transpose_flag, diag_val, handle, x, y)
    @ccall liboneapi_support.onemklSsparse_trsv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag,
                                                handle::matrix_handle_t, x::ZePtr{Cfloat},
                                                y::ZePtr{Cfloat})::Cint
end

function onemklDsparse_trsv(device_queue, uplo_flag, transpose_flag, diag_val, handle, x, y)
    @ccall liboneapi_support.onemklDsparse_trsv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag,
                                                handle::matrix_handle_t, x::ZePtr{Cdouble},
                                                y::ZePtr{Cdouble})::Cint
end

function onemklCsparse_trsv(device_queue, uplo_flag, transpose_flag, diag_val, handle, x, y)
    @ccall liboneapi_support.onemklCsparse_trsv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag,
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF32},
                                                y::ZePtr{ComplexF32})::Cint
end

function onemklZsparse_trsv(device_queue, uplo_flag, transpose_flag, diag_val, handle, x, y)
    @ccall liboneapi_support.onemklZsparse_trsv(device_queue::syclQueue_t,
                                                uplo_flag::onemklUplo,
                                                transpose_flag::onemklTranspose,
                                                diag_val::onemklDiag,
                                                handle::matrix_handle_t,
                                                x::ZePtr{ComplexF64},
                                                y::ZePtr{ComplexF64})::Cint
end

function onemklSsparse_gemm(device_queue, dense_matrix_layout, opA, opB, alpha, handle, b,
                            columns, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklSsparse_gemm(device_queue::syclQueue_t,
                                                dense_matrix_layout::onemklLayout,
                                                opA::onemklTranspose, opB::onemklTranspose,
                                                alpha::Ref{Float32},
                                                handle::matrix_handle_t, b::ZePtr{Float32},
                                                columns::Int64, ldb::Int64,
                                                beta::Ref{Float32}, c::ZePtr{Float32},
                                                ldc::Int64)::Cint
end

function onemklDsparse_gemm(device_queue, dense_matrix_layout, opA, opB, alpha, handle, b,
                            columns, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklDsparse_gemm(device_queue::syclQueue_t,
                                                dense_matrix_layout::onemklLayout,
                                                opA::onemklTranspose, opB::onemklTranspose,
                                                alpha::Ref{Float64},
                                                handle::matrix_handle_t, b::ZePtr{Float64},
                                                columns::Int64, ldb::Int64,
                                                beta::Ref{Float64}, c::ZePtr{Float64},
                                                ldc::Int64)::Cint
end

function onemklCsparse_gemm(device_queue, dense_matrix_layout, opA, opB, alpha, handle, b,
                            columns, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklCsparse_gemm(device_queue::syclQueue_t,
                                                dense_matrix_layout::onemklLayout,
                                                opA::onemklTranspose, opB::onemklTranspose,
                                                alpha::Ref{ComplexF32},
                                                handle::matrix_handle_t,
                                                b::ZePtr{ComplexF32}, columns::Int64,
                                                ldb::Int64, beta::Ref{ComplexF32},
                                                c::ZePtr{ComplexF32}, ldc::Int64)::Cint
end

function onemklZsparse_gemm(device_queue, dense_matrix_layout, opA, opB, alpha, handle, b,
                            columns, ldb, beta, c, ldc)
    @ccall liboneapi_support.onemklZsparse_gemm(device_queue::syclQueue_t,
                                                dense_matrix_layout::onemklLayout,
                                                opA::onemklTranspose, opB::onemklTranspose,
                                                alpha::Ref{ComplexF64},
                                                handle::matrix_handle_t,
                                                b::ZePtr{ComplexF64}, columns::Int64,
                                                ldb::Int64, beta::Ref{ComplexF64},
                                                c::ZePtr{ComplexF64}, ldc::Int64)::Cint
end

function onemklXsparse_init_matmat_descr(desc)
    @ccall liboneapi_support.onemklXsparse_init_matmat_descr(desc::Ptr{matmat_descr_t})::Cint
end

function onemklXsparse_release_matmat_descr(desc)
    @ccall liboneapi_support.onemklXsparse_release_matmat_descr(desc::Ptr{matmat_descr_t})::Cint
end

function onemklXsparse_set_matmat_data(descr, viewA, opA, viewB, opB, viewC)
    @ccall liboneapi_support.onemklXsparse_set_matmat_data(descr::matmat_descr_t,
                                                           viewA::onemklMatrixView,
                                                           opA::onemklTranspose,
                                                           viewB::onemklMatrixView,
                                                           opB::onemklTranspose,
                                                           viewC::onemklMatrixView)::Cint
end

function onemklDestroy()
    @ccall liboneapi_support.onemklDestroy()::Cint
end
