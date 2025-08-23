#include "onemkl.h"
#include "sycl.hpp"
#include <iostream>
#include <exception>
#include <memory>
#include <oneapi/mkl.hpp>

oneapi::mkl::transpose convert(onemklTranspose val) {
    switch (val) {
    case ONEMKL_TRANSPOSE_NONTRANS:
        return oneapi::mkl::transpose::nontrans;
    case ONEMKL_TRANSPOSE_TRANS:
        return oneapi::mkl::transpose::trans;
    case ONEMLK_TRANSPOSE_CONJTRANS:
        return oneapi::mkl::transpose::conjtrans;
    }
}

oneapi::mkl::transpose* convert(const onemklTranspose* vals, int64_t size) {
    oneapi::mkl::transpose* result = new oneapi::mkl::transpose[size];
    for (int64_t i = 0; i < size; ++i) {
        switch (vals[i]) {
            case ONEMKL_TRANSPOSE_NONTRANS:
                result[i] = oneapi::mkl::transpose::nontrans;
                break;
            case ONEMKL_TRANSPOSE_TRANS:
                result[i] = oneapi::mkl::transpose::trans;
                break;
            case ONEMLK_TRANSPOSE_CONJTRANS:
                result[i] = oneapi::mkl::transpose::conjtrans;
                break;
        }
    }
    return result;
}

oneapi::mkl::uplo convert(onemklUplo val) {
    switch(val) {
        case ONEMKL_UPLO_UPPER:
            return oneapi::mkl::uplo::upper;
        case ONEMKL_UPLO_LOWER:
            return oneapi::mkl::uplo::lower;
    }
}

oneapi::mkl::uplo* convert(const onemklUplo* vals, int64_t size) {
    oneapi::mkl::uplo* result = new oneapi::mkl::uplo[size];
    for (int64_t i = 0; i < size; ++i) {
        switch (vals[i]) {
            case ONEMKL_UPLO_UPPER:
                result[i] = oneapi::mkl::uplo::upper;
                break;
            case ONEMKL_UPLO_LOWER:
                result[i] = oneapi::mkl::uplo::lower;
                break;
        }
    }
    return result;
}

oneapi::mkl::diag convert(onemklDiag val) {
    switch(val) {
        case ONEMKL_DIAG_NONUNIT:
            return oneapi::mkl::diag::nonunit;
        case ONEMKL_DIAG_UNIT:
            return oneapi::mkl::diag::unit;
    }
}

oneapi::mkl::diag* convert(const onemklDiag* vals, int64_t size) {
    oneapi::mkl::diag* result = new oneapi::mkl::diag[size];
    for (int64_t i = 0; i < size; ++i) {
        switch (vals[i]) {
            case ONEMKL_DIAG_NONUNIT:
                result[i] = oneapi::mkl::diag::nonunit;
                break;
            case ONEMKL_DIAG_UNIT:
                result[i] = oneapi::mkl::diag::unit;
                break;
        }
    }
    return result;
}

oneapi::mkl::side convert(onemklSide val) {
    switch (val) {
    case ONEMKL_SIDE_LEFT:
        return oneapi::mkl::side::left;
    case ONEMKL_SIDE_RIGHT:
        return oneapi::mkl::side::right;
    }
}

oneapi::mkl::side* convert(const onemklSide* vals, int64_t size) {
    oneapi::mkl::side* result = new oneapi::mkl::side[size];
    for (int64_t i = 0; i < size; ++i) {
        switch (vals[i]) {
            case ONEMKL_SIDE_LEFT:
                result[i] = oneapi::mkl::side::left;
                break;
            case ONEMKL_SIDE_RIGHT:
                result[i] = oneapi::mkl::side::right;
                break;
        }
    }
    return result;
}

oneapi::mkl::offset convert(onemklOffset val) {
    switch (val) {
    case ONEMKL_OFFSET_ROW:
        return oneapi::mkl::offset::row;
    case ONEMKL_OFFSET_COL:
        return oneapi::mkl::offset::column;
    case ONEMKL_OFFSET_FIX:
        return oneapi::mkl::offset::fix;
    }
}

oneapi::mkl::job convert(onemklJob val) {
    switch (val) {
    case ONEMKL_JOB_N:
        return oneapi::mkl::job::N;
    case ONEMKL_JOB_V:
        return oneapi::mkl::job::V;
    case ONEMKL_JOB_U:
        return oneapi::mkl::job::U;
    case ONEMKL_JOB_A:
        return oneapi::mkl::job::A;
    case ONEMKL_JOB_S:
        return oneapi::mkl::job::S;
    case ONEMKL_JOB_O:
        return oneapi::mkl::job::O;
    }
}

oneapi::mkl::generate convert(onemklGenerate val) {
    switch (val) {
    case ONEMKL_GENERATE_Q:
        return oneapi::mkl::generate::Q;
    case ONEMKL_GENERATE_P:
        return oneapi::mkl::generate::P;
    case ONEMKL_GENERATE_N:
        return oneapi::mkl::generate::N;
    case ONEMKL_GENERATE_V:
        return oneapi::mkl::generate::V;
    }
}

oneapi::mkl::compz convert(onemklCompz val) {
    switch (val) {
    case ONEMKL_COMPZ_N:
        return oneapi::mkl::compz::N;
    case ONEMKL_COMPZ_V:
        return oneapi::mkl::compz::V;
    case ONEMKL_COMPZ_I:
        return oneapi::mkl::compz::I;
    }
}

oneapi::mkl::direct convert(onemklDirect val) {
    switch (val) {
    case ONEMKL_DIRECT_F:
        return oneapi::mkl::direct::F;
    case ONEMKL_DIRECT_B:
        return oneapi::mkl::direct::B;
    }
}

oneapi::mkl::storev convert(onemklStorev val) {
    switch (val) {
    case ONEMKL_STOREV_C:
        return oneapi::mkl::storev::C;
    case ONEMKL_STOREV_R:
        return oneapi::mkl::storev::R;
    }
}

oneapi::mkl::rangev convert(onemklRangev val) {
    switch (val) {
    case ONEMKL_RANGEV_A:
        return oneapi::mkl::rangev::A;
    case ONEMKL_RANGEV_V:
        return oneapi::mkl::rangev::V;
    case ONEMKL_RANGEV_I:
        return oneapi::mkl::rangev::I;
    }
}

oneapi::mkl::order convert(onemklOrder val) {
    switch (val) {
    case ONEMKL_ORDER_B:
        return oneapi::mkl::order::B;
    case ONEMKL_ORDER_E:
        return oneapi::mkl::order::E;
    }
}

oneapi::mkl::jobsvd convert(onemklJobsvd val) {
    switch (val) {
    case ONEMKL_JOBSVD_N:
        return oneapi::mkl::jobsvd::N;
    case ONEMKL_JOBSVD_A:
        return oneapi::mkl::jobsvd::A;
    case ONEMKL_JOBSVD_O:
        return oneapi::mkl::jobsvd::O;
    case ONEMKL_JOBSVD_S:
        return oneapi::mkl::jobsvd::S;
    }
}

oneapi::mkl::layout convert(onemklLayout val) {
    switch (val) {
    case ONEMKL_LAYOUT_ROW:
        return oneapi::mkl::layout::row_major;
    case ONEMKL_LAYOUT_COL:
        return oneapi::mkl::layout::col_major;
    }
}

oneapi::mkl::index_base convert(onemklIndex val) {
    switch (val) {
    case ONEMKL_INDEX_ZERO:
        return oneapi::mkl::index_base::zero;
    case ONEMKL_INDEX_ONE:
        return oneapi::mkl::index_base::one;
    }
}

oneapi::mkl::sparse::property convert(onemklProperty val) {
    switch (val) {
    case ONEMKL_PROPERTY_SYMMETRIC:
        return oneapi::mkl::sparse::property::symmetric;
    case ONEMKL_PROPERTY_SORTED:
        return oneapi::mkl::sparse::property::sorted;
    }
}

oneapi::mkl::sparse::matrix_view_descr convert(onemklMatrixView val) {
    switch (val) {
    case ONEMKL_MATRIX_VIEW_GENERAL:
        return oneapi::mkl::sparse::matrix_view_descr::general;
    }
}

oneapi::mkl::sparse::matmat_request convert(onemklMatmatRequest val) {
    switch (val) {
    case ONEMKL_MATMAT_REQUEST_GET_WORK_ESTIMATION_BUF_SIZE:
        return oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
    case ONEMKL_MATMAT_REQUEST_WORK_ESTIMATION:
        return oneapi::mkl::sparse::matmat_request::work_estimation;
    case ONEMKL_MATMAT_REQUEST_GET_COMPUTE_STRUCTURE_BUF_SIZE:
        return oneapi::mkl::sparse::matmat_request::get_compute_structure_buf_size;
    case ONEMKL_MATMAT_REQUEST_COMPUTE_STRUCTURE:
        return oneapi::mkl::sparse::matmat_request::compute_structure;
    case ONEMKL_MATMAT_REQUEST_FINALIZE_STRUCTURE:
        return oneapi::mkl::sparse::matmat_request::finalize_structure;
    case ONEMKL_MATMAT_REQUEST_GET_COMPUTE_BUF_SIZE:
        return oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
    case ONEMKL_MATMAT_REQUEST_COMPUTE:
        return oneapi::mkl::sparse::matmat_request::compute;
    case ONEMKL_MATMAT_REQUEST_GET_NNZ:
        return oneapi::mkl::sparse::matmat_request::get_nnz;
    case ONEMKL_MATMAT_REQUEST_FINALIZE:
        return oneapi::mkl::sparse::matmat_request::finalize;
    }
}

oneapi::mkl::sparse::omatconvert_alg convert(onemklOmatconvertAlg val) {
    switch (val) {
    case ONEMKL_OMATCONVERT_DEFAULT_ALG:
        return oneapi::mkl::sparse::omatconvert_alg::default_alg;
    }
}

oneapi::mkl::sparse::omatadd_alg convert(onemklOmataddAlg val) {
    switch (val) {
    case ONEMKL_OMATADD_DEFAULT_ALG:
        return oneapi::mkl::sparse::omatadd_alg::default_alg;
    }
}

// gemm
// https://spec.oneapi.io/versions/1.0-rev-1/elements/oneMKL/source/domains/blas/gemm.html
class gemmBatchInfo {
    public:
        oneapi::mkl::transpose *m_transa = nullptr;
        oneapi::mkl::transpose *m_transb = nullptr;
        sycl::device m_device;
        sycl::context m_context;
        oneapi::mkl::transpose m_ta;
        oneapi::mkl::transpose m_tb;
        // Constructor
        gemmBatchInfo(syclQueue_t device_queue,
                    int64_t group_count,
                    onemklTranspose transa,
                    onemklTranspose transb) {
            // Get device and context info from device_queue
            auto main_queue = device_queue->val;
            m_device = main_queue.get_device();
            m_context = main_queue.get_context();

            // Allocate transpose shared buffers
            try {
                m_transa = (oneapi::mkl::transpose *) malloc_shared(group_count * sizeof(oneapi::mkl::transpose),
                                                                    m_device, m_context);
                m_transb = (oneapi::mkl::transpose *) malloc_shared(group_count * sizeof(oneapi::mkl::transpose),
                                                                    m_device, m_context);
                m_ta = convert(transa);
                m_tb = convert(transb);
            } catch(const std::bad_alloc& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }

            // Initialize
            for (int i = 0; i < group_count; i++) {
                m_transa[i] = m_ta;
                m_transb[i] = m_tb;
            }
        };

        // Destructor
        ~gemmBatchInfo() {
            free(m_transa, m_context);
            free(m_transb, m_context);
        }
};

class trsmBatchInfo {
    public:
        oneapi::mkl::transpose *m_transa = nullptr;
        oneapi::mkl::side *m_leftright = nullptr;
        oneapi::mkl::uplo *m_upperlower = nullptr;
        oneapi::mkl::diag *m_unitdiag = nullptr;
        sycl::device m_device;
        sycl::context m_context;
        oneapi::mkl::transpose m_ta;
        oneapi::mkl::side m_side;
        oneapi::mkl::uplo m_uplo;
        oneapi::mkl::diag m_diag;

        // Constructor
        trsmBatchInfo(syclQueue_t device_queue,
                    onemklSide left_right,
                    onemklUplo upper_lower,
                    onemklTranspose transa,
                    onemklDiag unit_diag,
                    int64_t group_count) {
            // Get device and context info from device_queue
            auto main_queue = device_queue->val;
            m_device = main_queue.get_device();
            m_context = main_queue.get_context();
            try {
                // Allocate uniform arrays of group_size and transpose_a, transpose_b supporting oneMKL
                // gemm_batch API
                m_transa = (oneapi::mkl::transpose *) malloc_shared(group_count * sizeof(oneapi::mkl::transpose),
                                                                    m_device, m_context);
                m_leftright = (oneapi::mkl::side *) malloc_shared(group_count * sizeof(oneapi::mkl::side),
                                                                m_device, m_context);
                m_upperlower = (oneapi::mkl::uplo *) malloc_shared(group_count * sizeof(oneapi::mkl::uplo),
                                                                m_device, m_context);
                m_unitdiag = (oneapi::mkl::diag *) malloc_shared(group_count * sizeof(oneapi::mkl::diag),
                                                                m_device, m_context);
                m_ta = convert(transa);
                m_side = convert(left_right);
                m_uplo = convert(upper_lower);
                m_diag = convert(unit_diag);
            } catch(const std::bad_alloc& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
            // Initialize
            for (int i = 0; i < group_count; i++) {
                m_transa[i] = m_ta;
                m_leftright[i] = m_side;
                m_upperlower[i] = m_uplo;
                m_unitdiag[i] = m_diag;
            }
        };

        // Destructor
        ~trsmBatchInfo() {
            free(m_transa, m_context);
            free(m_upperlower, m_context);
            free(m_unitdiag, m_context);
            free(m_leftright, m_context);
        }
};

extern "C" int onemklHgemm_batch(syclQueue_t device_queue, onemklTranspose transa,
                                 onemklTranspose transb, int64_t *m,
                                 int64_t *n, int64_t *k, uint16_t *alpha,
                                 const short **a, int64_t *lda, const short **b,
                                 int64_t *ldb, uint16_t *beta, short **c,
                                 int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    device_queue->val.wait_and_throw();
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, reinterpret_cast<sycl::half *>(alpha),
                        reinterpret_cast<const sycl::half **>(&a[0]), lda,
                        reinterpret_cast<const sycl::half **>(&b[0]), ldb,
                        reinterpret_cast<sycl::half *>(beta), reinterpret_cast<sycl::half **>(&c[0]),
                        ldc, group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklSgemm_batch(syclQueue_t device_queue, onemklTranspose transa,
                                 onemklTranspose transb, int64_t *m,
                                 int64_t *n, int64_t *k, float *alpha,
                                 const float **a, int64_t *lda, const float **b,
                                 int64_t *ldb, float *beta, float **c,
                                 int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    device_queue->val.wait_and_throw();
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, alpha,
                        (const float **)&a[0], lda,
                        (const float **)&b[0], ldb,
                        beta, &c[0], ldc,
                        group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklDgemm_batch(syclQueue_t device_queue, onemklTranspose transa,
                                 onemklTranspose transb, int64_t *m,
                                 int64_t *n, int64_t *k, double *alpha,
                                 const double **a, int64_t *lda, const double **b,
                                 int64_t *ldb, double *beta, double **c,
                                 int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    device_queue->val.wait_and_throw();
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, alpha,
                        (const double **)&a[0], lda,
                        (const double **)&b[0], ldb,
                        beta, &c[0], ldc,
                        group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklCgemm_batch(syclQueue_t device_queue, onemklTranspose transa,
                                 onemklTranspose transb, int64_t *m,
                                 int64_t *n, int64_t *k, float _Complex *alpha,
                                 const float _Complex **a, int64_t *lda,
                                 const float _Complex **b,
                                 int64_t *ldb, float _Complex *beta, float _Complex **c,
                                 int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    device_queue->val.wait_and_throw();
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, reinterpret_cast<std::complex<float> *>(alpha),
                        reinterpret_cast<const std::complex<float> **>(&a[0]),
                        lda,
                        reinterpret_cast<const std::complex<float> **>(&b[0]),
                        ldb,
                        reinterpret_cast<std::complex<float> *>(beta),
                        reinterpret_cast<std::complex<float> **>(&c[0]), ldc,
                        group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklZgemm_batch(syclQueue_t device_queue, onemklTranspose transa,
                                 onemklTranspose transb, int64_t *m,
                                 int64_t *n, int64_t *k, double _Complex *alpha,
                                 const double _Complex **a, int64_t *lda,
                                 const double _Complex **b,
                                 int64_t *ldb, double _Complex *beta,
                                 double _Complex **c,
                                 int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    device_queue->val.wait_and_throw();
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, reinterpret_cast<std::complex<double> *>(alpha),
                        reinterpret_cast<const std::complex<double> **>(&a[0]),
                        lda,
                        reinterpret_cast<const std::complex<double> **>(&b[0]),
                        ldb,
                        reinterpret_cast<std::complex<double> *>(beta),
                        reinterpret_cast<std::complex<double> **>(&c[0]), ldc,
                        group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklStrsm_batch(syclQueue_t device_queue, onemklSide left_right,
                                 onemklUplo upper_lower, onemklTranspose transa,
                                 onemklDiag unit_diag, int64_t *m, int64_t *n, float *alpha,
                                 const float **a, int64_t *lda, float **b, int64_t *ldb,
                                 int64_t group_count, int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right, upper_lower, transa,
                           unit_diag, group_count);
    device_queue->val.wait_and_throw();

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, alpha, (const float **)&a[0], lda,
                        &b[0], ldb, group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklDtrsm_batch(syclQueue_t device_queue, onemklSide left_right,
                                 onemklUplo upper_lower, onemklTranspose transa,
                                 onemklDiag unit_diag, int64_t *m, int64_t *n,
                                 double *alpha, const double **a, int64_t *lda,
                                 double **b, int64_t *ldb, int64_t group_count,
                                 int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, group_count);
    device_queue->val.wait_and_throw();

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, alpha, (const double **)&a[0], lda, &b[0],
                        ldb, group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklCtrsm_batch(syclQueue_t device_queue, onemklSide left_right,
                                 onemklUplo upper_lower, onemklTranspose transa,
                                 onemklDiag unit_diag, int64_t *m, int64_t *n,
                                 float _Complex *alpha, const float _Complex **a,
                                 int64_t *lda, float _Complex **b, int64_t *ldb,
                                 int64_t group_count, int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, group_count);
    device_queue->val.wait_and_throw();

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, reinterpret_cast<std::complex<float> *>(alpha),
                        reinterpret_cast<const std::complex<float> **>(&a[0]),
                        lda, reinterpret_cast<std::complex<float> **>(&b[0]),
                        ldb, group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}

extern "C" int onemklZtrsm_batch(syclQueue_t device_queue, onemklSide left_right,
                                 onemklUplo upper_lower, onemklTranspose transa,
                                 onemklDiag unit_diag, int64_t *m, int64_t *n,
                                 double _Complex *alpha, const double _Complex **a,
                                 int64_t *lda, double _Complex **b, int64_t *ldb,
                                 int64_t group_count, int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right,
                                upper_lower, transa, unit_diag, group_count);
    device_queue->val.wait_and_throw();

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, reinterpret_cast<std::complex<double> *>(alpha),
                        reinterpret_cast<const std::complex<double> **>(&a[0]),
                        lda, reinterpret_cast<std::complex<double> **>(&b[0]),
                        ldb, group_count, group_size, {});
    device_queue->val.wait_and_throw();
    return 0;
}
