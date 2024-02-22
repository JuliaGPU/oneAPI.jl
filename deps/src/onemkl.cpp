#include "onemkl.h"
#include "sycl.hpp"
#include <iostream>
#include <exception>
#include <memory>
#include <oneapi/mkl.hpp>

// This is a workaround to flush MKL submissions into Level-zero queue, using
// unspecified but guaranteed behavior of intel-sycl runtime. Once SYCL standard
// committee approves sycl::queue::flush() we will change the macro to use that
#define __FORCE_MKL_FLUSH__(cmd) \
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(cmd)

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

oneapi::mkl::uplo convert(onemklUplo val) {
    switch(val) {
        case ONEMKL_UPLO_UPPER:
            return oneapi::mkl::uplo::upper;
        case ONEMKL_UPLO_LOWER:
            return oneapi::mkl::uplo::lower;
    }
}

oneapi::mkl::diag convert(onemklDiag val) {
    switch(val) {
        case ONEMKL_DIAG_NONUNIT:
            return oneapi::mkl::diag::nonunit;
        case ONEMKL_DIAG_UNIT:
            return oneapi::mkl::diag::unit;
    }
}

oneapi::mkl::side convert(onemklSide val) {
    switch (val) {
    case ONEMKL_SIDE_LEFT:
        return oneapi::mkl::side::left;
    case ONEMKL_SIDE_RIGHT:
        return oneapi::mkl::side::right;
    }
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

extern "C" int onemklHgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t *m,
                                  int64_t *n, int64_t *k, uint16_t *alpha,
                                  const short **a, int64_t *lda, const short **b,
                                  int64_t *ldb, uint16_t *beta, short **c,
                                  int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, reinterpret_cast<sycl::half *>(alpha),
                        reinterpret_cast<const sycl::half **>(&a[0]), lda,
                        reinterpret_cast<const sycl::half **>(&b[0]), ldb,
                        reinterpret_cast<sycl::half *>(beta), reinterpret_cast<sycl::half **>(&c[0]),
                        ldc, group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklSgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t *m,
                                  int64_t *n, int64_t *k, float *alpha,
                                  const float **a, int64_t *lda, const float **b,
                                  int64_t *ldb, float *beta, float **c,
                                  int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, alpha,
                        (const float **)&a[0], lda,
                        (const float **)&b[0], ldb,
                        beta, &c[0], ldc,
                        group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklDgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t *m,
                                  int64_t *n, int64_t *k, double *alpha,
                                  const double **a, int64_t *lda, const double **b,
                                  int64_t *ldb, double *beta, double **c,
                                  int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, alpha,
                        (const double **)&a[0], lda,
                        (const double **)&b[0], ldb,
                        beta, &c[0], ldc,
                        group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklCgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t *m,
                                  int64_t *n, int64_t *k, float _Complex *alpha,
                                  const float _Complex **a, int64_t *lda,
                                  const float _Complex **b,
                                  int64_t *ldb, float _Complex *beta, float _Complex **c,
                                  int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, reinterpret_cast<std::complex<float> *>(alpha),
                        reinterpret_cast<const std::complex<float> **>(&a[0]),
                        lda,
                        reinterpret_cast<const std::complex<float> **>(&b[0]),
                        ldb,
                        reinterpret_cast<std::complex<float> *>(beta),
                        reinterpret_cast<std::complex<float> **>(&c[0]), ldc,
                        group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklZgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t *m,
                                  int64_t *n, int64_t *k, double _Complex *alpha,
                                  const double _Complex **a, int64_t *lda,
                                  const double _Complex **b,
                                  int64_t *ldb, double _Complex *beta,
                                  double _Complex **c,
                                  int64_t *ldc, int64_t group_count, int64_t *group_size) {
    gemmBatchInfo gemmInfo(device_queue, group_count, transa, transb);
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        m, n, k, reinterpret_cast<std::complex<double> *>(alpha),
                        reinterpret_cast<const std::complex<double> **>(&a[0]),
                        lda,
                        reinterpret_cast<const std::complex<double> **>(&b[0]),
                        ldb,
                        reinterpret_cast<std::complex<double> *>(beta),
                        reinterpret_cast<std::complex<double> **>(&c[0]), ldc,
                        group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklStrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t *m, int64_t *n, float *alpha,
                            const float **a, int64_t *lda, float **b, int64_t *ldb,
                            int64_t group_count, int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right, upper_lower, transa,
                           unit_diag, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, alpha, (const float **)&a[0], lda,
                        &b[0], ldb, group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklDtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t *m, int64_t *n,
                            double *alpha, const double **a, int64_t *lda,
                            double **b, int64_t *ldb, int64_t group_count,
                            int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, alpha, (const double **)&a[0], lda, &b[0],
                        ldb, group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklCtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t *m, int64_t *n,
                            float _Complex *alpha, const float _Complex **a,
                            int64_t *lda, float _Complex **b, int64_t *ldb,
                            int64_t group_count, int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, reinterpret_cast<std::complex<float> *>(alpha),
                        reinterpret_cast<const std::complex<float> **>(&a[0]),
                        lda, reinterpret_cast<std::complex<float> **>(&b[0]),
                        ldb, group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklZtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t *m, int64_t *n,
                            double _Complex *alpha, const double _Complex **a,
                            int64_t *lda, double _Complex **b, int64_t *ldb,
                            int64_t group_count, int64_t *group_size) {
    trsmBatchInfo trsmInfo(device_queue, left_right,
                                upper_lower, transa, unit_diag, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        m, n, reinterpret_cast<std::complex<double> *>(alpha),
                        reinterpret_cast<const std::complex<double> **>(&a[0]),
                        lda, reinterpret_cast<std::complex<double> **>(&b[0]),
                        ldb, group_count, group_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklHgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        uint16_t alpha, const short *a, int64_t lda, int64_t stridea,
                        const short *b, int64_t ldb, int64_t strideb, uint16_t beta,
                        short *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, sycl::bit_cast<sycl::half>(alpha),
                                                    reinterpret_cast<const sycl::half *>(a), lda, stridea,
                                                    reinterpret_cast<const sycl::half *>(b), ldb, strideb,
                                                    sycl::bit_cast<sycl::half>(beta),
                                                    reinterpret_cast<sycl::half *>(c), ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklSgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        float alpha, const float *a, int64_t lda, int64_t stridea,
                        const float *b, int64_t ldb, int64_t strideb, float beta,
                        float *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, alpha, a, lda, stridea,
                                                    b, ldb, strideb, beta, c, ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklDgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        double alpha, const double *a, int64_t lda, int64_t stridea,
                        const double *b, int64_t ldb, int64_t strideb, double beta,
                        double *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, alpha, a, lda, stridea,
                                                    b, ldb, strideb, beta, c, ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklCgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        float _Complex alpha, const float _Complex *a, int64_t lda, int64_t stridea,
                        const float _Complex *b, int64_t ldb, int64_t strideb, float _Complex beta,
                        float _Complex *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, alpha,
                                                    reinterpret_cast<const std::complex<float> *>(a),
                                                    lda, stridea,
                                                    reinterpret_cast<const std::complex<float> *>(b),
                                                    ldb, strideb, beta,
                                                    reinterpret_cast<std::complex<float> *>(c),
                                                    ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklZgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        double _Complex alpha, const double _Complex *a, int64_t lda, int64_t stridea,
                        const double _Complex *b, int64_t ldb, int64_t strideb, double _Complex beta,
                        double _Complex *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, alpha,
                                                    reinterpret_cast<const std::complex<double> *>(a),
                                                    lda, stridea,
                                                    reinterpret_cast<const std::complex<double> *>(b),
                                                    ldb, strideb, beta,
                                                    reinterpret_cast<std::complex<double> *>(c),
                                                    ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, uint16_t alpha, const short *A, int64_t lda,
                           const short *B, int64_t ldb, uint16_t beta, short *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transA),
                                          convert(transB), m, n, k, sycl::bit_cast<sycl::half>(alpha),
                                          reinterpret_cast<const sycl::half *>(A), lda,
                                          reinterpret_cast<const sycl::half *>(B), ldb,
                                          sycl::bit_cast<sycl::half>(beta),
                                          reinterpret_cast<sycl::half *>(C), ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklHdot(syclQueue_t device_queue, int64_t n,
                           const short *x, int64_t incx, const short *y,
                           int64_t incy, short *result) {
    auto status = oneapi::mkl::blas::column_major::dot(device_queue->val, n,
                                        reinterpret_cast<const sycl::half *>(x),
                                        incx, reinterpret_cast<const sycl::half *>(y),
                                        incy, reinterpret_cast<sycl::half *>(result));
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklHaxpy(syclQueue_t device_queue, int64_t n, uint16_t alpha,
                            const short *x, std::int64_t incx, short *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n,
                                        sycl::bit_cast<sycl::half>(alpha),
                                        reinterpret_cast<const sycl::half *>(x),
                                        incx, reinterpret_cast<sycl::half *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklHscal(syclQueue_t device_queue, int64_t n, uint16_t alpha,
                            short *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, sycl::bit_cast<sycl::half>(alpha),
                                                        reinterpret_cast<sycl::half *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklHnrm2(syclQueue_t device_queue, int64_t n, const short *x,
                            int64_t incx, short *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n,
                        reinterpret_cast<const sycl::half *>(x), incx,
                        reinterpret_cast<sycl::half *>(result));
    __FORCE_MKL_FLUSH__(status);
    return 0;
}
// BLAS
extern "C" int onemklSgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transa), convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transa), convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transa), convert(transb), m, n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transa), convert(transb), m, n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right), convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right), convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right), convert(upper_lower), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right), convert(upper_lower), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChemm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::hemm(device_queue->val, convert(left_right), convert(upper_lower), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhemm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::hemm(device_queue->val, convert(left_right), convert(upper_lower), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float beta, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, a, lda, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double beta, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, a, lda, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCherk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float alpha, float _Complex *a, int64_t lda, float beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::herk(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, reinterpret_cast<std::complex<float>*>(a), lda, beta, reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZherk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double alpha, double _Complex *a, int64_t lda, double beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::herk(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, reinterpret_cast<std::complex<double>*>(a), lda, beta, reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCher2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::her2k(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, beta, reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZher2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::her2k(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, beta, reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, alpha, a, lda, b, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, alpha, a, lda, b, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, alpha, a, lda, b, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, alpha, a, lda, b, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float *a, int64_t lda, float *x, int64_t incx, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::dgmm(device_queue->val, convert(left_right), m, n, a, lda, x, incx, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double *a, int64_t lda, double *x, int64_t incx, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::dgmm(device_queue->val, convert(left_right), m, n, a, lda, x, incx, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::dgmm(device_queue->val, convert(left_right), m, n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::dgmm(device_queue->val, convert(left_right), m, n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl, int64_t ku, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl, int64_t ku, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl, int64_t ku, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans), m, n, kl, ku, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl, int64_t ku, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans), m, n, kl, ku, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSger(syclQueue_t device_queue, int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::ger(device_queue->val, m, n, alpha, x, incx, y, incy, a, lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDger(syclQueue_t device_queue, int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::ger(device_queue->val, m, n, alpha, x, incx, y, incy, a, lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgerc(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::gerc(device_queue->val, m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgerc(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::gerc(device_queue->val, m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgeru(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::geru(device_queue->val, m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgeru(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::geru(device_queue->val, m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::hbmv(device_queue->val, convert(upper_lower), n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::hbmv(device_queue->val, convert(upper_lower), n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChemv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::hemv(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhemv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::hemv(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCher(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float _Complex *x, int64_t incx, float _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::her(device_queue->val, convert(upper_lower), n, alpha, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZher(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double _Complex *x, int64_t incx, double _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::her(device_queue->val, convert(upper_lower), n, alpha, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCher2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::her2(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZher2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::her2(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChpmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *a, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::hpmv(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhpmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *a, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::hpmv(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChpr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float _Complex *x, int64_t incx, float _Complex *a) {
   auto status = oneapi::mkl::blas::column_major::hpr(device_queue->val, convert(upper_lower), n, alpha, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(a));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhpr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double _Complex *x, int64_t incx, double _Complex *a) {
   auto status = oneapi::mkl::blas::column_major::hpr(device_queue->val, convert(upper_lower), n, alpha, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(a));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChpr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a) {
   auto status = oneapi::mkl::blas::column_major::hpr2(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(a));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhpr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a) {
   auto status = oneapi::mkl::blas::column_major::hpr2(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(a));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::sbmv(device_queue->val, convert(upper_lower), n, k, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::sbmv(device_queue->val, convert(upper_lower), n, k, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::symv(device_queue->val, convert(upper_lower), n, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::symv(device_queue->val, convert(upper_lower), n, alpha, a, lda, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::symv(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::symv(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x, int64_t incx, float *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr(device_queue->val, convert(upper_lower), n, alpha, x, incx, a, lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *x, int64_t incx, double *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr(device_queue->val, convert(upper_lower), n, alpha, x, incx, a, lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr2(device_queue->val, convert(upper_lower), n, alpha, x, incx, y, incy, a, lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr2(device_queue->val, convert(upper_lower), n, alpha, x, incx, y, incy, a, lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr2(device_queue->val, convert(upper_lower), n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a, int64_t lda) {
   auto status = oneapi::mkl::blas::column_major::syr2(device_queue->val, convert(upper_lower), n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(a), lda);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSspmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *a, float *x, int64_t incx, float beta, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::spmv(device_queue->val, convert(upper_lower), n, alpha, a, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDspmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *a, double *x, int64_t incx, double beta, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::spmv(device_queue->val, convert(upper_lower), n, alpha, a, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSspr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x, int64_t incx, float *a) {
   auto status = oneapi::mkl::blas::column_major::spr(device_queue->val, convert(upper_lower), n, alpha, x, incx, a);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDspr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *x, int64_t incx, double *a) {
   auto status = oneapi::mkl::blas::column_major::spr(device_queue->val, convert(upper_lower), n, alpha, x, incx, a);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSspr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a) {
   auto status = oneapi::mkl::blas::column_major::spr2(device_queue->val, convert(upper_lower), n, alpha, x, incx, y, incy, a);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDspr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a) {
   auto status = oneapi::mkl::blas::column_major::spr2(device_queue->val, convert(upper_lower), n, alpha, x, incx, y, incy, a);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, float *a, int64_t lda, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, double *a, int64_t lda, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, float *a, int64_t lda, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, double *a, int64_t lda, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tbsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, k, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float *a, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double *a, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float _Complex *a, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<float>*>(a), reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double _Complex *a, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<double>*>(a), reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float *a, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double *a, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float _Complex *a, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<float>*>(a), reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double _Complex *a, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::tpsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<double>*>(a), reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float *a, int64_t lda, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double *a, int64_t lda, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float *a, int64_t lda, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double *a, int64_t lda, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, a, lda, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t n, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(upper_lower), convert(trans), convert(unit_diag), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCdotc(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *result) {
   auto status = oneapi::mkl::blas::column_major::dotc(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(result));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZdotc(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *result) {
   auto status = oneapi::mkl::blas::column_major::dotc(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(result));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCdotu(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *result) {
   auto status = oneapi::mkl::blas::column_major::dotu(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, reinterpret_cast<std::complex<float>*>(result));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZdotu(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *result) {
   auto status = oneapi::mkl::blas::column_major::dotu(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, reinterpret_cast<std::complex<double>*>(result));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSiamax(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, x, incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDiamax(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, x, incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCiamax(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZiamax(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSiamin(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, x, incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDiamin(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, x, incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCiamin(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZiamin(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, int64_t *result, onemklIndex base) {
   auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, result, convert(base));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSasum(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *result) {
   auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n, x, incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDasum(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *result) {
   auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n, x, incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCasum(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float *result) {
   auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZasum(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double *result) {
   auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSaxpy(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, alpha, x, incx, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDaxpy(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, alpha, x, incx, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCaxpy(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZaxpy(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSaxpby(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx, float beta, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, alpha, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDaxpby(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx, double beta, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, alpha, x, incx, beta, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCaxpby(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZaxpby(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklScopy(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n, x, incx, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDcopy(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n, x, incx, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCcopy(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZcopy(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSdot(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy, float *result) {
   auto status = oneapi::mkl::blas::column_major::dot(device_queue->val, n, x, incx, y, incy, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDdot(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy, double *result) {
   auto status = oneapi::mkl::blas::column_major::dot(device_queue->val, n, x, incx, y, incy, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsdsdot(syclQueue_t device_queue, int64_t n, float sb, float *x, int64_t incx, float *y, int64_t incy, float *result) {
   auto status = oneapi::mkl::blas::column_major::sdsdot(device_queue->val, n, sb, x, incx, y, incy, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSnrm2(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *result) {
   auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, x, incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDnrm2(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *result) {
   auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, x, incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCnrm2(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float *result) {
   auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZnrm2(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double *result) {
   auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, result);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSrot(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy, float c, float s) {
   auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, x, incx, y, incy, c, s);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDrot(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy, double c, double s) {
   auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, x, incx, y, incy, c, s);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCSrot(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float c, float s) {
   auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, c, s);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCrot(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float c, float _Complex s) {
   auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy, c, static_cast<std::complex<float> >(s));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZDrot(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double c, double s) {
   auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, c, s);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZrot(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double c, double _Complex s) {
   auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy, c, static_cast<std::complex<double> >(s));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSrotg(syclQueue_t device_queue, float *a, float *b, float *c, float *s) {
   auto status = oneapi::mkl::blas::column_major::rotg(device_queue->val, a, b, c, s);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDrotg(syclQueue_t device_queue, double *a, double *b, double *c, double *s) {
   auto status = oneapi::mkl::blas::column_major::rotg(device_queue->val, a, b, c, s);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCrotg(syclQueue_t device_queue, float _Complex *a, float _Complex *b, float *c, float _Complex *s) {
   auto status = oneapi::mkl::blas::column_major::rotg(device_queue->val, reinterpret_cast<std::complex<float>*>(a), reinterpret_cast<std::complex<float>*>(b), c, reinterpret_cast<std::complex<float>*>(s));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZrotg(syclQueue_t device_queue, double _Complex *a, double _Complex *b, double *c, double _Complex *s) {
   auto status = oneapi::mkl::blas::column_major::rotg(device_queue->val, reinterpret_cast<std::complex<double>*>(a), reinterpret_cast<std::complex<double>*>(b), c, reinterpret_cast<std::complex<double>*>(s));
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSrotm(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy, float *param) {
   auto status = oneapi::mkl::blas::column_major::rotm(device_queue->val, n, x, incx, y, incy, param);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDrotm(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy, double *param) {
   auto status = oneapi::mkl::blas::column_major::rotm(device_queue->val, n, x, incx, y, incy, param);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSrotmg(syclQueue_t device_queue, float *d1, float *d2, float *x1, float y1, float *param) {
   auto status = oneapi::mkl::blas::column_major::rotmg(device_queue->val, d1, d2, x1, y1, param);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDrotmg(syclQueue_t device_queue, double *d1, double *d2, double *x1, double y1, double *param) {
   auto status = oneapi::mkl::blas::column_major::rotmg(device_queue->val, d1, d2, x1, y1, param);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSscal(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDscal(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha, x, incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCSscal(syclQueue_t device_queue, int64_t n, float alpha, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha, reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZDscal(syclQueue_t device_queue, int64_t n, double alpha, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha, reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCscal(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZscal(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx) {
   auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSswap(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n, x, incx, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDswap(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n, x, incx, y, incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCswap(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, reinterpret_cast<std::complex<float>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZswap(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy) {
   auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, reinterpret_cast<std::complex<double>*>(y), incy);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa), convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, double beta, double *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa), convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, float _Complex beta, float _Complex *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa), convert(transb), m, n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, double _Complex beta, double _Complex *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa), convert(transb), m, n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float alpha, float *a, int64_t lda, int64_t stride_a, float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::syrk_batch(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double alpha, double *a, int64_t lda, int64_t stride_a, double beta, double *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::syrk_batch(device_queue->val, convert(upper_lower), convert(trans), n, k, alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex beta, float _Complex *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::syrk_batch(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, stride_a, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex beta, double _Complex *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::syrk_batch(device_queue->val, convert(upper_lower), convert(trans), n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, stride_a, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float alpha, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double alpha, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val, convert(left_right), convert(upper_lower), convert(trans), convert(unit_diag), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, int64_t stridea, float *x, int64_t incx, int64_t stridex, float beta, float *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemv_batch(device_queue->val, convert(trans), m, n, alpha, a, lda, stridea, x, incx, stridex, beta, y, incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, int64_t stridea, double *x, int64_t incx, int64_t stridex, double beta, double *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemv_batch(device_queue->val, convert(trans), m, n, alpha, a, lda, stridea, x, incx, stridex, beta, y, incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, int64_t stridea, float _Complex *x, int64_t incx, int64_t stridex, float _Complex beta, float _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemv_batch(device_queue->val, convert(trans), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, stridea, reinterpret_cast<std::complex<float>*>(x), incx, stridex, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(y), incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, int64_t stridea, double _Complex *x, int64_t incx, int64_t stridex, double _Complex beta, double _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::gemv_batch(device_queue->val, convert(trans), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, stridea, reinterpret_cast<std::complex<double>*>(x), incx, stridex, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(y), incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float *a, int64_t lda, int64_t stridea, float *x, int64_t incx, int64_t stridex, float *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::dgmm_batch(device_queue->val, convert(left_right), m, n, a, lda, stridea, x, incx, stridex, c, ldc, stridec, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double *a, int64_t lda, int64_t stridea, double *x, int64_t incx, int64_t stridex, double *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::dgmm_batch(device_queue->val, convert(left_right), m, n, a, lda, stridea, x, incx, stridex, c, ldc, stridec, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stridea, float _Complex *x, int64_t incx, int64_t stridex, float _Complex *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::dgmm_batch(device_queue->val, convert(left_right), m, n, reinterpret_cast<std::complex<float>*>(a), lda, stridea, reinterpret_cast<std::complex<float>*>(x), incx, stridex, reinterpret_cast<std::complex<float>*>(c), ldc, stridec, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stridea, double _Complex *x, int64_t incx, int64_t stridex, double _Complex *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::dgmm_batch(device_queue->val, convert(left_right), m, n, reinterpret_cast<std::complex<double>*>(a), lda, stridea, reinterpret_cast<std::complex<double>*>(x), incx, stridex, reinterpret_cast<std::complex<double>*>(c), ldc, stridec, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSaxpy_batch(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx, int64_t stridex, float *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::axpy_batch(device_queue->val, n, alpha, x, incx, stridex, y, incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDaxpy_batch(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx, int64_t stridex, double *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::axpy_batch(device_queue->val, n, alpha, x, incx, stridex, y, incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCaxpy_batch(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x, int64_t incx, int64_t stridex, float _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::axpy_batch(device_queue->val, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(x), incx, stridex, reinterpret_cast<std::complex<float>*>(y), incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZaxpy_batch(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x, int64_t incx, int64_t stridex, double _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::axpy_batch(device_queue->val, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(x), incx, stridex, reinterpret_cast<std::complex<double>*>(y), incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklScopy_batch(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, int64_t stridex, float *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::copy_batch(device_queue->val, n, x, incx, stridex, y, incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDcopy_batch(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, int64_t stridex, double *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::copy_batch(device_queue->val, n, x, incx, stridex, y, incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCcopy_batch(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, int64_t stridex, float _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::copy_batch(device_queue->val, n, reinterpret_cast<std::complex<float>*>(x), incx, stridex, reinterpret_cast<std::complex<float>*>(y), incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZcopy_batch(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, int64_t stridex, double _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::copy_batch(device_queue->val, n, reinterpret_cast<std::complex<double>*>(x), incx, stridex, reinterpret_cast<std::complex<double>*>(y), incy, stridey, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa, onemklTranspose transb, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemmt(device_queue->val, convert(upper_lower), convert(transa), convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa, onemklTranspose transb, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemmt(device_queue->val, convert(upper_lower), convert(transa), convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa, onemklTranspose transb, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemmt(device_queue->val, convert(upper_lower), convert(transa), convert(transb), n, k, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa, onemklTranspose transb, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::gemmt(device_queue->val, convert(upper_lower), convert(transa), convert(transb), n, k, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha, float *ab, int64_t lda, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::imatcopy(device_queue->val, convert(trans), m, n, alpha, ab, lda, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha, double *ab, int64_t lda, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::imatcopy(device_queue->val, convert(trans), m, n, alpha, ab, lda, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float _Complex alpha, float _Complex *ab, int64_t lda, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::imatcopy(device_queue->val, convert(trans), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(ab), lda, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double _Complex alpha, double _Complex *ab, int64_t lda, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::imatcopy(device_queue->val, convert(trans), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(ab), lda, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSomatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::omatcopy(device_queue->val, convert(trans), m, n, alpha, a, lda, b, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDomatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::omatcopy(device_queue->val, convert(trans), m, n, alpha, a, lda, b, ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklComatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::omatcopy(device_queue->val, convert(trans), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZomatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb) {
   auto status = oneapi::mkl::blas::column_major::omatcopy(device_queue->val, convert(trans), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSomatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float beta, float *b, int64_t ldb, float *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::omatadd(device_queue->val, convert(transa), convert(transb), m, n, alpha, a, lda, beta, b, ldb, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDomatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double beta, double *b, int64_t ldb, double *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::omatadd(device_queue->val, convert(transa), convert(transb), m, n, alpha, a, lda, beta, b, ldb, c, ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklComatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex beta, float _Complex *b, int64_t ldb, float _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::omatadd(device_queue->val, convert(transa), convert(transb), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(b), ldb, reinterpret_cast<std::complex<float>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZomatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex beta, double _Complex *b, int64_t ldb, double _Complex *c, int64_t ldc) {
   auto status = oneapi::mkl::blas::column_major::omatadd(device_queue->val, convert(transa), convert(transb), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(b), ldb, reinterpret_cast<std::complex<double>*>(c), ldc);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha, float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::imatcopy_batch(device_queue->val, convert(trans), m, n, alpha, ab, lda, ldb, stride, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha, double *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::imatcopy_batch(device_queue->val, convert(trans), m, n, alpha, ab, lda, ldb, stride, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float _Complex alpha, float _Complex *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::imatcopy_batch(device_queue->val, convert(trans), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(ab), lda, ldb, stride, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double _Complex alpha, double _Complex *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::imatcopy_batch(device_queue->val, convert(trans), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(ab), lda, ldb, stride, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSomatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatcopy_batch(device_queue->val, convert(trans), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDomatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatcopy_batch(device_queue->val, convert(trans), m, n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklComatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatcopy_batch(device_queue->val, convert(trans), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZomatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatcopy_batch(device_queue->val, convert(trans), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSomatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, float alpha, float *a, int64_t lda, int64_t stride_a, float beta, float *b, int64_t ldb, int64_t stride_b, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatadd_batch(device_queue->val, convert(transa), convert(transb), m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDomatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, double alpha, double *a, int64_t lda, int64_t stride_a, double beta, double *b, int64_t ldb, int64_t stride_b, double *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatadd_batch(device_queue->val, convert(transa), convert(transb), m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklComatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex beta, float _Complex *b, int64_t ldb, int64_t stride_b, float _Complex *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatadd_batch(device_queue->val, convert(transa), convert(transb), m, n, static_cast<std::complex<float> >(alpha), reinterpret_cast<std::complex<float>*>(a), lda, stride_a, static_cast<std::complex<float> >(beta), reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, reinterpret_cast<std::complex<float>*>(c), ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZomatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex beta, double _Complex *b, int64_t ldb, int64_t stride_b, double _Complex *c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
   auto status = oneapi::mkl::blas::column_major::omatadd_batch(device_queue->val, convert(transa), convert(transb), m, n, static_cast<std::complex<double> >(alpha), reinterpret_cast<std::complex<double>*>(a), lda, stride_a, static_cast<std::complex<double> >(beta), reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, reinterpret_cast<std::complex<double>*>(c), ldc, stride_c, batch_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

// LAPACK
extern "C" int onemklSpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklSpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<float>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklDpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<double>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklCpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklZpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int onemklSpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklSgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<float>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<double>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int onemklCgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float *d, float *e, float _Complex *tauq, float _Complex *taup, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, d, e, reinterpret_cast<std::complex<float>*>(tauq), reinterpret_cast<std::complex<float>*>(taup), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *d, double *e, double *tauq, double *taup, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *d, float *e, float *tauq, float *taup, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double *d, double *e, double _Complex *tauq, double _Complex *taup, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, d, e, reinterpret_cast<std::complex<double>*>(tauq), reinterpret_cast<std::complex<double>*>(taup), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<float>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int onemklCgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, float _Complex *a, int64_t lda, float *s, float _Complex *u, int64_t ldu, float _Complex *vt, int64_t ldvt, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<float>*>(a), lda, s, reinterpret_cast<std::complex<float>*>(u), ldu, reinterpret_cast<std::complex<float>*>(vt), ldvt, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, double _Complex *a, int64_t lda, double *s, double _Complex *u, int64_t ldu, double _Complex *vt, int64_t ldvt, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<double>*>(a), lda, s, reinterpret_cast<std::complex<double>*>(u), ldu, reinterpret_cast<std::complex<double>*>(vt), ldvt, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, double *a, int64_t lda, double *s, double *u, int64_t ldu, double *vt, int64_t ldvt, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, float *a, int64_t lda, float *s, float *u, int64_t ldu, float *vt, int64_t ldvt, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int onemklCgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, ipiv, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, ipiv, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<float>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int onemklCgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_scratchpad_size<float>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_scratchpad_size<double>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int onemklCgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp(device_queue->val, m, n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp(device_queue->val, m, n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<float>(device_queue->val, m, n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<double>(device_queue->val, m, n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int onemklCgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp_batch(device_queue->val, m, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp_batch(device_queue->val, m, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrfnp_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<float>(device_queue->val, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<double>(device_queue->val, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<std::complex<float>>(device_queue->val, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<std::complex<double>>(device_queue->val, n, lda);
   return scratchpad_size;
}

extern "C" int onemklCgetri(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, reinterpret_cast<std::complex<float>*>(a), lda, ipiv, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetri(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetri(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetri(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, reinterpret_cast<std::complex<double>*>(a), lda, ipiv, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<double>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int onemklCgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, ipiv, reinterpret_cast<std::complex<float>*>(b), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, ipiv, reinterpret_cast<std::complex<double>*>(b), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<float>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int onemklCgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, ipiv, stride_ipiv, reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, ipiv, stride_ipiv, reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrsnp_batch_scratchpad_size<float>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrsnp_batch_scratchpad_size<double>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrsnp_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrsnp_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int onemklCgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrsnp_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrsnp_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrsnp_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrsnp_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCheev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heev_scratchpad_size<std::complex<float>>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZheev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heev_scratchpad_size<std::complex<double>>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklCheev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heev(device_queue->val, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, w, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZheev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heev(device_queue->val, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, w, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<float>>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<double>>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklCheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heevd(device_queue->val, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, w, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heevd(device_queue->val, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, w, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklChegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklZhegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int onemklChegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hegvd(device_queue->val, itype, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, w, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hegvd(device_queue->val, itype, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, w, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklChetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZhetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklChetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *d, float *e, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, d, e, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *d, double *e, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, d, e, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklChetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, ipiv, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZhetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, ipiv, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklChetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZhetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklSorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgbr(device_queue->val, convert(vec), m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgbr(device_queue->val, convert(vec), m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgbr_scratchpad_size<float>(device_queue->val, convert(vect), m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgbr_scratchpad_size<double>(device_queue->val, convert(vect), m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklSorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<float>(device_queue->val, m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<double>(device_queue->val, m, n, k, lda);
   return scratchpad_size;
}

extern "C" int onemklDorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<float>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklDormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<double>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int onemklDormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormqr(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormqr(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::steqr_scratchpad_size<float>(device_queue->val, convert(compz), n, ldz);
   return scratchpad_size;
}

extern "C" int64_t onemklDsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::steqr_scratchpad_size<double>(device_queue->val, convert(compz), n, ldz);
   return scratchpad_size;
}

extern "C" int64_t onemklCsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::steqr_scratchpad_size<std::complex<float>>(device_queue->val, convert(compz), n, ldz);
   return scratchpad_size;
}

extern "C" int64_t onemklZsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::steqr_scratchpad_size<std::complex<double>>(device_queue->val, convert(compz), n, ldz);
   return scratchpad_size;
}

extern "C" int onemklCsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, float *d, float *e, float _Complex *z, int64_t ldz, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::steqr(device_queue->val, convert(compz), n, d, e, reinterpret_cast<std::complex<float>*>(z), ldz, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, double *d, double *e, double *z, int64_t ldz, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::steqr(device_queue->val, convert(compz), n, d, e, z, ldz, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, float *d, float *e, float *z, int64_t ldz, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::steqr(device_queue->val, convert(compz), n, d, e, z, ldz, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, double *d, double *e, double _Complex *z, int64_t ldz, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::steqr(device_queue->val, convert(compz), n, d, e, reinterpret_cast<std::complex<double>*>(z), ldz, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsyev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syev_scratchpad_size<float>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDsyev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syev_scratchpad_size<double>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklDsyev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syev(device_queue->val, convert(jobz), convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syev(device_queue->val, convert(jobz), convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<float>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklDsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevd(device_queue->val, convert(jobz), convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevd(device_queue->val, convert(jobz), convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsyevx_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, int64_t lda, float vl, float vu, int64_t il, int64_t iu, float abstol, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevx_scratchpad_size<float>(device_queue->val, convert(jobz), convert(range), convert(uplo), n, lda, vl, vu, il, iu, abstol, ldz);
   return scratchpad_size;
}

extern "C" int64_t onemklDsyevx_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, int64_t lda, double vl, double vu, int64_t il, int64_t iu, double abstol, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevx_scratchpad_size<double>(device_queue->val, convert(jobz), convert(range), convert(uplo), n, lda, vl, vu, il, iu, abstol, ldz);
   return scratchpad_size;
}

extern "C" int onemklDsyevx(syclQueue_t device_queue, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, double *a, int64_t lda, double vl, double vu, int64_t il, int64_t iu, double abstol, int64_t *m, double *w, double *z, int64_t ldz, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevx(device_queue->val, convert(jobz), convert(range), convert(uplo), n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsyevx(syclQueue_t device_queue, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, float *a, int64_t lda, float vl, float vu, int64_t il, int64_t iu, float abstol, int64_t *m, float *w, float *z, int64_t ldz, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevx(device_queue->val, convert(jobz), convert(range), convert(uplo), n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklDsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int onemklDsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *b, int64_t ldb, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvd(device_queue->val, itype, convert(jobz), convert(uplo), n, a, lda, b, ldb, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *b, int64_t ldb, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvd(device_queue->val, itype, convert(jobz), convert(uplo), n, a, lda, b, ldb, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsygvx_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb, float vl, float vu, int64_t il, int64_t iu, float abstol, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvx_scratchpad_size<float>(device_queue->val, itype, convert(jobz), convert(range), convert(uplo), n, lda, ldb, vl, vu, il, iu, abstol, ldz);
   return scratchpad_size;
}

extern "C" int64_t onemklDsygvx_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb, double vl, double vu, int64_t il, int64_t iu, double abstol, int64_t ldz) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvx_scratchpad_size<double>(device_queue->val, itype, convert(jobz), convert(range), convert(uplo), n, lda, ldb, vl, vu, il, iu, abstol, ldz);
   return scratchpad_size;
}

extern "C" int onemklDsygvx(syclQueue_t device_queue, int64_t itype, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *b, int64_t ldb, double vl, double vu, int64_t il, int64_t iu, double abstol, int64_t *m, double *w, double *z, int64_t ldz, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvx(device_queue->val, itype, convert(jobz), convert(range), convert(uplo), n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsygvx(syclQueue_t device_queue, int64_t itype, onemklCompz jobz, onemklRangev range, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *b, int64_t ldb, float vl, float vu, int64_t il, int64_t iu, float abstol, int64_t *m, float *w, float *z, int64_t ldz, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvx(device_queue->val, itype, convert(jobz), convert(range), convert(uplo), n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrd_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrd_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklDsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *d, double *e, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, a, lda, d, e, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *d, float *e, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, a, lda, d, e, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklStrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<float>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklDtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<double>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklCtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklZtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   return scratchpad_size;
}

extern "C" int onemklCtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(b), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklStrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(b), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungbr(device_queue->val, convert(vec), m, n, k, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungbr(device_queue->val, convert(vec), m, n, k, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<float>>(device_queue->val, convert(vect), m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<std::complex<double>>(device_queue->val, convert(vect), m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<float>>(device_queue->val, m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_scratchpad_size<std::complex<double>>(device_queue->val, m, n, k, lda);
   return scratchpad_size;
}

extern "C" int onemklCungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklZunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int onemklCunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmqr(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(c), ldc, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmqr(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(c), ldc, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<float>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<double>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   return scratchpad_size;
}

extern "C" int onemklSormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormrq(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormrq(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormrq_scratchpad_size<float>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklDormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormrq_scratchpad_size<double>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int onemklCunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmrq(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(c), ldc, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmrq(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(c), ldc, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklZunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int onemklSsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, ipiv, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, ipiv, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklSorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgtr_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgtr_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklCungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int onemklSormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<float>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklDormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<double>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
}

extern "C" int onemklCunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(c), ldc, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(c), ldc, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklCunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklZunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
}

extern "C" int onemklSpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr_batch(device_queue->val, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr_batch(device_queue->val, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr_batch(device_queue->val, m, n, k, reinterpret_cast<std::complex<float>*>(a), lda, stride_a, reinterpret_cast<std::complex<float>*>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr_batch(device_queue->val, m, n, k, reinterpret_cast<std::complex<double>*>(a), lda, stride_a, reinterpret_cast<std::complex<double>*>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stridea, float *b, int64_t ldb, int64_t strideb, int64_t batchsize, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gels_batch(device_queue->val, convert(trans), m, n, nrhs, a, lda, stridea, b, ldb, strideb, batchsize, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stridea, double *b, int64_t ldb, int64_t strideb, int64_t batchsize, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gels_batch(device_queue->val, convert(trans), m, n, nrhs, a, lda, stridea, b, ldb, strideb, batchsize, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stridea, float _Complex *b, int64_t ldb, int64_t strideb, int64_t batchsize, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gels_batch(device_queue->val, convert(trans), m, n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, stridea, reinterpret_cast<std::complex<float>*>(b), ldb, strideb, batchsize, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stridea, double _Complex *b, int64_t ldb, int64_t strideb, int64_t batchsize, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gels_batch(device_queue->val, convert(trans), m, n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, stridea, reinterpret_cast<std::complex<double>*>(b), ldb, strideb, batchsize, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int64_t onemklSpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklSpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<float>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<double>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklSgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<float>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<double>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklSorgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_batch_scratchpad_size<float>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDorgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_batch_scratchpad_size<double>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklSgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gels_batch_scratchpad_size<float>(device_queue->val, convert(trans), m, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gels_batch_scratchpad_size<double>(device_queue->val, convert(trans), m, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gels_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), m, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gels_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), m, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   return scratchpad_size;
}

// other

// oneMKL keeps a cache of SYCL queues and tries to destroy them when unloading the library.
// that is incompatible with oneAPI.jl destroying queues before that, so expose a function
// to manually wipe the device cache when we're destroying queues.

namespace oneapi {
namespace mkl {
namespace gpu {
int clean_gpu_caches();
}
}
}

extern "C" void onemklDestroy() {
    oneapi::mkl::gpu::clean_gpu_caches();
}
