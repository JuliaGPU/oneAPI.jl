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

// gemm

// https://spec.oneapi.io/versions/1.0-rev-1/elements/oneMKL/source/domains/blas/gemm.html

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

extern "C" int onemklSgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, float alpha, const float *A, int64_t lda,
                           const float *B, int64_t ldb, float beta, float *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklDgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, double alpha, const double *A,
                           int64_t lda, const double *B, int64_t ldb,
                           double beta, double *C, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklCgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, float _Complex alpha,
                           const float _Complex *A, int64_t lda,
                           const float _Complex *B, int64_t ldb,
                           float _Complex beta, float _Complex *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(
        device_queue->val, convert(transA), convert(transB), m, n, k, alpha,
        reinterpret_cast<const std::complex<float> *>(A), lda,
        reinterpret_cast<const std::complex<float> *>(B), ldb, beta,
        reinterpret_cast<std::complex<float> *>(C), ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" int onemklZgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, double _Complex alpha,
                           const double _Complex *A, int64_t lda,
                           const double _Complex *B, int64_t ldb,
                           double _Complex beta, double _Complex *C,
                           int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::gemm(
        device_queue->val, convert(transA), convert(transB), m, n, k, alpha,
        reinterpret_cast<const std::complex<double> *>(A), lda,
        reinterpret_cast<const std::complex<double> *>(B), ldb, beta,
        reinterpret_cast<std::complex<double> *>(C), ldc);
    __FORCE_MKL_FLUSH__(status);
    return 0;
}

extern "C" void onemklHgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
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

}

extern "C" void onemklSgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklDgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklCgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklZgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklHgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklSgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        float alpha, const float *a, int64_t lda, int64_t stridea,
                        const float *b, int64_t ldb, int64_t strideb, float beta,
                        float *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, alpha, a, lda, stridea,
                                                    b, ldb, strideb, beta, c, ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                        double alpha, const double *a, int64_t lda, int64_t stridea,
                        const double *b, int64_t ldb, int64_t strideb, double beta,
                        double *c, int64_t ldc, int64_t stridec, int64_t batch_size) {
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val, convert(transa),
                                                    convert(transb), m, n, k, alpha, a, lda, stridea,
                                                    b, ldb, strideb, beta, c, ldc, stridec, batch_size);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklZgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
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
}

extern "C" void onemklSsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            float alpha, const float *a, int64_t lda, const float *b,
                            int64_t ldb, float beta, float *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue->val,
                                        convert(left_right), convert(upper_lower),
                                        m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            double alpha, const double *a, int64_t lda, const double *b,
                            int64_t ldb, double beta, double *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *b, int64_t ldb, float _Complex beta,
                            float _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(b),
                                          ldb, beta, reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *b, int64_t ldb, double _Complex beta,
                            double _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a), lda,
                                          reinterpret_cast<const std::complex<double> *>(b), ldb,
                                          static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, float alpha,
                            const float *a, int64_t lda, float beta, float *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, alpha, a, lda, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, double alpha,
                            const double *a, int64_t lda, double beta, double *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, alpha, a, lda, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k,
                            float _Complex alpha, const float _Complex *a,
                            int64_t lda, float _Complex beta,
                            float _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(a), lda,
                                        static_cast<std::complex<float> >(beta),
                                        reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k,
                            double _Complex alpha, const double _Complex *a,
                            int64_t lda, double _Complex beta,
                            double _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a), lda,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                             int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                             const float *b, int64_t ldb, float beta, float *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                             int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                             const double *b, int64_t ldb, double beta, double *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                             int64_t n, int64_t k, float _Complex alpha, const float _Complex *a,
                             int64_t lda, const float _Complex *b, int64_t ldb, float _Complex beta,
                             float _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(a), lda,
                                        reinterpret_cast<const std::complex<float> *>(b), ldb,
                                        static_cast<std::complex<float> >(beta),
                                        reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                             int64_t n, int64_t k, double _Complex alpha, const double _Complex *a,
                             int64_t lda, const double _Complex *b, int64_t ldb, double _Complex beta,
                             double _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::syr2k(device_queue->val, convert(upper_lower),
                                        convert(trans), n, k, static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a), lda,
                                        reinterpret_cast<const std::complex<double> *>(b), ldb,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrmm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo uppler_lower, onemklTranspose trans,
                            onemklDiag diag, int64_t m, int64_t n, float alpha,
                            const float *a, int64_t lda, float *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right),
                                          convert(uppler_lower), convert(trans),
                                          convert(diag), m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrmm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo uppler_lower, onemklTranspose trans,
                            onemklDiag diag, int64_t m, int64_t n, double alpha,
                            const double *a, int64_t lda, double *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right),
                                          convert(uppler_lower), convert(trans),
                                          convert(diag), m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrmm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo uppler_lower, onemklTranspose trans,
                            onemklDiag diag, int64_t m, int64_t n, float _Complex alpha,
                            const float _Complex *a, int64_t lda, float _Complex *b,
                            int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right),
                                          convert(uppler_lower), convert(trans),
                                          convert(diag), m, n, static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a), lda,
                                          reinterpret_cast<std::complex<float> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrmm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo uppler_lower, onemklTranspose trans,
                            onemklDiag diag, int64_t m, int64_t n, double _Complex alpha,
                            const double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trmm(device_queue->val, convert(left_right),
                                          convert(uppler_lower), convert(trans),
                                          convert(diag), m, n, static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a), lda,
                                          reinterpret_cast<std::complex<double> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                            onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                            float alpha, const float *a, int64_t lda, float *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right),
                                        convert(upper_lower), convert(transa), convert(unit_diag),
                                        m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                            onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                            double alpha, const double *a, int64_t lda, double *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right),
                                        convert(upper_lower), convert(transa), convert(unit_diag),
                                        m, n, alpha, a, lda, b, ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                            onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                            float _Complex alpha, const float _Complex *a, int64_t lda, float _Complex *b,
                            int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right),
                                        convert(upper_lower), convert(transa), convert(unit_diag),
                                        m, n, static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<const std::complex<float> *>(a), lda,
                                        reinterpret_cast<std::complex<float> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                            onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            double _Complex *b, int64_t ldb) {
    auto status = oneapi::mkl::blas::column_major::trsm(device_queue->val, convert(left_right),
                                        convert(upper_lower), convert(transa), convert(unit_diag),
                                        m, n, static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a), lda,
                                        reinterpret_cast<std::complex<double> *>(b), ldb);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrsmBatched(syclQueue_t device_queue, onemklSide left_right,
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
}

extern "C" void onemklDtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
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
}

extern "C" void onemklCtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
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
}

extern "C" void onemklZtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
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
}

extern "C" void onemklChemm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            float _Complex alpha, const float _Complex *a,
                            int64_t lda, const float _Complex *b, int64_t ldb,
                            float _Complex beta, float _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::hemm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(b),
                                          ldb, static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhemm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            double _Complex alpha, const double _Complex *a,
                            int64_t lda, const double _Complex *b, int64_t ldb,
                            double _Complex beta, double _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::hemm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(b),
                                          ldb, static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCherk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, float alpha,
                            const float _Complex *a, int64_t lda, float beta,
                            float _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::herk(device_queue->val, convert(upper_lower),
                                          convert(trans), n, k, alpha,
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, beta, reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZherk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, double alpha,
                            const double _Complex *a, int64_t lda, double beta,
                            double _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::herk(device_queue->val, convert(upper_lower),
                                          convert(trans), n, k, alpha,
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, beta, reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCher2k(syclQueue_t device_queue, onemklUplo upper_lower,
                             onemklTranspose trans, int64_t n, int64_t k,
                             float _Complex alpha, const float _Complex *a,
                             int64_t lda, const float _Complex *b, int64_t ldb,
                             float beta, float _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::her2k(device_queue->val, convert(upper_lower),
                                           convert(trans), n, k, static_cast<std::complex<float> >(alpha),
                                           reinterpret_cast<const std::complex<float> *>(a), lda,
                                           reinterpret_cast<const std::complex<float> *>(b), ldb,
                                           beta, reinterpret_cast<std::complex<float> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZher2k(syclQueue_t device_queue, onemklUplo upper_lower,
                             onemklTranspose trans, int64_t n, int64_t k,
                             double _Complex alpha, const double _Complex *a,
                             int64_t lda, const double _Complex *b, int64_t ldb,
                             double beta, double _Complex *c, int64_t ldc) {
    auto status = oneapi::mkl::blas::column_major::her2k(device_queue->val, convert(upper_lower),
                                           convert(trans), n, k, static_cast<std::complex<double> >(alpha),
                                           reinterpret_cast<const std::complex<double> *>(a), lda,
                                           reinterpret_cast<const std::complex<double> *>(b), ldb,
                                           beta, reinterpret_cast<std::complex<double> *>(c), ldc);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgbmv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            float alpha, const float *a, int64_t lda,
                            const float *x, int64_t incx, float beta, float *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val,
                                convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgbmv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            double alpha, const double *a, int64_t lda,
                            const double *x, int64_t incx, double beta, double *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans),
                                    m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgbmv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans),
                                    m, n, kl, ku, static_cast<std::complex<float> >(alpha),
                                    reinterpret_cast<const std::complex<float> *>(a),
                                    lda, reinterpret_cast<const std::complex<float> *>(x),
                                    incx, static_cast<std::complex<float> >(beta),
                                    reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgbmv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, int64_t kl, int64_t ku,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gbmv(device_queue->val, convert(trans), m,
                                        n, kl, ku, static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<const std::complex<double> *>(x), incx,
                                        static_cast<std::complex<double> >(beta),
                                        reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklHdot(syclQueue_t device_queue, int64_t n,
                           const short *x, int64_t incx, const short *y,
                           int64_t incy, short *result) {
    auto status = oneapi::mkl::blas::column_major::dot(device_queue->val, n,
                                        reinterpret_cast<const sycl::half *>(x),
                                        incx, reinterpret_cast<const sycl::half *>(y),
                                        incy, reinterpret_cast<sycl::half *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSdot(syclQueue_t device_queue, int64_t n,
                           const float *x, int64_t incx, const float *y,
                           int64_t incy, float *result) {
    auto status = oneapi::mkl::blas::column_major::dot(device_queue->val, n, x,
                                                       incx, y, incy, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDdot(syclQueue_t device_queue, int64_t n,
                           const double *x, int64_t incx, const double *y,
                           int64_t incy, double *result) {
    auto status = oneapi::mkl::blas::column_major::dot(device_queue->val, n, x,
                                                       incx, y, incy, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCdotc(syclQueue_t device_queue, int64_t n,
                           const float _Complex *x, int64_t incx, const float _Complex *y,
                           int64_t incy, float _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotc(device_queue->val, n,
                                                reinterpret_cast<const std::complex<float> *>(x), incx,
                                                reinterpret_cast<const std::complex<float> *>(y), incy,
                                                reinterpret_cast<std::complex<float> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdotc(syclQueue_t device_queue, int64_t n,
                           const double _Complex *x, int64_t incx, const double _Complex *y,
                           int64_t incy, double _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotc(device_queue->val, n,
                                                reinterpret_cast<const std::complex<double> *>(x), incx,
                                                reinterpret_cast<const std::complex<double> *>(y), incy,
                                                reinterpret_cast<std::complex<double> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCdotu(syclQueue_t device_queue, int64_t n,
                           const float _Complex *x, int64_t incx, const float _Complex *y,
                           int64_t incy, float _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotu(device_queue->val, n,
                                                reinterpret_cast<const std::complex<float> *>(x), incx,
                                                reinterpret_cast<const std::complex<float> *>(y), incy,
                                                reinterpret_cast<std::complex<float> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdotu(syclQueue_t device_queue, int64_t n,
                           const double _Complex *x, int64_t incx, const double _Complex *y,
                           int64_t incy, double _Complex *result) {
    auto status = oneapi::mkl::blas::column_major::dotu(device_queue->val, n,
                                                reinterpret_cast<const std::complex<double> *>(x), incx,
                                                reinterpret_cast<const std::complex<double> *>(y), incy,
                                                reinterpret_cast<std::complex<double> *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSasum(syclQueue_t device_queue, int64_t n,
                            const float *x, int64_t incx,
                            float *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n, x,
                                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDasum(syclQueue_t device_queue, int64_t n,
                            const double *x, int64_t incx,
                            double *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n, x,
                                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCasum(syclQueue_t device_queue, int64_t n,
                            const float _Complex *x, int64_t incx,
                            float *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n,
                                        reinterpret_cast<const std::complex<float> *>(x),
                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZasum(syclQueue_t device_queue, int64_t n,
                            const double _Complex *x, int64_t incx,
                            double *result) {
    auto status = oneapi::mkl::blas::column_major::asum(device_queue->val, n,
                                        reinterpret_cast<const std::complex<double> *>(x),
                                        incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklHaxpy(syclQueue_t device_queue, int64_t n, uint16_t alpha,
                            const short *x, int64_t incx, short *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n,
                                        sycl::bit_cast<sycl::half>(alpha),
                                        reinterpret_cast<const sycl::half *>(x),
                                        incx, reinterpret_cast<sycl::half *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSaxpy(syclQueue_t device_queue, int64_t n, float alpha,
                            const float *x, int64_t incx, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, alpha, x,
                                                incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDaxpy(syclQueue_t device_queue, int64_t n, double alpha,
                            const double *x, int64_t incx, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, alpha, x,
                                                incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCaxpy(syclQueue_t device_queue, int64_t n, float _Complex alpha,
                            const float _Complex *x, int64_t incx, float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, static_cast<std::complex<float> >(alpha),
                            reinterpret_cast<const std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZaxpy(syclQueue_t device_queue, int64_t n, double _Complex alpha,
                            const double _Complex *x, int64_t incx, double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, static_cast<std::complex<double> >(alpha),
                            reinterpret_cast<const std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSaxpby(syclQueue_t device_queue, int64_t n, float alpha,
                             const float *x, int64_t incx, float beta, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, alpha, x,
                                                incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDaxpby(syclQueue_t device_queue, int64_t n, double alpha,
                             const double *x, int64_t incx, double beta, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, alpha, x,
                                                incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCaxpby(syclQueue_t device_queue, int64_t n, float _Complex alpha,
                             const float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, static_cast<std::complex<float> >(alpha),
                            reinterpret_cast<const std::complex<float> *>(x), incx, static_cast<std::complex<float> >(beta),
                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZaxpby(syclQueue_t device_queue, int64_t n, double _Complex alpha,
                             const double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpby(device_queue->val, n, static_cast<std::complex<double> >(alpha),
                            reinterpret_cast<const std::complex<double> *>(x), incx, static_cast<std::complex<double> >(beta),
                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSrot(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy, float c, float s) {
    auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, x, incx, y, incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDrot(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy, double c, double s) {
    auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n, x, incx, y, incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCrot(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float c, float _Complex s) {
    auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n,
                            reinterpret_cast<std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy, c, static_cast<std::complex<float> >(s));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZrot(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double c, double _Complex s) {
    auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n,
                            reinterpret_cast<std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy, c, static_cast<std::complex<double> >(s));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsrot(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float c, float s) {
    auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n,
                            reinterpret_cast<std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdrot(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double c, double s) {
    auto status = oneapi::mkl::blas::column_major::rot(device_queue->val, n,
                            reinterpret_cast<std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy, c, s);
    __FORCE_MKL_FLUSH__(status);
}

// Support Level-1: SCAL primitive
extern "C" void onemklDscal(syclQueue_t device_queue, int64_t n, double alpha,
                            double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha,
                                                    x, incx);
    __FORCE_MKL_FLUSH__(status);

}

extern "C" void onemklHscal(syclQueue_t device_queue, int64_t n, uint16_t alpha,
                            short *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, sycl::bit_cast<sycl::half>(alpha),
                                                        reinterpret_cast<sycl::half *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSscal(syclQueue_t device_queue, int64_t n, float alpha,
                            float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha,
                                                         x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCscal(syclQueue_t device_queue, int64_t n,
                            float _Complex alpha, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n,
                                        static_cast<std::complex<float> >(alpha),
                                        reinterpret_cast<std::complex<float> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsscal(syclQueue_t device_queue, int64_t n,
                            float alpha, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha,
                                        reinterpret_cast<std::complex<float> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZscal(syclQueue_t device_queue, int64_t n,
                            double _Complex alpha, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n,
                                        static_cast<std::complex<double> >(alpha),
                                        reinterpret_cast<std::complex<double> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZdscal(syclQueue_t device_queue, int64_t n,
                            double alpha, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::scal(device_queue->val, n, alpha,
                                        reinterpret_cast<std::complex<double> *>(x),incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgemv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, float alpha, const float *a,
                            int64_t lda, const float *x, int64_t incx, float beta,
                            float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans),
                                            m, n, alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgemv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, double alpha, const double *a,
                            int64_t lda, const double *x, int64_t incx, double beta,
                            double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans),
                                            m, n, alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgemv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, float _Complex alpha,
                            const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx,
                            float _Complex beta, float _Complex *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans), m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(a), lda,
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            static_cast<std::complex<float> >(beta),
                                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgemv(syclQueue_t device_queue, onemklTranspose trans,
                            int64_t m, int64_t n, double _Complex alpha,
                            const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx,
                            double _Complex beta, double _Complex *y,
                            int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::gemv(device_queue->val, convert(trans), m, n,
                                            static_cast<std::complex<double> >(alpha),
                                            reinterpret_cast<const std::complex<double> *>(a), lda,
                                            reinterpret_cast<const std::complex<double> *>(x), incx,
                                            static_cast<std::complex<double> >(beta),
                                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSger(syclQueue_t device_queue, int64_t m, int64_t n, float alpha,
                           const float *x, int64_t incx, const float *y, int64_t incy,
                           float *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::ger(device_queue->val, m, n, alpha, x,
                                                    incx, y, incy, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDger(syclQueue_t device_queue, int64_t m, int64_t n, double alpha,
                           const double *x, int64_t incx, const double *y, int64_t incy,
                           double *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::ger(device_queue->val, m, n, alpha, x,
                                                    incx, y, incy, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgerc(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex alpha,
                           const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                           float _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::gerc(device_queue->val, m, n,
                                            static_cast<std::complex<float> >(alpha),
                                            reinterpret_cast<const std::complex<float> *>(x), incx,
                                            reinterpret_cast<const std::complex<float> *>(y), incy,
                                            reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgerc(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex alpha,
                           const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                           double _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::gerc(device_queue->val, m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChemv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hemv(device_queue->val, convert(uplo), n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(x), incx,
                                          static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhemv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hemv(device_queue->val, convert(uplo), n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(x), incx,
                                          static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                            int64_t k, float _Complex alpha, const float _Complex *a,
                            int64_t lda, const float _Complex *x, int64_t incx, float _Complex beta,
                            float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hbmv(device_queue->val, convert(uplo), n,
                                          k, static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(x),
                                          incx, static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                            int64_t k, double _Complex alpha, const double _Complex *a,
                            int64_t lda, const double _Complex *x, int64_t incx, double _Complex beta,
                            double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::hbmv(device_queue->val, convert(uplo), n,
                                          k, static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<const std::complex<double> *>(x),
                                          incx, static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCher(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float _Complex *x, int64_t incx, float _Complex *a,
                           int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her(device_queue->val, convert(uplo), n, alpha,
                                        reinterpret_cast<const std::complex<float> *>(x), incx,
                                        reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZher(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double _Complex *x, int64_t incx, double _Complex *a,
                           int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her(device_queue->val, convert(uplo), n, alpha,
                                        reinterpret_cast<const std::complex<double> *>(x), incx,
                                        reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCher2(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                            const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                            float _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her2(device_queue->val, convert(uplo), n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(x), incx,
                                          reinterpret_cast<const std::complex<float> *>(y), incy,
                                          reinterpret_cast<std::complex<float> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZher2(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                            const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                            double _Complex *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::her2(device_queue->val, convert(uplo), n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(x), incx,
                                          reinterpret_cast<const std::complex<double> *>(y), incy,
                                          reinterpret_cast<std::complex<double> *>(a), lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t k,
                            float alpha, const float *a, int64_t lda, const float *x,
                            int64_t incx, float beta, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::sbmv(device_queue->val, convert(uplo), n, k,
                                                    alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t k,
                            double alpha, const double *a, int64_t lda, const double *x,
                            int64_t incx, double beta, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::sbmv(device_queue->val, convert(uplo), n, k,
                                                    alpha, a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsymv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float alpha,
                            const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                            float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::symv(device_queue->val, convert(uplo), n, alpha,
                                                    a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsymv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double alpha,
                            const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                            double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::symv(device_queue->val, convert(uplo), n, alpha,
                                                    a, lda, x, incx, beta, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float *x, int64_t incx, float *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::syr(device_queue->val, convert(uplo), n, alpha,
                                                    x, incx, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsyr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double *x, int64_t incx, double *a, int64_t lda) {
    auto status = oneapi::mkl::blas::column_major::syr(device_queue->val, convert(uplo), n, alpha,
                                                    x, incx, a, lda);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStbmv(syclQueue_t device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float *a, int64_t lda, float *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(uplo), convert(trans),
                                                        convert(diag), n, k, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtbmv(syclQueue_t device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double *a, int64_t lda, double *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(uplo), convert(trans),
                                                    convert(diag), n, k, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtbmv(syclQueue_t device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(uplo), convert(trans),
                                            convert(diag), n, k, reinterpret_cast<const std::complex<float> *>(a),
                                            lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtbmv(syclQueue_t device_queue, onemklUplo uplo,
                            onemklTranspose trans, onemklDiag diag, int64_t n,
                            int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::tbmv(device_queue->val, convert(uplo), convert(trans),
                                        convert(diag), n, k, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

// trmv - level2
extern "C" void onemklStrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(uplo), convert(trans),
                                        convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(uplo), convert(trans),
                                        convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                        lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trmv(device_queue->val, convert(uplo), convert(trans),
                                        convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                        lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

// trsv
extern "C" void onemklStrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(uplo), convert(trans),
                                          convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                            int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(uplo), convert(trans),
                                          convert(diag), n, a, lda, x, incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const float  _Complex *a, int64_t lda,
                            float _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(uplo), convert(trans),
                                          convert(diag), n, reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<std::complex<float> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                            onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda,
                            double _Complex *x, int64_t incx) {
    auto status = oneapi::mkl::blas::column_major::trsv(device_queue->val, convert(uplo), convert(trans),
                                          convert(diag), n, reinterpret_cast<const std::complex<double> *>(a),
                                          lda, reinterpret_cast<std::complex<double> *>(x), incx);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklHnrm2(syclQueue_t device_queue, int64_t n, const short *x,
                            int64_t incx, short *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n,
                        reinterpret_cast<const sycl::half *>(x), incx,
                        reinterpret_cast<sycl::half *>(result));
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDnrm2(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, double *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSnrm2(syclQueue_t device_queue, int64_t n, const float *x,
                            int64_t incx, float *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCnrm2(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, float *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n,
                    reinterpret_cast<const std::complex<float> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZnrm2(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, double *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n,
                    reinterpret_cast<const std::complex<double> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDcopy(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklScopy(syclQueue_t device_queue, int64_t n, const float *x,
                            int64_t incx, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZcopy(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, double _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n,
        reinterpret_cast<const std::complex<double> *>(x), incx,
        reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCcopy(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::copy(device_queue->val, n,
        reinterpret_cast<const std::complex<float> *>(x), incx,
        reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDamax(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklSamax(syclQueue_t device_queue, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZamax(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCamax(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDamin(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklSamin(syclQueue_t device_queue, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, x, incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklZamin(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}
extern "C" void onemklCamin(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSswap(syclQueue_t device_queue, int64_t n, float *x, int64_t incx,\
                            float *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDswap(syclQueue_t device_queue, int64_t n, double *x, int64_t incx,
                            double *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n, x, incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCswap(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx,
                            float _Complex *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n,
                            reinterpret_cast<std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZswap(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx,
                            double _Complex *y, int64_t incy){
    auto status = oneapi::mkl::blas::column_major::swap(device_queue->val, n,
                            reinterpret_cast<std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

// LAPACK
extern "C" void onemklCgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float *d, float *e, float _Complex *tauq, float _Complex *taup, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, d, e, reinterpret_cast<std::complex<float> *>(tauq), reinterpret_cast<std::complex<float> *>(taup), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *d, double *e, double *tauq, double *taup, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *d, float *e, float *tauq, float *taup, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double *d, double *e, double _Complex *tauq, double _Complex *taup, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, d, e, reinterpret_cast<std::complex<double> *>(tauq), reinterpret_cast<std::complex<double> *>(taup), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetri(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetri(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetri(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetri(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(b), ldb, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(b), ldb, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, double *a, int64_t lda, double *s, double *u, int64_t ldu, double *vt, int64_t ldvt, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, float *a, int64_t lda, float *s, float *u, int64_t ldu, float *vt, int64_t ldvt, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, float _Complex *a, int64_t lda, float *s, float _Complex *u, int64_t ldu, float _Complex *vt, int64_t ldvt, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<float> *>(a), lda, s, reinterpret_cast<std::complex<float> *>(u), ldu, reinterpret_cast<std::complex<float> *>(vt), ldvt, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, double _Complex *a, int64_t lda, double *s, double _Complex *u, int64_t ldu, double _Complex *vt, int64_t ldvt, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, convert(jobu), convert(jobvt), m, n, reinterpret_cast<std::complex<double> *>(a), lda, s, reinterpret_cast<std::complex<double> *>(u), ldu, reinterpret_cast<std::complex<double> *>(vt), ldvt, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heevd(device_queue->val, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, w, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heevd(device_queue->val, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, w, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hegvd(device_queue->val, itype, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(b), ldb, w, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hegvd(device_queue->val, itype, convert(jobz), convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(b), ldb, w, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *d, float *e, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, d, e, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *d, double *e, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, d, e, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgbr(device_queue->val, convert(vec), m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgbr(device_queue->val, convert(vec), m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormrq(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormrq(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormqr(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormqr(device_queue->val, convert(side), convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(b), ldb, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(b), ldb, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevd(device_queue->val, convert(jobz), convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevd(device_queue->val, convert(jobz), convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *b, int64_t ldb, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvd(device_queue->val, itype, convert(jobz), convert(uplo), n, a, lda, b, ldb, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *b, int64_t ldb, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvd(device_queue->val, itype, convert(jobz), convert(uplo), n, a, lda, b, ldb, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *d, double *e, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, a, lda, d, e, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *d, float *e, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, a, lda, d, e, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(b), ldb, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(b), ldb, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungbr(device_queue->val, convert(vec), m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungbr(device_queue->val, convert(vec), m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmrq(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(c), ldc, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmrq(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(c), ldc, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmqr(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(c), ldc, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmqr(device_queue->val, convert(side), convert(trans), m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(c), ldc, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(c), ldc, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmtr(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(c), ldc, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, reinterpret_cast<std::complex<float> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, reinterpret_cast<std::complex<double> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetri_batch(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetri_batch(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetri_batch(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetri_batch(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, ipiv, stride_ipiv, reinterpret_cast<std::complex<float> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, ipiv, stride_ipiv, reinterpret_cast<std::complex<double> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr_batch(device_queue->val, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr_batch(device_queue->val, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, reinterpret_cast<std::complex<float> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, reinterpret_cast<std::complex<double> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr_batch(device_queue->val, m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, reinterpret_cast<std::complex<float> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr_batch(device_queue->val, m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, reinterpret_cast<std::complex<double> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
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

extern "C" int64_t onemklSgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
   return scratchpad_size;
}

extern "C" int64_t onemklDgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
   return scratchpad_size;
}

extern "C" int64_t onemklCgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
   return scratchpad_size;
}

extern "C" int64_t onemklZgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(device_queue->val, convert(jobu), convert(jobvt), m, n, lda, ldu, ldvt);
   return scratchpad_size;
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

extern "C" int64_t onemklCheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<float>>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<double>>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklChegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<float>>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklZhegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklChetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZhetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklChetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZhetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklSorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgbr_scratchpad_size<float>(device_queue->val, convert(vect), m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgbr_scratchpad_size<double>(device_queue->val, convert(vect), m, n, k, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklSorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgtr_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgtr_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
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

extern "C" int64_t onemklSormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormrq_scratchpad_size<float>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklDormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormrq_scratchpad_size<double>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklSormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<float>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklDormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<double>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklSormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<float>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklDormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<double>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
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

extern "C" int64_t onemklSsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<float>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(device_queue->val, convert(jobz), convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklSsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklDsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(device_queue->val, itype, convert(jobz), convert(uplo), n, lda, ldb);
   return scratchpad_size;
}

extern "C" int64_t onemklSsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrd_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklDsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrd_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
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

extern "C" int64_t onemklCungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklZungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   return scratchpad_size;
}

extern "C" int64_t onemklCunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklZunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklCunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklZunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(trans), m, n, k, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklCunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
}

extern "C" int64_t onemklZunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(device_queue->val, convert(side), convert(uplo), convert(trans), m, n, lda, ldc);
   return scratchpad_size;
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

extern "C" int64_t onemklSgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<float>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklDgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<double>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklCgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<float>>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
}

extern "C" int64_t onemklZgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<double>>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   return scratchpad_size;
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
