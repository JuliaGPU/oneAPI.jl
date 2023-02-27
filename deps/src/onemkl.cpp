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

template <typename T>
class gemmBatchInfo {
    public:
        int64_t *m_mbuf = nullptr;
        int64_t *m_nbuf = nullptr;
        int64_t *m_kbuf = nullptr;
        int64_t *m_ldabuf = nullptr;
        int64_t *m_ldbbuf = nullptr;
        int64_t *m_ldcbuf = nullptr;
        oneapi::mkl::transpose *m_transa = nullptr;
        oneapi::mkl::transpose *m_transb = nullptr;
        T *m_alphabuf = nullptr;
        T *m_betabuf = nullptr;
        int64_t *m_group_size = nullptr;
        sycl::device m_device;
        sycl::context m_context;
        oneapi::mkl::transpose m_ta;
        oneapi::mkl::transpose m_tb;

        // Constructor
        gemmBatchInfo(syclQueue_t device_queue,
                    int64_t group_count,
                    onemklTranspose transa,
                    onemklTranspose transb,
                    int64_t m, int64_t n, int64_t k,
                    int64_t lda, int64_t ldb, int64_t ldc,
                    T alpha, T beta) {
            // Get device and context info from device_queue
            auto main_queue = device_queue->val;
            m_device = main_queue.get_device();
            m_context = main_queue.get_context();
            try {
                // Allocate uniform arrays of m,n,k,lda,ldb,ldc,alpha,beta
                // group_size and transpose_a, transpose_b supporting oneMKL
                // gemm_batch API
                m_mbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_nbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_kbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_ldabuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_ldbbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_ldcbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_alphabuf = (T *) malloc_shared(group_count * sizeof(T), m_device, m_context);
                m_betabuf = (T *) malloc_shared(group_count * sizeof(T), m_device, m_context);
                m_transa = (oneapi::mkl::transpose *) malloc_shared(group_count * sizeof(oneapi::mkl::transpose),
                                                                    m_device, m_context);
                m_transb = (oneapi::mkl::transpose *) malloc_shared(group_count * sizeof(oneapi::mkl::transpose),
                                                                    m_device, m_context);
                m_ta = convert(transa);
                m_tb = convert(transb);
                m_group_size = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
            } catch(const std::bad_alloc& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
            for (int i = 0; i < group_count; i++) {
                m_mbuf[i] = m;
                m_nbuf[i] = n;
                m_kbuf[i] = k;
                m_ldabuf[i] = lda;
                m_ldbbuf[i] = ldb;
                m_ldcbuf[i] = ldc;
                m_alphabuf[i] = alpha;
                m_betabuf[i] = beta;
                m_transa[i] = m_ta;
                m_transb[i] = m_tb;
                m_group_size[i] = 1;
            }
        };

        // Destructor
        ~gemmBatchInfo() {
            free(m_mbuf, m_context);
            free(m_nbuf, m_context);
            free(m_kbuf, m_context);
            free(m_ldabuf, m_context);
            free(m_ldbbuf, m_context);
            free(m_ldcbuf, m_context);
            free(m_alphabuf, m_context);
            free(m_betabuf, m_context);
            free(m_transa, m_context);
            free(m_transb, m_context);
            free(m_group_size, m_context);
        }
};

template <typename T>
class trsmBatchInfo {
    public:
        int64_t *m_mbuf = nullptr;
        int64_t *m_nbuf = nullptr;
        int64_t *m_ldabuf = nullptr;
        int64_t *m_ldbbuf = nullptr;
        oneapi::mkl::transpose *m_transa = nullptr;
        oneapi::mkl::side *m_leftright = nullptr;
        oneapi::mkl::uplo *m_upperlower = nullptr;
        oneapi::mkl::diag *m_unitdiag = nullptr;
        T *m_alphabuf = nullptr;
        int64_t *m_group_size = nullptr;
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
                    int64_t m, int64_t n, T alpha,
                    int64_t lda, int64_t ldb, int64_t group_count) {
            // Get device and context info from device_queue
            auto main_queue = device_queue->val;
            m_device = main_queue.get_device();
            m_context = main_queue.get_context();
            try {
                // Allocate uniform arrays of m,n,k,lda,ldb,ldc,alpha,beta
                // group_size and transpose_a, transpose_b supporting oneMKL
                // gemm_batch API
                m_mbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_nbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_ldabuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_ldbbuf = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
                m_alphabuf = (T *) malloc_shared(group_count * sizeof(T), m_device, m_context);
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
                m_group_size = (int64_t *) malloc_shared(group_count * sizeof(int64_t), m_device, m_context);
            } catch(const std::bad_alloc& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
            // Initialize
            for (int i = 0; i < group_count; i++) {
                m_mbuf[i] = m;
                m_nbuf[i] = n;
                m_ldabuf[i] = lda;
                m_ldbbuf[i] = ldb;
                m_alphabuf[i] = alpha;
                m_transa[i] = m_ta;
                m_leftright[i] = m_side;
                m_upperlower[i] = m_uplo;
                m_unitdiag[i] = m_diag;
                m_group_size[i] = 1;
            }
        };

        // Destructor
        ~trsmBatchInfo() {
            free(m_mbuf, m_context);
            free(m_nbuf, m_context);
            free(m_ldabuf, m_context);
            free(m_ldbbuf, m_context);
            free(m_alphabuf, m_context);
            free(m_transa, m_context);
            free(m_upperlower, m_context);
            free(m_unitdiag, m_context);
            free(m_leftright, m_context);
            free(m_group_size, m_context);
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
                                  onemklTranspose transb, int64_t m,
                                  int64_t n, int64_t k, uint16_t alpha,
                                  const short **a, int64_t lda, const short **b,
                                  int64_t ldb, uint16_t beta, short **c,
                                  int64_t ldc, int64_t group_count) {
    gemmBatchInfo<sycl::half> gemmInfo(device_queue, group_count, transa, transb,
                          m, n, k, lda, ldb, ldc, sycl::bit_cast<sycl::half>(alpha),
                          sycl::bit_cast<sycl::half>(beta));
    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        &gemmInfo.m_mbuf[0], &gemmInfo.m_nbuf[0],
                        &gemmInfo.m_kbuf[0], &gemmInfo.m_alphabuf[0],
                        reinterpret_cast<const sycl::half **>(&a[0]), &gemmInfo.m_ldabuf[0],
                        reinterpret_cast<const sycl::half **>(&b[0]), &gemmInfo.m_ldbbuf[0],
                        &gemmInfo.m_betabuf[0], reinterpret_cast<sycl::half **>(&c[0]),
                        &gemmInfo.m_ldcbuf[0], group_count, &gemmInfo.m_group_size[0]);

    __FORCE_MKL_FLUSH__(status);

}

extern "C" void onemklSgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t m,
                                  int64_t n, int64_t k, float alpha,
                                  const float **a, int64_t lda, const float **b,
                                  int64_t ldb, float beta, float **c,
                                  int64_t ldc, int64_t group_count) {
    gemmBatchInfo<float> gemmInfo(device_queue, group_count, transa, transb,
                          m, n, k, lda, ldb, ldc, alpha, beta);

    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        &gemmInfo.m_mbuf[0], &gemmInfo.m_nbuf[0],
                        &gemmInfo.m_kbuf[0], &gemmInfo.m_alphabuf[0],
                        (const float **)&a[0], &gemmInfo.m_ldabuf[0],
                        (const float **)&b[0], &gemmInfo.m_ldbbuf[0],
                        &gemmInfo.m_betabuf[0], &c[0], &gemmInfo.m_ldcbuf[0],
                        group_count, &gemmInfo.m_group_size[0]);

    __FORCE_MKL_FLUSH__(status);

}

extern "C" void onemklDgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t m,
                                  int64_t n, int64_t k, double alpha,
                                  const double **a, int64_t lda, const double **b,
                                  int64_t ldb, double beta, double **c,
                                  int64_t ldc, int64_t group_count) {
    gemmBatchInfo<double> gemmInfo(device_queue, group_count, transa, transb,
                          m, n, k, lda, ldb, ldc, alpha, beta);

    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        &gemmInfo.m_mbuf[0], &gemmInfo.m_nbuf[0],
                        &gemmInfo.m_kbuf[0], &gemmInfo.m_alphabuf[0],
                        (const double **)&a[0], &gemmInfo.m_ldabuf[0],
                        (const double **)&b[0], &gemmInfo.m_ldbbuf[0],
                        &gemmInfo.m_betabuf[0], &c[0], &gemmInfo.m_ldcbuf[0],
                        group_count, &gemmInfo.m_group_size[0]);

    __FORCE_MKL_FLUSH__(status);

}

extern "C" void onemklCgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t m,
                                  int64_t n, int64_t k, float _Complex alpha,
                                  const float _Complex **a, int64_t lda,
                                  const float _Complex **b,
                                  int64_t ldb, float _Complex beta, float _Complex **c,
                                  int64_t ldc, int64_t group_count) {
    gemmBatchInfo<std::complex<float>> gemmInfo(device_queue, group_count, transa, transb,
                          m, n, k, lda, ldb, ldc, alpha, beta);

    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        &gemmInfo.m_mbuf[0], &gemmInfo.m_nbuf[0],
                        &gemmInfo.m_kbuf[0], &gemmInfo.m_alphabuf[0],
                        reinterpret_cast<const std::complex<float> **>(&a[0]),
                        &gemmInfo.m_ldabuf[0],
                        reinterpret_cast<const std::complex<float> **>(&b[0]),
                        &gemmInfo.m_ldbbuf[0],
                        &gemmInfo.m_betabuf[0],
                        reinterpret_cast<std::complex<float> **>(&c[0]), &gemmInfo.m_ldcbuf[0],
                        group_count, &gemmInfo.m_group_size[0]);

    __FORCE_MKL_FLUSH__(status);

}

extern "C" void onemklZgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                                  onemklTranspose transb, int64_t m,
                                  int64_t n, int64_t k, double _Complex alpha,
                                  const double _Complex **a, int64_t lda,
                                  const double _Complex **b,
                                  int64_t ldb, double _Complex beta,
                                  double _Complex **c,
                                  int64_t ldc, int64_t group_count) {
    gemmBatchInfo<std::complex<double>> gemmInfo(device_queue, group_count, transa, transb,
                          m, n, k, lda, ldb, ldc, alpha, beta);

    auto status = oneapi::mkl::blas::column_major::gemm_batch(device_queue->val,
                        &gemmInfo.m_transa[0], &gemmInfo.m_transb[0],
                        &gemmInfo.m_mbuf[0], &gemmInfo.m_nbuf[0],
                        &gemmInfo.m_kbuf[0], &gemmInfo.m_alphabuf[0],
                        reinterpret_cast<const std::complex<double> **>(&a[0]),
                        &gemmInfo.m_ldabuf[0],
                        reinterpret_cast<const std::complex<double> **>(&b[0]),
                        &gemmInfo.m_ldbbuf[0],
                        &gemmInfo.m_betabuf[0],
                        reinterpret_cast<std::complex<double> **>(&c[0]), &gemmInfo.m_ldcbuf[0],
                        group_count, &gemmInfo.m_group_size[0]);

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
                            onemklDiag unit_diag, int64_t m, int64_t n,
                            float alpha, const float **a, int64_t lda,
                            float **b, int64_t ldb, int64_t group_count) {
    trsmBatchInfo<float> trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, m, n, alpha, lda, ldb, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        &trsmInfo.m_mbuf[0], &trsmInfo.m_nbuf[0],
                        &trsmInfo.m_alphabuf[0], (const float **)&a[0],
                        &trsmInfo.m_ldabuf[0], &b[0],
                        &trsmInfo.m_ldbbuf[0], group_count,
                        &trsmInfo.m_group_size[0]);
        __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t m, int64_t n,
                            double alpha, const double **a, int64_t lda,
                            double **b, int64_t ldb, int64_t group_count) {
    trsmBatchInfo<double> trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, m, n, alpha, lda, ldb, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        &trsmInfo.m_mbuf[0], &trsmInfo.m_nbuf[0],
                        &trsmInfo.m_alphabuf[0], (const double **)&a[0],
                        &trsmInfo.m_ldabuf[0], &b[0],
                        &trsmInfo.m_ldbbuf[0], group_count,
                        &trsmInfo.m_group_size[0]);
        __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t m, int64_t n,
                            float _Complex alpha, const float _Complex **a,
                            int64_t lda, float _Complex **b, int64_t ldb,
                            int64_t group_count) {
    trsmBatchInfo<std::complex<float>> trsmInfo(device_queue, left_right, upper_lower, transa,
                                unit_diag, m, n, alpha, lda, ldb, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        &trsmInfo.m_mbuf[0], &trsmInfo.m_nbuf[0],
                        &trsmInfo.m_alphabuf[0],
                        reinterpret_cast<const std::complex<float> **>(&a[0]),
                        &trsmInfo.m_ldabuf[0],
                        reinterpret_cast<std::complex<float> **>(&b[0]),
                        &trsmInfo.m_ldbbuf[0], group_count,
                        &trsmInfo.m_group_size[0]);
        __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, onemklTranspose transa,
                            onemklDiag unit_diag, int64_t m, int64_t n,
                            double _Complex alpha, const double _Complex **a,
                            int64_t lda, double _Complex **b, int64_t ldb,
                            int64_t group_count) {
    trsmBatchInfo<std::complex<double>> trsmInfo(device_queue, left_right,
                                upper_lower, transa, unit_diag, m, n, alpha,
                                lda, ldb, group_count);

    auto status = oneapi::mkl::blas::column_major::trsm_batch(device_queue->val,
                        &trsmInfo.m_leftright[0], &trsmInfo.m_upperlower[0],
                        &trsmInfo.m_transa[0], &trsmInfo.m_unitdiag[0],
                        &trsmInfo.m_mbuf[0], &trsmInfo.m_nbuf[0],
                        &trsmInfo.m_alphabuf[0],
                        reinterpret_cast<const std::complex<double> **>(&a[0]),
                        &trsmInfo.m_ldabuf[0],
                        reinterpret_cast<std::complex<double> **>(&b[0]),
                        &trsmInfo.m_ldbbuf[0], group_count,
                        &trsmInfo.m_group_size[0]);
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
                            const short *x, std::int64_t incx, short *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n,
                                        sycl::bit_cast<sycl::half>(alpha),
                                        reinterpret_cast<const sycl::half *>(x),
                                        incx, reinterpret_cast<sycl::half *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSaxpy(syclQueue_t device_queue, int64_t n, float alpha,
                            const float *x, std::int64_t incx, float *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, alpha, x,
                                                incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDaxpy(syclQueue_t device_queue, int64_t n, double alpha,
                            const double *x, std::int64_t incx, double *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, alpha, x,
                                                incx, y, incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCaxpy(syclQueue_t device_queue, int64_t n, float _Complex alpha,
                            const float _Complex *x, std::int64_t incx, float _Complex *y, int64_t incy) {
    auto status = oneapi::mkl::blas::column_major::axpy(device_queue->val, n, static_cast<std::complex<float> >(alpha),
                            reinterpret_cast<const std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
    __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZaxpy(syclQueue_t device_queue, int64_t n, double _Complex alpha,
                            const double _Complex *x, std::int64_t incx, double _Complex *y, int64_t incy) {
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
