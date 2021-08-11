#include <jlcxx/jlcxx.hpp>

#include <oneapi/mkl.hpp>

// https://spec.oneapi.io/versions/1.0-rev-1/elements/oneMKL/source/domains/blas/gemm.html

void oneapiHgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 half alpha, const half *A, int64_t lda, const half *B,
                 int64_t ldb, half beta, half *C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void oneapiSgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 float alpha, const float *A, int64_t lda, const float *B,
                 int64_t ldb, float beta, float *C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void oneapiDgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 double alpha, const double *A, int64_t lda, const double *B,
                 int64_t ldb, double beta, double *C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void oneapiCgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 std::complex<float> alpha, const std::complex<float> *A,
                 int64_t lda, const std::complex<float> *B, int64_t ldb,
                 std::complex<float> beta, std::complex<float> *C,
                 int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void oneapiZgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 std::complex<double> alpha, const std::complex<double> *A,
                 int64_t lda, const std::complex<double> *B, int64_t ldb,
                 std::complex<double> beta, std::complex<double> *C,
                 int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

JLCXX_MODULE define_module_mkl(jlcxx::Module &mod) {
    mod.map_type<half>("Float16");

    mod.add_bits<oneapi::mkl::transpose>("Transpose",
                                         jlcxx::julia_type("CppEnum"));
    mod.set_const("nontrans", oneapi::mkl::transpose::nontrans);
    mod.set_const("trans", oneapi::mkl::transpose::trans);
    mod.set_const("conjtrans", oneapi::mkl::transpose::conjtrans);

    mod.method("oneapiHgemm", oneapiHgemm);
    mod.method("oneapiSgemm", oneapiSgemm);
    mod.method("oneapiDgemm", oneapiDgemm);
    mod.method("oneapiCgemm", oneapiCgemm);
    mod.method("oneapiZgemm", oneapiZgemm);
}
