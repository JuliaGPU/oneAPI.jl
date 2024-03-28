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

extern "C" int onemklStrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtri(device_queue->val, convert(uplo), convert(diag), n, a, lda, scratchpad, scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDtrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtri(device_queue->val, convert(uplo), convert(diag), n, a, lda, scratchpad, scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCtrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtri(device_queue->val, convert(uplo), convert(diag), n, reinterpret_cast<std::complex<float>*>(a), lda, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZtrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtri(device_queue->val, convert(uplo), convert(diag), n, reinterpret_cast<std::complex<double>*>(a), lda, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklSgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesv(device_queue->val, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesv(device_queue->val, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklCgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesv(device_queue->val, n, nrhs, reinterpret_cast<std::complex<float>*>(a), lda, ipiv, reinterpret_cast<std::complex<float>*>(b), ldb, reinterpret_cast<std::complex<float>*>(scratchpad), scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklZgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesv(device_queue->val, n, nrhs, reinterpret_cast<std::complex<double>*>(a), lda, ipiv, reinterpret_cast<std::complex<double>*>(b), ldb, reinterpret_cast<std::complex<double>*>(scratchpad), scratchpad_size, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklXsparse_matmat(syclQueue_t device_queue, matrix_handle_t A, matrix_handle_t B, matrix_handle_t C, onemklMatmatRequest req, matmat_descr_t descr, int64_t *sizeTempBuffer, void *tempBuffer) {
   auto status = oneapi::mkl::sparse::matmat(device_queue->val, (oneapi::mkl::sparse::matrix_handle_t) A, (oneapi::mkl::sparse::matrix_handle_t) B, (oneapi::mkl::sparse::matrix_handle_t) C, convert(req), (oneapi::mkl::sparse::matmat_descr_t) descr, sizeTempBuffer, tempBuffer, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

extern "C" int onemklDestroy() {
    oneapi::mkl::gpu::clean_gpu_caches();
    return 0;
}
