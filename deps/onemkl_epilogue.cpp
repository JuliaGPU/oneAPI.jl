extern "C" int onemklXsparse_matmat(syclQueue_t device_queue, matrix_handle_t A, matrix_handle_t B, matrix_handle_t C, onemklMatmatRequest req, matmat_descr_t descr, int64_t *sizeTempBuffer, void *tempBuffer) {
   auto status = oneapi::mkl::sparse::matmat(device_queue->val, (oneapi::mkl::sparse::matrix_handle_t) A, (oneapi::mkl::sparse::matrix_handle_t) B, (oneapi::mkl::sparse::matrix_handle_t) C, convert(req), (oneapi::mkl::sparse::matmat_descr_t) descr, sizeTempBuffer, tempBuffer, {});
   __FORCE_MKL_FLUSH__(status);
   return 0;
}

// other

// oneMKL keeps a cache of SYCL queues and tries to destroy them when unloading the library.
// that is incompatible with oneAPI.jl destroying queues before that, so call mkl_free_buffers
// and mkl_sycl_destructor to manually cleanup oneMKL cache when we're destroying queues.

extern "C" int onemklDestroy() {
    mkl_free_buffers();
    mkl_sycl_destructor();
    return 0;
}
