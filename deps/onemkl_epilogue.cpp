extern "C" int onemklXsparse_matmat(syclQueue_t device_queue, matrix_handle_t A, matrix_handle_t B, matrix_handle_t C, onemklMatmatRequest req, matmat_descr_t descr, int64_t *sizeTempBuffer, void *tempBuffer) {
   try {
      auto status = oneapi::mkl::sparse::matmat(device_queue->val, (oneapi::mkl::sparse::matrix_handle_t) A, (oneapi::mkl::sparse::matrix_handle_t) B, (oneapi::mkl::sparse::matrix_handle_t) C, convert(req), (oneapi::mkl::sparse::matmat_descr_t) descr, sizeTempBuffer, tempBuffer, {});
      device_queue->val.wait_and_throw();
   } catch (const std::exception& e) { return -1; } catch (...) { return -1; }
   return 0;
}

// other

// oneMKL keeps a cache of SYCL queues and tries to destroy them when unloading the library.
// that is incompatible with oneAPI.jl destroying queues before that, so call mkl_free_buffers
// to manually wipe the device cache when we're destroying queues.

extern "C" int onemklDestroy() {
    try {
    mkl_free_buffers();
    return 0;
    } catch (const std::exception& e) { return -1; } catch (...) { return -1; }
}
