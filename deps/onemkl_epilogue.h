int onemklXsparse_matmat(syclQueue_t device_queue, matrix_handle_t A, matrix_handle_t B,
                         matrix_handle_t C, onemklMatmatRequest req, matmat_descr_t
                         descr, int64_t *sizeTempBuffer, void *tempBuffer);

int onemklDestroy(void);
#ifdef __cplusplus
}
#endif
