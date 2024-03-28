int onemklStrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n, float *a,
                 int64_t lda, float *scratchpad, int64_t scratchpad_size);

int onemklDtrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n, double *a,
                 int64_t lda, double *scratchpad, int64_t scratchpad_size);

int onemklCtrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n,
                 float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZtrtri(syclQueue_t device_queue, onemklUplo uplo, onemklDiag diag, int64_t n,
                 double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size);

int onemklSgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t
                *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size);

int onemklDgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t
                *ipiv, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size);

int onemklCgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda,
                int64_t *ipiv, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t
                scratchpad_size);

int onemklZgesv(syclQueue_t device_queue, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda,
                int64_t *ipiv, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t
                scratchpad_size);

int onemklXsparse_matmat(syclQueue_t device_queue, matrix_handle_t A, matrix_handle_t B,
                         matrix_handle_t C, onemklMatmatRequest req, matmat_descr_t
                         descr, int64_t *sizeTempBuffer, void *tempBuffer);

int onemklDestroy(void);
#ifdef __cplusplus
}
#endif
