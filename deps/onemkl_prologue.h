#pragma once

#include "sycl.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// BLAS types
typedef enum {
    ONEMKL_TRANSPOSE_NONTRANS,
    ONEMKL_TRANSPOSE_TRANS,
    ONEMLK_TRANSPOSE_CONJTRANS
} onemklTranspose;

typedef enum {
    ONEMKL_UPLO_UPPER,
    ONEMKL_UPLO_LOWER
} onemklUplo;

typedef enum {
    ONEMKL_DIAG_NONUNIT,
    ONEMKL_DIAG_UNIT
 } onemklDiag;

typedef enum {
    ONEMKL_SIDE_LEFT,
    ONEMKL_SIDE_RIGHT
} onemklSide;

typedef enum {
    ONEMKL_OFFSET_ROW,
    ONEMKL_OFFSET_COL,
    ONEMKL_OFFSET_FIX,
} onemklOffset;

// LAPACK types
typedef enum {
    ONEMKL_JOB_N,
    ONEMKL_JOB_V,
    ONEMKL_JOB_U,
    ONEMKL_JOB_A,
    ONEMKL_JOB_S,
    ONEMKL_JOB_O
} onemklJob;

typedef enum {
    ONEMKL_GENERATE_Q,
    ONEMKL_GENERATE_P,
    ONEMKL_GENERATE_N,
    ONEMKL_GENERATE_V
} onemklGenerate;

typedef enum {
    ONEMKL_COMPZ_N,
    ONEMKL_COMPZ_V,
    ONEMKL_COMPZ_I
} onemklCompz;

typedef enum {
    ONEMKL_DIRECT_F,
    ONEMKL_DIRECT_B
} onemklDirect;

typedef enum {
    ONEMKL_STOREV_C,
    ONEMKL_STOREV_R
} onemklStorev;

typedef enum {
    ONEMKL_RANGEV_A,
    ONEMKL_RANGEV_V,
    ONEMKL_RANGEV_I
} onemklRangev;

typedef enum {
    ONEMKL_ORDER_B,
    ONEMKL_ORDER_E
} onemklOrder;

typedef enum {
    ONEMKL_JOBSVD_N,
    ONEMKL_JOBSVD_A,
    ONEMKL_JOBSVD_O,
    ONEMKL_JOBSVD_S
} onemklJobsvd;

typedef enum {
    ONEMKL_LAYOUT_ROW,
    ONEMKL_LAYOUT_COL,
} onemklLayout;

typedef enum {
    ONEMKL_INDEX_ZERO,
    ONEMKL_INDEX_ONE,
} onemklIndex;

// SPARSE types
typedef enum {
    ONEMKL_PROPERTY_SYMMETRIC,
    ONEMKL_PROPERTY_SORTED,
} onemklProperty;

typedef enum {
    ONEMKL_MATRIX_VIEW_GENERAL,
} onemklMatrixView;

typedef enum {
    ONEMKL_MATMAT_REQUEST_GET_WORK_ESTIMATION_BUF_SIZE,
    ONEMKL_MATMAT_REQUEST_WORK_ESTIMATION,
    ONEMKL_MATMAT_REQUEST_GET_COMPUTE_STRUCTURE_BUF_SIZE,
    ONEMKL_MATMAT_REQUEST_COMPUTE_STRUCTURE,
    ONEMKL_MATMAT_REQUEST_FINALIZE_STRUCTURE,
    ONEMKL_MATMAT_REQUEST_GET_COMPUTE_BUF_SIZE,
    ONEMKL_MATMAT_REQUEST_COMPUTE,
    ONEMKL_MATMAT_REQUEST_GET_NNZ,
    ONEMKL_MATMAT_REQUEST_FINALIZE,
} onemklMatmatRequest;

struct matrix_handle;
typedef struct matrix_handle *matrix_handle_t;

struct matmat_descr;
typedef struct matmat_descr *matmat_descr_t;

int onemklHgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                       onemklTranspose transb, int64_t *m,
                       int64_t *n, int64_t *k, uint16_t *alpha,
                       const short **a, int64_t *lda, const short **b,
                       int64_t *ldb, uint16_t *beta, short **c,
                       int64_t *ldc, int64_t group_count, int64_t *group_size);

int onemklSgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                       onemklTranspose transb, int64_t *m,
                       int64_t *n, int64_t *k, float *alpha,
                       const float **a, int64_t *lda, const float **b,
                       int64_t *ldb, float *beta, float **c,
                       int64_t *ldc, int64_t group_count, int64_t *group_size);

int onemklDgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                       onemklTranspose transb, int64_t *m,
                       int64_t *n, int64_t *k, double *alpha,
                       const double **a, int64_t *lda, const double **b,
                       int64_t *ldb, double *beta, double **c,
                       int64_t *ldc, int64_t group_count, int64_t *group_size);

int onemklCgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                       onemklTranspose transb, int64_t *m,
                       int64_t *n, int64_t *k, float _Complex *alpha,
                       const float _Complex **a, int64_t *lda,
                       const float _Complex **b,
                       int64_t *ldb, float _Complex *beta,
                       float _Complex **c, int64_t *ldc,
                       int64_t group_count, int64_t *group_size);

int onemklZgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                       onemklTranspose transb, int64_t *m,
                       int64_t *n, int64_t *k, double _Complex *alpha,
                       const double _Complex **a, int64_t *lda,
                       const double _Complex **b,
                       int64_t *ldb, double _Complex *beta,
                       double _Complex **c, int64_t *ldc,
                       int64_t group_count, int64_t *group_size);

int onemklStrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                       onemklUplo upper_lower, onemklTranspose transa,
                       onemklDiag unit_diag, int64_t *m, int64_t *n,
                       float *alpha, const float **a, int64_t *lda,
                       float **b, int64_t *ldb, int64_t group_count,
                       int64_t *group_size);

int onemklDtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                       onemklUplo upper_lower, onemklTranspose transa,
                       onemklDiag unit_diag, int64_t *m, int64_t *n,
                       double *alpha, const double **a, int64_t *lda,
                       double **b, int64_t *ldb, int64_t group_count,
                       int64_t *group_size);

int onemklCtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                       onemklUplo upper_lower, onemklTranspose transa,
                       onemklDiag unit_diag, int64_t *m, int64_t *n,
                       float _Complex *alpha, const float _Complex **a, int64_t *lda,
                       float _Complex **b, int64_t *ldb, int64_t group_count,
                       int64_t *group_size);

int onemklZtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                       onemklUplo upper_lower, onemklTranspose transa,
                       onemklDiag unit_diag, int64_t *m, int64_t *n,
                       double _Complex *alpha, const double _Complex **a, int64_t *lda,
                       double _Complex **b, int64_t *ldb, int64_t group_count,
                       int64_t *group_size);

int onemklHgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            uint16_t *alpha, const short *a, int64_t lda, int64_t stridea,
                            const short *b, int64_t ldb, int64_t strideb, uint16_t *beta,
                            short *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklSgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            float *alpha, const float *a, int64_t lda, int64_t stridea,
                            const float *b, int64_t ldb, int64_t strideb, float *beta,
                            float *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklDgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            double *alpha, const double *a, int64_t lda, int64_t stridea,
                            const double *b, int64_t ldb, int64_t strideb, double *beta,
                            double *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklCgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            float _Complex *alpha, const float _Complex *a, int64_t lda,
                            int64_t stridea, const float _Complex *b, int64_t ldb,
                            int64_t strideb, float _Complex *beta, float _Complex *c,
                            int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklZgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            double _Complex *alpha, const double _Complex *a, int64_t lda,
                            int64_t stridea, const double _Complex *b, int64_t ldb,
                            int64_t strideb, double _Complex *beta, double _Complex *c,
                            int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n,
                int64_t k, uint16_t *alpha, const short *A, int64_t lda,
                const short *B, int64_t ldb, uint16_t *beta, short *C,
                int64_t ldc);

int onemklHaxpy(syclQueue_t device_queue, int64_t n, uint16_t *alpha, const short *x,
                int64_t incx, short *y, int64_t incy);

int onemklHscal(syclQueue_t device_queue, int64_t n, uint16_t *alpha,
                short *x, int64_t incx);

int onemklHnrm2(syclQueue_t device_queue, int64_t n, const short *x,
                int64_t incx, short *result);

int onemklHdot(syclQueue_t device_queue, int64_t n, const short *x,
               int64_t incx, const short *y, int64_t incy, short *result);
