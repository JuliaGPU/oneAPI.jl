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

// I need help :(
typedef struct MatrixHandle_st *MatrixHandle_t;

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
                            uint16_t alpha, const short *a, int64_t lda, int64_t stridea,
                            const short *b, int64_t ldb, int64_t strideb, uint16_t beta,
                            short *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklSgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            float alpha, const float *a, int64_t lda, int64_t stridea,
                            const float *b, int64_t ldb, int64_t strideb, float beta,
                            float *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklDgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            double alpha, const double *a, int64_t lda, int64_t stridea,
                            const double *b, int64_t ldb, int64_t strideb, double beta,
                            double *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklCgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            int64_t stridea, const float _Complex *b, int64_t ldb,
                            int64_t strideb, float _Complex beta, float _Complex *c,
                            int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklZgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                            onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            int64_t stridea, const double _Complex *b, int64_t ldb,
                            int64_t strideb, double _Complex beta, double _Complex *c,
                            int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n,
                int64_t k, uint16_t alpha, const short *A, int64_t lda,
                const short *B, int64_t ldb, uint16_t beta, short *C,
                int64_t ldc);

int onemklHaxpy(syclQueue_t device_queue, int64_t n, uint16_t alpha, const short *x,
                int64_t incx, short *y, int64_t incy);

int onemklHscal(syclQueue_t device_queue, int64_t n, uint16_t alpha,
                short *x, int64_t incx);

int onemklHnrm2(syclQueue_t device_queue, int64_t n, const short *x,
                int64_t incx, short *result);

int onemklHdot(syclQueue_t device_queue, int64_t n, const short *x,
               int64_t incx, const short *y, int64_t incy, short *result);
// BLAS
int onemklSgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t
                m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float
                beta, float *c, int64_t ldc);

int onemklDgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t
                m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb,
                double beta, double *c, int64_t ldc);

int onemklCgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t
                m, int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float
                _Complex *b, int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc);

int onemklZgemm(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb, int64_t
                m, int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double
                _Complex *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc);

int onemklSsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t
                m, int64_t n, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta,
                float *c, int64_t ldc);

int onemklDsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t
                m, int64_t n, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta,
                double *c, int64_t ldc);

int onemklCsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t
                m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b,
                int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc);

int onemklZsymm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t
                m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex
                *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc);

int onemklChemm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t
                m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b,
                int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc);

int onemklZhemm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower, int64_t
                m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex
                *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc);

int onemklSsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                n, int64_t k, float alpha, float *a, int64_t lda, float beta, float *c, int64_t ldc);

int onemklDsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                n, int64_t k, double alpha, double *a, int64_t lda, double beta, double *c, int64_t ldc);

int onemklCsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex
                beta, float _Complex *c, int64_t ldc);

int onemklZsyrk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex
                beta, double _Complex *c, int64_t ldc);

int onemklCherk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                n, int64_t k, float alpha, float _Complex *a, int64_t lda, float beta, float _Complex
                *c, int64_t ldc);

int onemklZherk(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                n, int64_t k, double alpha, double _Complex *a, int64_t lda, double beta, double
                _Complex *c, int64_t ldc);

int onemklSsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                 n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta,
                 float *c, int64_t ldc);

int onemklDsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                 n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double
                 beta, double *c, int64_t ldc);

int onemklCsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                 n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b,
                 int64_t ldb, float _Complex beta, float _Complex *c, int64_t ldc);

int onemklZsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                 n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex
                 *b, int64_t ldb, double _Complex beta, double _Complex *c, int64_t ldc);

int onemklCher2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                 n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b,
                 int64_t ldb, float beta, float _Complex *c, int64_t ldc);

int onemklZher2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans, int64_t
                 n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex
                 *b, int64_t ldb, double beta, double _Complex *c, int64_t ldc);

int onemklStrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float alpha,
                float *a, int64_t lda, float *b, int64_t ldb);

int onemklDtrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double alpha,
                double *a, int64_t lda, double *b, int64_t ldb);

int onemklCtrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float _Complex
                alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb);

int onemklZtrmm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double _Complex
                alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb);

int onemklStrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float alpha,
                float *a, int64_t lda, float *b, int64_t ldb);

int onemklDtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double alpha,
                double *a, int64_t lda, double *b, int64_t ldb);

int onemklCtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float _Complex
                alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb);

int onemklZtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double _Complex
                alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb);

int onemklSdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float *a,
                int64_t lda, float *x, int64_t incx, float *c, int64_t ldc);

int onemklDdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double *a,
                int64_t lda, double *x, int64_t incx, double *c, int64_t ldc);

int onemklCdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float
                _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex *c, int64_t
                ldc);

int onemklZdgmm(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double
                _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex *c,
                int64_t ldc);

int onemklSgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float alpha,
                float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy);

int onemklDgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double alpha,
                double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy);

int onemklCgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float
                _Complex beta, float _Complex *y, int64_t incy);

int onemklZgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx,
                double _Complex beta, double _Complex *y, int64_t incy);

int onemklSgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl,
                int64_t ku, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta,
                float *y, int64_t incy);

int onemklDgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl,
                int64_t ku, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta,
                double *y, int64_t incy);

int onemklCgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl,
                int64_t ku, float _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx, float _Complex beta, float _Complex *y, int64_t incy);

int onemklZgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, int64_t kl,
                int64_t ku, double _Complex alpha, double _Complex *a, int64_t lda, double _Complex
                *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy);

int onemklSger(syclQueue_t device_queue, int64_t m, int64_t n, float alpha, float *x, int64_t incx,
               float *y, int64_t incy, float *a, int64_t lda);

int onemklDger(syclQueue_t device_queue, int64_t m, int64_t n, double alpha, double *x, int64_t incx,
               double *y, int64_t incy, double *a, int64_t lda);

int onemklCgerc(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex alpha, float _Complex
                *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a, int64_t lda);

int onemklZgerc(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex alpha, double
                _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a,
                int64_t lda);

int onemklCgeru(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex alpha, float _Complex
                *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a, int64_t lda);

int onemklZgeru(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex alpha, double
                _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex *a,
                int64_t lda);

int onemklChbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, float
                _Complex alpha, float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float
                _Complex beta, float _Complex *y, int64_t incy);

int onemklZhbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, double
                _Complex alpha, double _Complex *a, int64_t lda, double _Complex *x, int64_t incx,
                double _Complex beta, double _Complex *y, int64_t incy);

int onemklChemv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
                float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta,
                float _Complex *y, int64_t incy);

int onemklZhemv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
                double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex
                beta, double _Complex *y, int64_t incy);

int onemklCher(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float
               _Complex *x, int64_t incx, float _Complex *a, int64_t lda);

int onemklZher(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double
               _Complex *x, int64_t incx, double _Complex *a, int64_t lda);

int onemklCher2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
                float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a,
                int64_t lda);

int onemklZher2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
                double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex
                *a, int64_t lda);

int onemklChpmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
                float _Complex *a, float _Complex *x, int64_t incx, float _Complex beta, float
                _Complex *y, int64_t incy);

int onemklZhpmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
                double _Complex *a, double _Complex *x, int64_t incx, double _Complex beta, double
                _Complex *y, int64_t incy);

int onemklChpr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float
               _Complex *x, int64_t incx, float _Complex *a);

int onemklZhpr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double
               _Complex *x, int64_t incx, double _Complex *a);

int onemklChpr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
                float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a);

int onemklZhpr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
                double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex
                *a);

int onemklSsbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, float alpha,
                float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy);

int onemklDsbmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, int64_t k, double
                alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t
                incy);

int onemklSsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *a,
                int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy);

int onemklDsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double
                *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy);

int onemklCsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
                float _Complex *a, int64_t lda, float _Complex *x, int64_t incx, float _Complex beta,
                float _Complex *y, int64_t incy);

int onemklZsymv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
                double _Complex *a, int64_t lda, double _Complex *x, int64_t incx, double _Complex
                beta, double _Complex *y, int64_t incy);

int onemklSsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x,
               int64_t incx, float *a, int64_t lda);

int onemklDsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *x,
               int64_t incx, double *a, int64_t lda);

int onemklCsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
               float _Complex *x, int64_t incx, float _Complex *a, int64_t lda);

int onemklZsyr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
               double _Complex *x, int64_t incx, double _Complex *a, int64_t lda);

int onemklSsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x,
                int64_t incx, float *y, int64_t incy, float *a, int64_t lda);

int onemklDsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double
                *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda);

int onemklCsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float _Complex alpha,
                float _Complex *x, int64_t incx, float _Complex *y, int64_t incy, float _Complex *a,
                int64_t lda);

int onemklZsyr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double _Complex alpha,
                double _Complex *x, int64_t incx, double _Complex *y, int64_t incy, double _Complex
                *a, int64_t lda);

int onemklSspmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *a,
                float *x, int64_t incx, float beta, float *y, int64_t incy);

int onemklDspmv(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double
                *a, double *x, int64_t incx, double beta, double *y, int64_t incy);

int onemklSspr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x,
               int64_t incx, float *a);

int onemklDspr(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double *x,
               int64_t incx, double *a);

int onemklSspr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, float alpha, float *x,
                int64_t incx, float *y, int64_t incy, float *a);

int onemklDspr2(syclQueue_t device_queue, onemklUplo upper_lower, int64_t n, double alpha, double
                *x, int64_t incx, double *y, int64_t incy, double *a);

int onemklStbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, float *a, int64_t lda, float *x, int64_t
                incx);

int onemklDtbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, double *a, int64_t lda, double *x, int64_t
                incx);

int onemklCtbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, float _Complex *a, int64_t lda, float
                _Complex *x, int64_t incx);

int onemklZtbmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, double _Complex *a, int64_t lda, double
                _Complex *x, int64_t incx);

int onemklStbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, float *a, int64_t lda, float *x, int64_t
                incx);

int onemklDtbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, double *a, int64_t lda, double *x, int64_t
                incx);

int onemklCtbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, float _Complex *a, int64_t lda, float
                _Complex *x, int64_t incx);

int onemklZtbsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, int64_t k, double _Complex *a, int64_t lda, double
                _Complex *x, int64_t incx);

int onemklStpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float *a, float *x, int64_t incx);

int onemklDtpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double *a, double *x, int64_t incx);

int onemklCtpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float _Complex *a, float _Complex *x, int64_t incx);

int onemklZtpmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double _Complex *a, double _Complex *x, int64_t
                incx);

int onemklStpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float *a, float *x, int64_t incx);

int onemklDtpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double *a, double *x, int64_t incx);

int onemklCtpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float _Complex *a, float _Complex *x, int64_t incx);

int onemklZtpsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double _Complex *a, double _Complex *x, int64_t
                incx);

int onemklStrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float *a, int64_t lda, float *x, int64_t incx);

int onemklDtrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double *a, int64_t lda, double *x, int64_t incx);

int onemklCtrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

int onemklZtrmv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

int onemklStrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float *a, int64_t lda, float *x, int64_t incx);

int onemklDtrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double *a, int64_t lda, double *x, int64_t incx);

int onemklCtrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

int onemklZtrsv(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                onemklDiag unit_diag, int64_t n, double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

int onemklCdotc(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex
                *y, int64_t incy, float _Complex *result);

int onemklZdotc(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                _Complex *y, int64_t incy, double _Complex *result);

int onemklCdotu(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex
                *y, int64_t incy, float _Complex *result);

int onemklZdotu(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                _Complex *y, int64_t incy, double _Complex *result);

int onemklSiamax(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, int64_t *result,
                 onemklIndex base);

int onemklDiamax(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, int64_t *result,
                 onemklIndex base);

int onemklCiamax(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, int64_t
                 *result, onemklIndex base);

int onemklZiamax(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, int64_t
                 *result, onemklIndex base);

int onemklSiamin(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, int64_t *result,
                 onemklIndex base);

int onemklDiamin(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, int64_t *result,
                 onemklIndex base);

int onemklCiamin(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, int64_t
                 *result, onemklIndex base);

int onemklZiamin(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, int64_t
                 *result, onemklIndex base);

int onemklSasum(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *result);

int onemklDasum(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *result);

int onemklCasum(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float *result);

int onemklZasum(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                *result);

int onemklSaxpy(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx, float *y,
                int64_t incy);

int onemklDaxpy(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx, double *y,
                int64_t incy);

int onemklCaxpy(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x, int64_t
                incx, float _Complex *y, int64_t incy);

int onemklZaxpy(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x,
                int64_t incx, double _Complex *y, int64_t incy);

int onemklSaxpby(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx, float beta,
                 float *y, int64_t incy);

int onemklDaxpby(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx, double
                 beta, double *y, int64_t incy);

int onemklCaxpby(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x,
                 int64_t incx, float _Complex beta, float _Complex *y, int64_t incy);

int onemklZaxpby(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x,
                 int64_t incx, double _Complex beta, double _Complex *y, int64_t incy);

int onemklScopy(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy);

int onemklDcopy(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy);

int onemklCcopy(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex
                *y, int64_t incy);

int onemklZcopy(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                _Complex *y, int64_t incy);

int onemklSdot(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy,
               float *result);

int onemklDdot(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy,
               double *result);

int onemklSsdsdot(syclQueue_t device_queue, int64_t n, float sb, float *x, int64_t incx, float *y,
                  int64_t incy, float *result);

int onemklSnrm2(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *result);

int onemklDnrm2(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *result);

int onemklCnrm2(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float *result);

int onemklZnrm2(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                *result);

int onemklSrot(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy,
               float c, float s);

int onemklDrot(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy,
               double c, double s);

int onemklCSrot(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex
                *y, int64_t incy, float c, float s);

int onemklCrot(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex
               *y, int64_t incy, float c, float _Complex s);

int onemklZDrot(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                _Complex *y, int64_t incy, double c, double s);

int onemklZrot(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double _Complex
               *y, int64_t incy, double c, double _Complex s);

int onemklSrotg(syclQueue_t device_queue, float *a, float *b, float *c, float *s);

int onemklDrotg(syclQueue_t device_queue, double *a, double *b, double *c, double *s);

int onemklCrotg(syclQueue_t device_queue, float _Complex *a, float _Complex *b, float *c, float
                _Complex *s);

int onemklZrotg(syclQueue_t device_queue, double _Complex *a, double _Complex *b, double *c, double
                _Complex *s);

int onemklSrotm(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy,
                float *param);

int onemklDrotm(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy,
                double *param);

int onemklSrotmg(syclQueue_t device_queue, float *d1, float *d2, float *x1, float y1, float *param);

int onemklDrotmg(syclQueue_t device_queue, double *d1, double *d2, double *x1, double y1, double
                 *param);

int onemklSscal(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx);

int onemklDscal(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx);

int onemklCSscal(syclQueue_t device_queue, int64_t n, float alpha, float _Complex *x, int64_t incx);

int onemklZDscal(syclQueue_t device_queue, int64_t n, double alpha, double _Complex *x, int64_t incx);

int onemklCscal(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x, int64_t
                incx);

int onemklZscal(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex *x,
                int64_t incx);

int onemklSswap(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy);

int onemklDswap(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, double *y, int64_t incy);

int onemklCswap(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, float _Complex
                *y, int64_t incy);

int onemklZswap(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, double
                _Complex *y, int64_t incy);

int onemklSgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                      int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, int64_t
                      stride_a, float *b, int64_t ldb, int64_t stride_b, float beta, float *c,
                      int64_t ldc, int64_t stride_c, int64_t batch_size);

int onemklDgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                      int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, int64_t
                      stride_a, double *b, int64_t ldb, int64_t stride_b, double beta, double *c,
                      int64_t ldc, int64_t stride_c, int64_t batch_size);

int onemklCgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                      int64_t m, int64_t n, int64_t k, float _Complex alpha, float _Complex *a,
                      int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t
                      stride_b, float _Complex beta, float _Complex *c, int64_t ldc, int64_t
                      stride_c, int64_t batch_size);

int onemklZgemm_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                      int64_t m, int64_t n, int64_t k, double _Complex alpha, double _Complex *a,
                      int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t
                      stride_b, double _Complex beta, double _Complex *c, int64_t ldc, int64_t
                      stride_c, int64_t batch_size);

int onemklSsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                      int64_t n, int64_t k, float alpha, float *a, int64_t lda, int64_t stride_a,
                      float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size);

int onemklDsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                      int64_t n, int64_t k, double alpha, double *a, int64_t lda, int64_t stride_a,
                      double beta, double *c, int64_t ldc, int64_t stride_c, int64_t batch_size);

int onemklCsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                      int64_t n, int64_t k, float _Complex alpha, float _Complex *a, int64_t lda,
                      int64_t stride_a, float _Complex beta, float _Complex *c, int64_t ldc, int64_t
                      stride_c, int64_t batch_size);

int onemklZsyrk_batch(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                      int64_t n, int64_t k, double _Complex alpha, double _Complex *a, int64_t lda,
                      int64_t stride_a, double _Complex beta, double _Complex *c, int64_t ldc,
                      int64_t stride_c, int64_t batch_size);

int onemklStrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                      onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float
                      alpha, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t
                      stride_b, int64_t batch_size);

int onemklDtrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                      onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double
                      alpha, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                      int64_t stride_b, int64_t batch_size);

int onemklCtrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                      onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, float
                      _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a, float
                      _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size);

int onemklZtrsm_batch(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                      onemklTranspose trans, onemklDiag unit_diag, int64_t m, int64_t n, double
                      _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a, double
                      _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size);

int onemklSgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                      alpha, float *a, int64_t lda, int64_t stridea, float *x, int64_t incx, int64_t
                      stridex, float beta, float *y, int64_t incy, int64_t stridey, int64_t
                      batch_size);

int onemklDgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                      alpha, double *a, int64_t lda, int64_t stridea, double *x, int64_t incx,
                      int64_t stridex, double beta, double *y, int64_t incy, int64_t stridey,
                      int64_t batch_size);

int onemklCgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                      _Complex alpha, float _Complex *a, int64_t lda, int64_t stridea, float
                      _Complex *x, int64_t incx, int64_t stridex, float _Complex beta, float
                      _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size);

int onemklZgemv_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                      _Complex alpha, double _Complex *a, int64_t lda, int64_t stridea, double
                      _Complex *x, int64_t incx, int64_t stridex, double _Complex beta, double
                      _Complex *y, int64_t incy, int64_t stridey, int64_t batch_size);

int onemklSdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float
                      *a, int64_t lda, int64_t stridea, float *x, int64_t incx, int64_t stridex,
                      float *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklDdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double
                      *a, int64_t lda, int64_t stridea, double *x, int64_t incx, int64_t stridex,
                      double *c, int64_t ldc, int64_t stridec, int64_t batch_size);

int onemklCdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, float
                      _Complex *a, int64_t lda, int64_t stridea, float _Complex *x, int64_t incx,
                      int64_t stridex, float _Complex *c, int64_t ldc, int64_t stridec, int64_t
                      batch_size);

int onemklZdgmm_batch(syclQueue_t device_queue, onemklSide left_right, int64_t m, int64_t n, double
                      _Complex *a, int64_t lda, int64_t stridea, double _Complex *x, int64_t incx,
                      int64_t stridex, double _Complex *c, int64_t ldc, int64_t stridec, int64_t
                      batch_size);

int onemklSaxpy_batch(syclQueue_t device_queue, int64_t n, float alpha, float *x, int64_t incx,
                      int64_t stridex, float *y, int64_t incy, int64_t stridey, int64_t batch_size);

int onemklDaxpy_batch(syclQueue_t device_queue, int64_t n, double alpha, double *x, int64_t incx,
                      int64_t stridex, double *y, int64_t incy, int64_t stridey, int64_t
                      batch_size);

int onemklCaxpy_batch(syclQueue_t device_queue, int64_t n, float _Complex alpha, float _Complex *x,
                      int64_t incx, int64_t stridex, float _Complex *y, int64_t incy, int64_t
                      stridey, int64_t batch_size);

int onemklZaxpy_batch(syclQueue_t device_queue, int64_t n, double _Complex alpha, double _Complex
                      *x, int64_t incx, int64_t stridex, double _Complex *y, int64_t incy, int64_t
                      stridey, int64_t batch_size);

int onemklScopy_batch(syclQueue_t device_queue, int64_t n, float *x, int64_t incx, int64_t stridex,
                      float *y, int64_t incy, int64_t stridey, int64_t batch_size);

int onemklDcopy_batch(syclQueue_t device_queue, int64_t n, double *x, int64_t incx, int64_t stridex,
                      double *y, int64_t incy, int64_t stridey, int64_t batch_size);

int onemklCcopy_batch(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx, int64_t
                      stridex, float _Complex *y, int64_t incy, int64_t stridey, int64_t
                      batch_size);

int onemklZcopy_batch(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx, int64_t
                      stridex, double _Complex *y, int64_t incy, int64_t stridey, int64_t
                      batch_size);

int onemklSgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa,
                 onemklTranspose transb, int64_t n, int64_t k, float alpha, float *a, int64_t lda,
                 float *b, int64_t ldb, float beta, float *c, int64_t ldc);

int onemklDgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa,
                 onemklTranspose transb, int64_t n, int64_t k, double alpha, double *a, int64_t lda,
                 double *b, int64_t ldb, double beta, double *c, int64_t ldc);

int onemklCgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa,
                 onemklTranspose transb, int64_t n, int64_t k, float _Complex alpha, float _Complex
                 *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex beta, float _Complex
                 *c, int64_t ldc);

int onemklZgemmt(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose transa,
                 onemklTranspose transb, int64_t n, int64_t k, double _Complex alpha, double
                 _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex beta,
                 double _Complex *c, int64_t ldc);

int onemklSimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                    alpha, float *ab, int64_t lda, int64_t ldb);

int onemklDimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                    alpha, double *ab, int64_t lda, int64_t ldb);

int onemklCimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                    _Complex alpha, float _Complex *ab, int64_t lda, int64_t ldb);

int onemklZimatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                    _Complex alpha, double _Complex *ab, int64_t lda, int64_t ldb);

int onemklSomatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                    alpha, float *a, int64_t lda, float *b, int64_t ldb);

int onemklDomatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                    alpha, double *a, int64_t lda, double *b, int64_t ldb);

int onemklComatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, float
                    _Complex alpha, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb);

int onemklZomatcopy(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n, double
                    _Complex alpha, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb);

int onemklSomatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                   int64_t m, int64_t n, float alpha, float *a, int64_t lda, float beta, float *b,
                   int64_t ldb, float *c, int64_t ldc);

int onemklDomatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                   int64_t m, int64_t n, double alpha, double *a, int64_t lda, double beta, double *b,
                   int64_t ldb, double *c, int64_t ldc);

int onemklComatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                   int64_t m, int64_t n, float _Complex alpha, float _Complex *a, int64_t lda, float
                   _Complex beta, float _Complex *b, int64_t ldb, float _Complex *c, int64_t ldc);

int onemklZomatadd(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose transb,
                   int64_t m, int64_t n, double _Complex alpha, double _Complex *a, int64_t lda,
                   double _Complex beta, double _Complex *b, int64_t ldb, double _Complex *c, int64_t
                   ldc);

int onemklSimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          float alpha, float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t
                          batch_size);

int onemklDimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          double alpha, double *ab, int64_t lda, int64_t ldb, int64_t stride,
                          int64_t batch_size);

int onemklCimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          float _Complex alpha, float _Complex *ab, int64_t lda, int64_t ldb,
                          int64_t stride, int64_t batch_size);

int onemklZimatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          double _Complex alpha, double _Complex *ab, int64_t lda, int64_t ldb,
                          int64_t stride, int64_t batch_size);

int onemklSomatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          float alpha, float *a, int64_t lda, int64_t stride_a, float *b, int64_t
                          ldb, int64_t stride_b, int64_t batch_size);

int onemklDomatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          double alpha, double *a, int64_t lda, int64_t stride_a, double *b, int64_t
                          ldb, int64_t stride_b, int64_t batch_size);

int onemklComatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          float _Complex alpha, float _Complex *a, int64_t lda, int64_t stride_a,
                          float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size);

int onemklZomatcopy_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                          double _Complex alpha, double _Complex *a, int64_t lda, int64_t stride_a,
                          double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size);

int onemklSomatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose
                         transb, int64_t m, int64_t n, float alpha, float *a, int64_t lda, int64_t
                         stride_a, float beta, float *b, int64_t ldb, int64_t stride_b, float *c,
                         int64_t ldc, int64_t stride_c, int64_t batch_size);

int onemklDomatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose
                         transb, int64_t m, int64_t n, double alpha, double *a, int64_t lda, int64_t
                         stride_a, double beta, double *b, int64_t ldb, int64_t stride_b, double *c,
                         int64_t ldc, int64_t stride_c, int64_t batch_size);

int onemklComatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose
                         transb, int64_t m, int64_t n, float _Complex alpha, float _Complex *a,
                         int64_t lda, int64_t stride_a, float _Complex beta, float _Complex *b,
                         int64_t ldb, int64_t stride_b, float _Complex *c, int64_t ldc, int64_t
                         stride_c, int64_t batch_size);

int onemklZomatadd_batch(syclQueue_t device_queue, onemklTranspose transa, onemklTranspose
                         transb, int64_t m, int64_t n, double _Complex alpha, double _Complex *a,
                         int64_t lda, int64_t stride_a, double _Complex beta, double _Complex *b,
                         int64_t ldb, int64_t stride_b, double _Complex *c, int64_t ldc, int64_t
                         stride_c, int64_t batch_size);

// LAPACK
int onemklSpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                 *scratchpad, int64_t scratchpad_size);

int onemklDpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double
                 *scratchpad, int64_t scratchpad_size);

int onemklCpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                 lda, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                 lda, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklDpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklCpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklZpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklSpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a,
                 int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size);

int onemklDpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a,
                 int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size);

int onemklCpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex
                 *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklZpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex
                 *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad,
                 int64_t scratchpad_size);

int64_t onemklSpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklDpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklCpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklZpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int onemklSpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                 *scratchpad, int64_t scratchpad_size);

int onemklDpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double
                 *scratchpad, int64_t scratchpad_size);

int onemklCpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                 lda, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                 lda, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklDpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklCpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklZpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklSgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int onemklCgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float
                 *d, float *e, float _Complex *tauq, float _Complex *taup, float _Complex
                 *scratchpad, int64_t scratchpad_size);

int onemklDgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *d,
                 double *e, double *tauq, double *taup, double *scratchpad, int64_t
                 scratchpad_size);

int onemklSgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *d, float
                 *e, float *tauq, float *taup, float *scratchpad, int64_t scratchpad_size);

int onemklZgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                 double *d, double *e, double _Complex *tauq, double _Complex *taup, double _Complex
                 *scratchpad, int64_t scratchpad_size);

int64_t onemklSgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int onemklCgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float
                 _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau,
                 double *scratchpad, int64_t scratchpad_size);

int onemklSgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau,
                 float *scratchpad, int64_t scratchpad_size);

int onemklZgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                 double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size);

int onemklCgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                 int64_t n, float _Complex *a, int64_t lda, float *s, float _Complex *u, int64_t ldu,
                 float _Complex *vt, int64_t ldvt, float _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklZgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                 int64_t n, double _Complex *a, int64_t lda, double *s, double _Complex *u, int64_t
                 ldu, double _Complex *vt, int64_t ldvt, double _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklDgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                 int64_t n, double *a, int64_t lda, double *s, double *u, int64_t ldu, double *vt,
                 int64_t ldvt, double *scratchpad, int64_t scratchpad_size);

int onemklSgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                 int64_t n, float *a, int64_t lda, float *s, float *u, int64_t ldu, float *vt, int64_t
                 ldvt, float *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int onemklCgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda,
                 int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t
                 *ipiv, double *scratchpad, int64_t scratchpad_size);

int onemklSgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t *ipiv,
                 float *scratchpad, int64_t scratchpad_size);

int onemklZgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                 int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_ipiv,
                                           int64_t batch_size);

int64_t onemklDgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_ipiv,
                                           int64_t batch_size);

int64_t onemklCgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_ipiv,
                                           int64_t batch_size);

int64_t onemklZgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_ipiv,
                                           int64_t batch_size);

int onemklCgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t
                       lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                       batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda,
                       int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size,
                       double *scratchpad, int64_t scratchpad_size);

int onemklSgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t
                       stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float
                       *scratchpad, int64_t scratchpad_size);

int onemklZgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t
                       lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                       batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t
                                       lda);

int64_t onemklDgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t
                                       lda);

int64_t onemklCgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t
                                       lda);

int64_t onemklZgetrfnp_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t
                                       lda);

int onemklCgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda,
                   float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double
                   *scratchpad, int64_t scratchpad_size);

int onemklSgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float
                   *scratchpad, int64_t scratchpad_size);

int onemklZgetrfnp(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                   double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                             int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklDgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                             int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklCgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                             int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklZgetrfnp_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                             int64_t lda, int64_t stride_a, int64_t batch_size);

int onemklCgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t
                         lda, int64_t stride_a, int64_t batch_size, float _Complex *scratchpad,
                         int64_t scratchpad_size);

int onemklDgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t
                         scratchpad_size);

int onemklSgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda,
                         int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t
                         scratchpad_size);

int onemklZgetrfnp_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t
                         lda, int64_t stride_a, int64_t batch_size, double _Complex *scratchpad,
                         int64_t scratchpad_size);

int64_t onemklSgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklDgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklCgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklZgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int onemklCgetri(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv,
                 float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgetri(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t *ipiv, double
                 *scratchpad, int64_t scratchpad_size);

int onemklSgetri(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t *ipiv, float
                 *scratchpad, int64_t scratchpad_size);

int onemklZgetri(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv,
                 double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklDgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklCgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklZgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                     int64_t nrhs, int64_t lda, int64_t ldb);

int onemklCgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float
                 _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *b, int64_t ldb, float
                 _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double
                 *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad, int64_t
                 scratchpad_size);

int onemklSgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a,
                 int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t
                 scratchpad_size);

int onemklZgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double
                 _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *b, int64_t ldb, double
                 _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                           int64_t n, int64_t nrhs, int64_t lda, int64_t
                                           stride_a, int64_t stride_ipiv, int64_t ldb, int64_t
                                           stride_b, int64_t batch_size);

int64_t onemklDgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                           int64_t n, int64_t nrhs, int64_t lda, int64_t
                                           stride_a, int64_t stride_ipiv, int64_t ldb, int64_t
                                           stride_b, int64_t batch_size);

int64_t onemklCgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                           int64_t n, int64_t nrhs, int64_t lda, int64_t
                                           stride_a, int64_t stride_ipiv, int64_t ldb, int64_t
                                           stride_b, int64_t batch_size);

int64_t onemklZgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                           int64_t n, int64_t nrhs, int64_t lda, int64_t
                                           stride_a, int64_t stride_ipiv, int64_t ldb, int64_t
                                           stride_b, int64_t batch_size);

int onemklCgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                       float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                       stride_ipiv, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t
                       batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                       double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                       stride_ipiv, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       double *scratchpad, int64_t scratchpad_size);

int onemklSgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                       float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv,
                       float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float
                       *scratchpad, int64_t scratchpad_size);

int onemklZgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                       double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                       stride_ipiv, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t
                       batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                             int64_t n, int64_t nrhs, int64_t lda, int64_t
                                             stride_a, int64_t ldb, int64_t stride_b, int64_t
                                             batch_size);

int64_t onemklDgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                             int64_t n, int64_t nrhs, int64_t lda, int64_t
                                             stride_a, int64_t ldb, int64_t stride_b, int64_t
                                             batch_size);

int64_t onemklCgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                             int64_t n, int64_t nrhs, int64_t lda, int64_t
                                             stride_a, int64_t ldb, int64_t stride_b, int64_t
                                             batch_size);

int64_t onemklZgetrsnp_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                             int64_t n, int64_t nrhs, int64_t lda, int64_t
                                             stride_a, int64_t ldb, int64_t stride_b, int64_t
                                             batch_size);

int onemklCgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b,
                         int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex
                         *scratchpad, int64_t scratchpad_size);

int onemklDgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t
                         stride_b, int64_t batch_size, double *scratchpad, int64_t
                         scratchpad_size);

int onemklSgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t
                         stride_b, int64_t batch_size, float *scratchpad, int64_t
                         scratchpad_size);

int onemklZgetrsnp_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b,
                         int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex
                         *scratchpad, int64_t scratchpad_size);

int64_t onemklCheev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo,
                                    int64_t n, int64_t lda);

int64_t onemklZheev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo,
                                    int64_t n, int64_t lda);

int onemklCheev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, float
                _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t
                scratchpad_size);

int onemklZheev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, double
                _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t
                scratchpad_size);

int64_t onemklCheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                     int64_t n, int64_t lda);

int64_t onemklZheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                     int64_t n, int64_t lda);

int onemklCheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float
                 _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklZheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double
                 _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t
                 scratchpad_size);

int64_t onemklChegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                     onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int64_t onemklZhegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                     onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int onemklChegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                 n, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float *w, float
                 _Complex *scratchpad, int64_t scratchpad_size);

int onemklZhegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                 n, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double *w, double
                 _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklChetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklZhetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklChetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                 lda, float *d, float *e, float _Complex *tau, float _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklZhetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                 lda, double *d, double *e, double _Complex *tau, double _Complex *scratchpad,
                 int64_t scratchpad_size);

int onemklChetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                 lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZhetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                 lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklChetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklZhetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklSorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float
                 *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size);

int onemklDorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k,
                 double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size);

int64_t onemklSorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                     int64_t n, int64_t k, int64_t lda);

int64_t onemklDorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                     int64_t n, int64_t k, int64_t lda);

int64_t onemklSorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                     int64_t lda);

int64_t onemklDorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                     int64_t lda);

int onemklDorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda,
                 double *tau, double *scratchpad, int64_t scratchpad_size);

int onemklSorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda,
                 float *tau, float *scratchpad, int64_t scratchpad_size);

int64_t onemklSormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int64_t onemklDormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int onemklDormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc,
                 double *scratchpad, int64_t scratchpad_size);

int onemklSormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float
                 *scratchpad, int64_t scratchpad_size);

int64_t onemklSsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n,
                                     int64_t ldz);

int64_t onemklDsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n,
                                     int64_t ldz);

int64_t onemklCsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n,
                                     int64_t ldz);

int64_t onemklZsteqr_scratchpad_size(syclQueue_t device_queue, onemklCompz compz, int64_t n,
                                     int64_t ldz);

int onemklCsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, float *d, float *e, float
                 _Complex *z, int64_t ldz, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, double *d, double *e, double
                 *z, int64_t ldz, double *scratchpad, int64_t scratchpad_size);

int onemklSsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, float *d, float *e, float *z,
                 int64_t ldz, float *scratchpad, int64_t scratchpad_size);

int onemklZsteqr(syclQueue_t device_queue, onemklCompz compz, int64_t n, double *d, double *e, double
                 _Complex *z, int64_t ldz, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSsyev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo,
                                    int64_t n, int64_t lda);

int64_t onemklDsyev_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo,
                                    int64_t n, int64_t lda);

int onemklDsyev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, double *a,
                int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size);

int onemklSsyev(syclQueue_t device_queue, onemklCompz jobz, onemklUplo uplo, int64_t n, float *a,
                int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size);

int64_t onemklSsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                     int64_t n, int64_t lda);

int64_t onemklDsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                     int64_t n, int64_t lda);

int onemklDsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double *a,
                 int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size);

int onemklSsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float *a,
                 int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size);

int64_t onemklSsyevx_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklRangev
                                     range, onemklUplo uplo, int64_t n, int64_t lda, float vl,
                                     float vu, int64_t il, int64_t iu, float abstol, int64_t ldz);

int64_t onemklDsyevx_scratchpad_size(syclQueue_t device_queue, onemklCompz jobz, onemklRangev
                                     range, onemklUplo uplo, int64_t n, int64_t lda, double vl,
                                     double vu, int64_t il, int64_t iu, double abstol, int64_t ldz);

int onemklDsyevx(syclQueue_t device_queue, onemklCompz jobz, onemklRangev range, onemklUplo uplo,
                 int64_t n, double *a, int64_t lda, double vl, double vu, int64_t il, int64_t iu, double
                 abstol, int64_t *m, double *w, double *z, int64_t ldz, double *scratchpad, int64_t
                 scratchpad_size);

int onemklSsyevx(syclQueue_t device_queue, onemklCompz jobz, onemklRangev range, onemklUplo uplo,
                 int64_t n, float *a, int64_t lda, float vl, float vu, int64_t il, int64_t iu, float
                 abstol, int64_t *m, float *w, float *z, int64_t ldz, float *scratchpad, int64_t
                 scratchpad_size);

int64_t onemklSsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                     onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int64_t onemklDsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                     onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int onemklDsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                 n, double *a, int64_t lda, double *b, int64_t ldb, double *w, double *scratchpad,
                 int64_t scratchpad_size);

int onemklSsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                 n, float *a, int64_t lda, float *b, int64_t ldb, float *w, float *scratchpad, int64_t
                 scratchpad_size);

int64_t onemklSsygvx_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklCompz jobz,
                                     onemklRangev range, onemklUplo uplo, int64_t n, int64_t lda,
                                     int64_t ldb, float vl, float vu, int64_t il, int64_t iu, float
                                     abstol, int64_t ldz);

int64_t onemklDsygvx_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklCompz jobz,
                                     onemklRangev range, onemklUplo uplo, int64_t n, int64_t lda,
                                     int64_t ldb, double vl, double vu, int64_t il, int64_t iu,
                                     double abstol, int64_t ldz);

int onemklDsygvx(syclQueue_t device_queue, int64_t itype, onemklCompz jobz, onemklRangev range,
                 onemklUplo uplo, int64_t n, double *a, int64_t lda, double *b, int64_t ldb, double vl,
                 double vu, int64_t il, int64_t iu, double abstol, int64_t *m, double *w, double *z,
                 int64_t ldz, double *scratchpad, int64_t scratchpad_size);

int onemklSsygvx(syclQueue_t device_queue, int64_t itype, onemklCompz jobz, onemklRangev range,
                 onemklUplo uplo, int64_t n, float *a, int64_t lda, float *b, int64_t ldb, float vl,
                 float vu, int64_t il, int64_t iu, float abstol, int64_t *m, float *w, float *z, int64_t
                 ldz, float *scratchpad, int64_t scratchpad_size);

int64_t onemklSsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklDsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklDsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double
                 *d, double *e, double *tau, double *scratchpad, int64_t scratchpad_size);

int onemklSsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                 *d, float *e, float *tau, float *scratchpad, int64_t scratchpad_size);

int64_t onemklStrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                     trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda,
                                     int64_t ldb);

int64_t onemklDtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                     trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda,
                                     int64_t ldb);

int64_t onemklCtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                     trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda,
                                     int64_t ldb);

int64_t onemklZtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                     trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda,
                                     int64_t ldb);

int onemklCtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                 diag, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b,
                 int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklDtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                 diag, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double
                 *scratchpad, int64_t scratchpad_size);

int onemklStrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                 diag, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float
                 *scratchpad, int64_t scratchpad_size);

int onemklZtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                 diag, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b,
                 int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size);

int onemklCungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float
                 _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklZungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k,
                 double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad,
                 int64_t scratchpad_size);

int64_t onemklCungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                     int64_t n, int64_t k, int64_t lda);

int64_t onemklZungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                     int64_t n, int64_t k, int64_t lda);

int64_t onemklCungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                     int64_t lda);

int64_t onemklZungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                     int64_t lda);

int onemklCungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t
                 lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a,
                 int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t
                 scratchpad_size);

int64_t onemklCunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int64_t onemklZunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int onemklCunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float
                 _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double
                 _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size);

int onemklSgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau,
                 float *scratchpad, int64_t scratchpad_size);

int onemklDgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau,
                 double *scratchpad, int64_t scratchpad_size);

int onemklCgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float
                 _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                 double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int onemklSormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float
                 *scratchpad, int64_t scratchpad_size);

int onemklDormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc,
                 double *scratchpad, int64_t scratchpad_size);

int64_t onemklSormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int64_t onemklDormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int onemklCunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float
                 _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                 int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double
                 _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklCunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int64_t onemklZunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                     trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                     ldc);

int onemklSsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t
                 *ipiv, float *scratchpad, int64_t scratchpad_size);

int onemklDsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                 int64_t *ipiv, double *scratchpad, int64_t scratchpad_size);

int onemklCsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                 lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                 lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklSsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklDsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklCsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklZsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklSorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                 *tau, float *scratchpad, int64_t scratchpad_size);

int onemklDorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double
                 *tau, double *scratchpad, int64_t scratchpad_size);

int64_t onemklSorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklDorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklCungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                 lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                 lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size);

int64_t onemklCungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int64_t onemklZungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                     int64_t lda);

int onemklSormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                 trans, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *c, int64_t ldc,
                 float *scratchpad, int64_t scratchpad_size);

int onemklDormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                 trans, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *c, int64_t
                 ldc, double *scratchpad, int64_t scratchpad_size);

int64_t onemklSormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                     onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                     int64_t ldc);

int64_t onemklDormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                     onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                     int64_t ldc);

int onemklCunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                 trans, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau,
                 float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t
                 scratchpad_size);

int onemklZunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                 trans, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau,
                 double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t
                 scratchpad_size);

int64_t onemklCunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                     onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                     int64_t ldc);

int64_t onemklZunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                     onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                     int64_t ldc);

int onemklSpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda,
                       int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t
                       scratchpad_size);

int onemklDpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                       int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t
                       scratchpad_size);

int onemklCpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a,
                       int64_t lda, int64_t stride_a, int64_t batch_size, float _Complex
                       *scratchpad, int64_t scratchpad_size);

int onemklZpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a,
                       int64_t lda, int64_t stride_a, int64_t batch_size, double _Complex
                       *scratchpad, int64_t scratchpad_size);

int onemklSpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a,
                       int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, float *scratchpad, int64_t scratchpad_size);

int onemklDpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double
                       *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, double *scratchpad, int64_t scratchpad_size);

int onemklCpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float
                       _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb,
                       int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t
                       scratchpad_size);

int onemklZpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double
                       _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb,
                       int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t
                       scratchpad_size);

int onemklSgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t
                       stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float
                       *scratchpad, int64_t scratchpad_size);

int onemklDgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda,
                       int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size,
                       double *scratchpad, int64_t scratchpad_size);

int onemklCgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t
                       lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t
                       batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t
                       lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t
                       batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

int onemklSorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t
                       lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size,
                       float *scratchpad, int64_t scratchpad_size);

int onemklDorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t
                       lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size,
                       double *scratchpad, int64_t scratchpad_size);

int onemklCungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a,
                       int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau,
                       int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

int onemklZungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a,
                       int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau,
                       int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

int onemklSgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                      int64_t nrhs, float *a, int64_t lda, int64_t stridea, float *b, int64_t ldb,
                      int64_t strideb, int64_t batchsize, float *scratchpad, int64_t
                      scratchpad_size);

int onemklDgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                      int64_t nrhs, double *a, int64_t lda, int64_t stridea, double *b, int64_t ldb,
                      int64_t strideb, int64_t batchsize, double *scratchpad, int64_t
                      scratchpad_size);

int onemklCgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                      int64_t nrhs, float _Complex *a, int64_t lda, int64_t stridea, float _Complex
                      *b, int64_t ldb, int64_t strideb, int64_t batchsize, float _Complex
                      *scratchpad, int64_t scratchpad_size);

int onemklZgels_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t m, int64_t n,
                      int64_t nrhs, double _Complex *a, int64_t lda, int64_t stridea, double
                      _Complex *b, int64_t ldb, int64_t strideb, int64_t batchsize, double _Complex
                      *scratchpad, int64_t scratchpad_size);

int64_t onemklSpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklDpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklCpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklZpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t batch_size);

int64_t onemklSpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t nrhs, int64_t lda, int64_t stride_a, int64_t
                                           ldb, int64_t stride_b, int64_t batch_size);

int64_t onemklDpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t nrhs, int64_t lda, int64_t stride_a, int64_t
                                           ldb, int64_t stride_b, int64_t batch_size);

int64_t onemklCpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t nrhs, int64_t lda, int64_t stride_a, int64_t
                                           ldb, int64_t stride_b, int64_t batch_size);

int64_t onemklZpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                           int64_t nrhs, int64_t lda, int64_t stride_a, int64_t
                                           ldb, int64_t stride_b, int64_t batch_size);

int64_t onemklSgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_tau,
                                           int64_t batch_size);

int64_t onemklDgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_tau,
                                           int64_t batch_size);

int64_t onemklCgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_tau,
                                           int64_t batch_size);

int64_t onemklZgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t lda, int64_t stride_a, int64_t stride_tau,
                                           int64_t batch_size);

int64_t onemklSorgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t k, int64_t lda, int64_t stride_a, int64_t
                                           stride_tau, int64_t batch_size);

int64_t onemklDorgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t k, int64_t lda, int64_t stride_a, int64_t
                                           stride_tau, int64_t batch_size);

int64_t onemklCungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t k, int64_t lda, int64_t stride_a, int64_t
                                           stride_tau, int64_t batch_size);

int64_t onemklZungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n,
                                           int64_t k, int64_t lda, int64_t stride_a, int64_t
                                           stride_tau, int64_t batch_size);

int64_t onemklSgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                          int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t
                                          stride_a, int64_t ldb, int64_t stride_b, int64_t
                                          batch_size);

int64_t onemklDgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                          int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t
                                          stride_a, int64_t ldb, int64_t stride_b, int64_t
                                          batch_size);

int64_t onemklCgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                          int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t
                                          stride_a, int64_t ldb, int64_t stride_b, int64_t
                                          batch_size);

int64_t onemklZgels_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans,
                                          int64_t m, int64_t n, int64_t nrhs, int64_t lda, int64_t
                                          stride_a, int64_t ldb, int64_t stride_b, int64_t
                                          batch_size);

void onemklDestroy(void);
#ifdef __cplusplus
}
#endif
