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

typedef enum {
    ONEMKL_LAYOUT_ROW,
    ONEMKL_LAYOUT_COL,
} onemklLayout;

typedef enum {
    ONEMKL_INDEX_ZERO,
    ONEMKL_INDEX_ONE,
} onemklIndex;

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
    ONEMKL_JOBSVD_N,
    ONEMKL_JOBSVD_A,
    ONEMKL_JOBSVD_O,
    ONEMKL_JOBSVD_S
} onemklJobsvd;

typedef enum {
    ONEMKL_GENERATE_Q,
    ONEMKL_GENERATE_P,
    ONEMKL_GENERATE_N,
    ONEMKL_GENERATE_V
} onemklGenerate;

// XXX: how to expose half in C?
// int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
//                 onemklTranspose transB, int64_t m, int64_t n, int64_t k,
//                 half alpha, const half *A, int64_t lda, const half *B,
//                 int64_t ldb, half beta, half *C, int64_t ldc);

int onemklSgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                float alpha, const float *A, int64_t lda, const float *B,
                int64_t ldb, float beta, float *C, int64_t ldc);
int onemklDgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc);
int onemklCgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex *A, int64_t lda,
                const float _Complex *B, int64_t ldb, float _Complex beta,
                float _Complex *C, int64_t ldc);
int onemklZgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex *A, int64_t lda,
                const double _Complex *B, int64_t ldb, double _Complex beta,
                double _Complex *C, int64_t ldc);
int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n,
                int64_t k, uint16_t alpha, const short *A, int64_t lda,
                const short *B, int64_t ldb, uint16_t beta, short *C,
                int64_t ldc);

void onemklHgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t *m,
                        int64_t *n, int64_t *k, uint16_t *alpha,
                        const short **a, int64_t *lda, const short **b,
                        int64_t *ldb, uint16_t *beta, short **c,
                        int64_t *ldc, int64_t group_count, int64_t *group_size);

void onemklSgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t *m,
                        int64_t *n, int64_t *k, float *alpha,
                        const float **a, int64_t *lda, const float **b,
                        int64_t *ldb, float *beta, float **c,
                        int64_t *ldc, int64_t group_count, int64_t *group_size);

void onemklDgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t *m,
                        int64_t *n, int64_t *k, double *alpha,
                        const double **a, int64_t *lda, const double **b,
                        int64_t *ldb, double *beta, double **c,
                        int64_t *ldc, int64_t group_count, int64_t *group_size);

void onemklCgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t *m,
                        int64_t *n, int64_t *k, float _Complex *alpha,
                        const float _Complex **a, int64_t *lda,
                        const float _Complex **b,
                        int64_t *ldb, float _Complex *beta,
                        float _Complex **c, int64_t *ldc,
                        int64_t group_count, int64_t *group_size);

void onemklZgemmBatched(syclQueue_t device_queue, onemklTranspose transa,
                        onemklTranspose transb, int64_t *m,
                        int64_t *n, int64_t *k, double _Complex *alpha,
                        const double _Complex **a, int64_t *lda,
                        const double _Complex **b,
                        int64_t *ldb, double _Complex *beta,
                        double _Complex **c, int64_t *ldc,
                        int64_t group_count, int64_t *group_size);

void onemklHgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                    onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                    uint16_t alpha, const short *a, int64_t lda, int64_t stridea,
                    const short *b, int64_t ldb, int64_t strideb, uint16_t beta,
                    short *c, int64_t ldc, int64_t stridec, int64_t batch_size);
void onemklSgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                    onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                    float alpha, const float *a, int64_t lda, int64_t stridea,
                    const float *b, int64_t ldb, int64_t strideb, float beta,
                    float *c, int64_t ldc, int64_t stridec, int64_t batch_size);
void onemklDgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                    onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                    double alpha, const double *a, int64_t lda, int64_t stridea,
                    const double *b, int64_t ldb, int64_t strideb, double beta,
                    double *c, int64_t ldc, int64_t stridec, int64_t batch_size);
void onemklCgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                    onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                    float _Complex alpha, const float _Complex *a, int64_t lda,
                    int64_t stridea, const float _Complex *b, int64_t ldb,
                    int64_t strideb, float _Complex beta, float _Complex *c,
                    int64_t ldc, int64_t stridec, int64_t batch_size);
void onemklZgemmBatchStrided(syclQueue_t device_queue, onemklTranspose transa,
                    onemklTranspose transb, int64_t m, int64_t n, int64_t k,
                    double _Complex alpha, const double _Complex *a, int64_t lda,
                    int64_t stridea, const double _Complex *b, int64_t ldb,
                    int64_t strideb, double _Complex beta, double _Complex *c,
                    int64_t ldc, int64_t stridec, int64_t batch_size);

void onemklSsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                float alpha, const float *a, int64_t lda, const float *b,
                int64_t ldb, float beta, float *c, int64_t ldc);
void onemklDsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                double alpha, const double *a, int64_t lda, const double *b,
                int64_t ldb, double beta, double *c, int64_t ldc);
void onemklCsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex *a, int64_t lda,
                const float _Complex *b, int64_t ldb, float _Complex beta,
                float _Complex *c, int64_t ldc);
void onemklZsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex *a, int64_t lda,
                const double _Complex *b, int64_t ldb, double _Complex beta,
                double _Complex *c, int64_t ldc);

void onemklSsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, float alpha,
                const float *a, int64_t lda, float beta, float *c, int64_t ldc);
void onemklDsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, double alpha,
                const double *a, int64_t lda, double beta, double *c, int64_t ldc);
void onemklCsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha,
                const float _Complex *a, int64_t lda, float _Complex beta, float _Complex *c,
                int64_t ldc);
void onemklZsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha,
                const double _Complex *a, int64_t lda, double _Complex beta, double _Complex *c,
                int64_t ldc);

void onemklSsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                const float *b, int64_t ldb, float beta, float *c, int64_t ldc);
void onemklDsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                const double *b, int64_t ldb, double beta, double *c, int64_t ldc);
void onemklCsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                int64_t n, int64_t k, float _Complex alpha, const float _Complex *a,
                int64_t lda, const float _Complex *b, int64_t ldb, float _Complex beta,
                float _Complex *c, int64_t ldc);
void onemklZsyr2k(syclQueue_t device_queue, onemklUplo upper_lower, onemklTranspose trans,
                int64_t n, int64_t k, double _Complex alpha, const double _Complex *a,
                int64_t lda, const double _Complex *b, int64_t ldb, double _Complex beta,
                double _Complex *c, int64_t ldc);

void onemklStrmm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo uppler_lower, onemklTranspose trans,
                onemklDiag diag, int64_t m, int64_t n, float alpha,
                const float *a, int64_t lda, float *b, int64_t ldb);
void onemklDtrmm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo uppler_lower, onemklTranspose trans,
                onemklDiag diag, int64_t m, int64_t n, double alpha,
                const double *a, int64_t lda, double *b, int64_t ldb);
void onemklCtrmm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo uppler_lower, onemklTranspose trans,
                onemklDiag diag, int64_t m, int64_t n, float _Complex alpha,
                const float _Complex *a, int64_t lda, float _Complex *b,
                int64_t ldb);
void onemklZtrmm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo uppler_lower, onemklTranspose trans,
                onemklDiag diag, int64_t m, int64_t n, double _Complex alpha,
                const double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb);

void onemklStrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                float alpha, const float *a, int64_t lda, float *b, int64_t ldb);
void onemklDtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                double alpha, const double *a, int64_t lda, double *b, int64_t ldb);
void onemklCtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex *a, int64_t lda, float _Complex *b,
                int64_t ldb);
void onemklZtrsm(syclQueue_t device_queue, onemklSide left_right, onemklUplo upper_lower,
                onemklTranspose transa, onemklDiag unit_diag, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex *a, int64_t lda, double _Complex *b,
                int64_t ldb);

void onemklStrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, onemklTranspose transa,
                onemklDiag unit_diag, int64_t *m, int64_t *n,
                float *alpha, const float **a, int64_t *lda,
                float **b, int64_t *ldb, int64_t group_count,
                int64_t *group_size);

void onemklDtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, onemklTranspose transa,
                onemklDiag unit_diag, int64_t *m, int64_t *n,
                double *alpha, const double **a, int64_t *lda,
                double **b, int64_t *ldb, int64_t group_count,
                int64_t *group_size);

void onemklCtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, onemklTranspose transa,
                onemklDiag unit_diag, int64_t *m, int64_t *n,
                float _Complex *alpha, const float _Complex **a, int64_t *lda,
                float _Complex **b, int64_t *ldb, int64_t group_count,
                int64_t *group_size);

void onemklZtrsmBatched(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, onemklTranspose transa,
                onemklDiag unit_diag, int64_t *m, int64_t *n,
                double _Complex *alpha, const double _Complex **a, int64_t *lda,
                double _Complex **b, int64_t *ldb, int64_t group_count,
                int64_t *group_size);

void onemklChemm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            float _Complex alpha, const float _Complex *a,
                            int64_t lda, const float _Complex *b, int64_t ldb,
                            float _Complex beta, float _Complex *c, int64_t ldc);
void onemklZhemm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            double _Complex alpha, const double _Complex *a,
                            int64_t lda, const double _Complex *b, int64_t ldb,
                            double _Complex beta, double _Complex *c, int64_t ldc);

void onemklCherk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, float alpha,
                const float _Complex *a, int64_t lda, float beta,
                float _Complex *c, int64_t ldc);
void onemklZherk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, double alpha,
                const double _Complex *a, int64_t lda, double beta,
                double _Complex *c, int64_t ldc);

void onemklCher2k(syclQueue_t device_queue, onemklUplo upper_lower,
                             onemklTranspose trans, int64_t n, int64_t k,
                             float _Complex alpha, const float _Complex *a,
                             int64_t lda, const float _Complex *b, int64_t ldb,
                             float beta, float _Complex *c, int64_t ldc);
void onemklZher2k(syclQueue_t device_queue, onemklUplo upper_lower,
                             onemklTranspose trans, int64_t n, int64_t k,
                             double _Complex alpha, const double _Complex *a,
                             int64_t lda, const double _Complex *b, int64_t ldb,
                             double beta, double _Complex *c, int64_t ldc);

void onemklSgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, float alpha, const float *a,
                int64_t lda, const float *x, int64_t incx, float beta, float *y,
                int64_t incy);
void onemklDgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, double alpha, const double *a,
                int64_t lda, const double *x, int64_t incx, double beta, double *y,
                int64_t incy);
void onemklCgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, float _Complex alpha, const float
                _Complex *a, int64_t lda, const float _Complex *x, int64_t incx,
                float _Complex beta, float _Complex *y, int64_t incy);
void onemklZgbmv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                int64_t n, int64_t kl, int64_t ku, double _Complex alpha,
                const double _Complex *a, int64_t lda, const double _Complex *x,
                int64_t incx, double _Complex beta, double _Complex *y, int64_t incy);

void onemklSgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, float alpha, const float *a, int64_t lda,
                 const float *x, int64_t incx, float beta, float *y, int64_t incy);
void onemklDgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, double alpha, const double *a, int64_t lda,
                 const double *x, int64_t incx, double beta, double *y, int64_t incy);
void onemklCgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, float _Complex alpha, const float _Complex *a, int64_t lda,
                 const float _Complex *x, int64_t incx, float _Complex beta,
                 float _Complex *y, int64_t incy);
void onemklZgemv(syclQueue_t device_queue, onemklTranspose trans, int64_t m,
                 int64_t n, double _Complex alpha, const double _Complex *a, int64_t lda,
                 const double _Complex *x, int64_t incx, double _Complex beta,
                 double _Complex *y, int64_t incy);

void onemklSger(syclQueue_t device_queue, int64_t m, int64_t n, float alpha,
                const float *x, int64_t incx, const float *y, int64_t incy,
                float *a, int64_t lda);
void onemklDger(syclQueue_t device_queue, int64_t m, int64_t n, double alpha,
                const double *x, int64_t incx, const double *y, int64_t incy,
                double *a, int64_t lda);
void onemklCgerc(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *a, int64_t lda);
void onemklZgerc(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *a, int64_t lda);

void onemklSasum(syclQueue_t device_queue, int64_t n,
                const float *x, int64_t incx, float *result);
void onemklDasum(syclQueue_t device_queue, int64_t n,
                const double *x, int64_t incx, double *result);
void onemklCasum(syclQueue_t device_queue, int64_t n,
                const float _Complex *x, int64_t incx, float *result);
void onemklZasum(syclQueue_t device_queue, int64_t n,
                const double _Complex *x, int64_t incx, double *result);

void onemklSaxpy(syclQueue_t device_queue, int64_t n, float alpha, const float *x,
                int64_t incx, float *y, int64_t incy);
void onemklDaxpy(syclQueue_t device_queue, int64_t n, double alpha, const double *x,
                int64_t incx, double *y, int64_t incy);
void onemklCaxpy(syclQueue_t device_queue, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, float _Complex *y, int64_t incy);
void onemklZaxpy(syclQueue_t device_queue, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, double _Complex *y, int64_t incy);
void onemklHaxpy(syclQueue_t device_queue, int64_t n, uint16_t alpha, const short *x,
                int64_t incx, short *y, int64_t incy);

void onemklSaxpby(syclQueue_t device_queue, int64_t n, float alpha, const float *x,
                  int64_t incx, float beta, float *y, int64_t incy);
void onemklDaxpby(syclQueue_t device_queue, int64_t n, double alpha, const double *x,
                  int64_t incx, double beta, double *y, int64_t incy);
void onemklCaxpby(syclQueue_t device_queue, int64_t n, float _Complex alpha,
                  const float _Complex *x, int64_t incx, float _Complex beta, float _Complex *y, int64_t incy);
void onemklZaxpby(syclQueue_t device_queue, int64_t n, double _Complex alpha,
                  const double _Complex *x, int64_t incx, double _Complex beta, double _Complex *y, int64_t incy);

void onemklSrot(syclQueue_t device_queue, int64_t n, float *x,
                int64_t incx, float *y, int64_t incy, float c, float s);
void onemklDrot(syclQueue_t device_queue, int64_t n, double *x,
                int64_t incx, double *y, int64_t incy, double c, double s);
void onemklCrot(syclQueue_t device_queue, int64_t n, float _Complex *x,
                int64_t incx, float _Complex *y, int64_t incy, float c, float _Complex s);
void onemklZrot(syclQueue_t device_queue, int64_t n, double _Complex *x,
                int64_t incx, double _Complex *y, int64_t incy, double c, double _Complex s);
void onemklCsrot(syclQueue_t device_queue, int64_t n, float _Complex *x,
                int64_t incx, float _Complex *y, int64_t incy, float c, float s);
void onemklZdrot(syclQueue_t device_queue, int64_t n, double _Complex *x,
                int64_t incx, double _Complex *y, int64_t incy, double c, double s);

// Level-1: scal oneMKL
void onemklDscal(syclQueue_t device_queue, int64_t n, double alpha,
                double *x, int64_t incx);
void onemklSscal(syclQueue_t device_queue, int64_t n, float alpha,
                float *x, int64_t incx);
void onemklCscal(syclQueue_t device_queue, int64_t n, float _Complex alpha,
                float _Complex *x, int64_t incx);
void onemklCsscal(syclQueue_t device_queue, int64_t n, float alpha,
                float _Complex *x, int64_t incx);
void onemklZscal(syclQueue_t device_queue, int64_t n, double _Complex alpha,
                double _Complex *x, int64_t incx);
void onemklZdscal(syclQueue_t device_queue, int64_t n, double alpha,
                double _Complex *x, int64_t incx);
void onemklHscal(syclQueue_t device_queue, int64_t n, uint16_t alpha, 
                short *x, int64_t incx);

void onemklChemv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                float _Complex alpha, const float _Complex *a, int64_t lda,
                const float _Complex *x, int64_t incx, float _Complex beta,
                float _Complex *y, int64_t incy);
void onemklZhemv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                double _Complex alpha, const double _Complex *a, int64_t lda,
                const double _Complex *x, int64_t incx, double _Complex beta,
                double _Complex *y, int64_t incy);
void onemklChbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                int64_t k, float _Complex alpha, const float _Complex *a,
                int64_t lda, const float _Complex *x, int64_t incx, float _Complex beta,
                float _Complex *y, int64_t incy);
void onemklZhbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                int64_t k, double _Complex alpha, const double _Complex *a,
                int64_t lda, const double _Complex *x, int64_t incx, double _Complex beta,
                double _Complex *y, int64_t incy);
void onemklCher(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float alpha,
                const float _Complex *x, int64_t incx, float _Complex *a,
                int64_t lda);
void onemklZher(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double alpha,
                const double _Complex *x, int64_t incx, double _Complex *a,
                int64_t lda);
void onemklCher2(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex alpha,
                const float _Complex *x, int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *a, int64_t lda);
void onemklZher2(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex alpha,
                const double _Complex *x, int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *a, int64_t lda);

void onemklSsbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t k,
                 float alpha, const float *a, int64_t lda, const float *x,
                 int64_t incx, float beta, float *y, int64_t incy);
void onemklDsbmv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t k,
                 double alpha, const double *a, int64_t lda, const double *x,
                 int64_t incx, double beta, double *y, int64_t incy);
void onemklSsymv(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float alpha,
                 const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                 float *y, int64_t incy);
void onemklDsymv(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                 double alpha, const double *a, int64_t lda, const double *x,
                 int64_t incx, double beta, double *y, int64_t incy);
void onemklSsyr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float alpha,
                           const float *x, int64_t incx, float *a, int64_t lda);
void onemklDsyr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double alpha,
                           const double *x, int64_t incx, double *a, int64_t lda);
void onemklStbmv(syclQueue_t device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const float *a, int64_t lda, float *x, int64_t incx);

void onemklDtbmv(syclQueue_t device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const double *a, int64_t lda, double *x, int64_t incx);

void onemklCtbmv(syclQueue_t device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtbmv(syclQueue_t device_queue, onemklUplo uplo,
                onemklTranspose trans, onemklDiag diag, int64_t n,
                int64_t k, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

void onemklStrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                int64_t incx);

void onemklDtrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                int64_t incx);

void onemklCtrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtrmv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

// trsv
void onemklStrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float *a, int64_t lda, float *x,
                int64_t incx);

void onemklDtrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double *a, int64_t lda, double *x,
                int64_t incx);

void onemklCtrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const float _Complex *a, int64_t lda, float _Complex *x,
                int64_t incx);

void onemklZtrsv(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans,
                onemklDiag diag, int64_t n, const double _Complex *a, int64_t lda, double _Complex *x,
                int64_t incx);

// Supported Level-1: Nrm2
void onemklDnrm2(syclQueue_t device_queue, int64_t n, const double *x,
                 int64_t incx, double *result);
void onemklSnrm2(syclQueue_t device_queue, int64_t n, const float *x,
                 int64_t incx, float *result);
void onemklCnrm2(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                 int64_t incx, float *result);
void onemklZnrm2(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                 int64_t incx, double *result);
void onemklHnrm2(syclQueue_t device_queue, int64_t n, const short *x,
                 int64_t incx, short *result);

void onemklSdot(syclQueue_t device_queue, int64_t n, const float *x,
                int64_t incx, const float *y, int64_t incy, float *result);
void onemklDdot(syclQueue_t device_queue, int64_t n, const double *x,
                int64_t incx, const double *y, int64_t incy, double *result);
void onemklCdotc(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *result);
void onemklZdotc(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *result);
void onemklCdotu(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                int64_t incx, const float _Complex *y, int64_t incy,
                float _Complex *result);
void onemklZdotu(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                int64_t incx, const double _Complex *y, int64_t incy,
                double _Complex *result);
void onemklHdot(syclQueue_t device_queue, int64_t n, const short *x,
                int64_t incx, const short *y, int64_t incy, short *result);

void onemklDcopy(syclQueue_t device_queue, int64_t n, const double *x,
                 int64_t incx, double *y, int64_t incy);
void onemklScopy(syclQueue_t device_queue, int64_t n, const float *x,
                 int64_t incx, float *y, int64_t incy);
void onemklZcopy(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                 int64_t incx, double _Complex *y, int64_t incy);
void onemklCcopy(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                 int64_t incx, float _Complex *y, int64_t incy);

void onemklDamax(syclQueue_t device_queue, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
void onemklSamax(syclQueue_t device_queue, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
void onemklZamax(syclQueue_t device_queue, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
void onemklCamax(syclQueue_t device_queue, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);

void onemklDamin(syclQueue_t device_queue, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
void onemklSamin(syclQueue_t device_queue, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
void onemklZamin(syclQueue_t device_queue, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
void onemklCamin(syclQueue_t device_queue, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);

void onemklSswap(syclQueue_t device_queue, int64_t n, float *x, int64_t incx,
                float *y, int64_t incy);
void onemklDswap(syclQueue_t device_queue, int64_t n, double *x, int64_t incx,
                double *y, int64_t incy);
void onemklCswap(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx,
                float _Complex *y, int64_t incy);
void onemklZswap(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx,
                double _Complex *y, int64_t incy);

void onemklDestroy(void);

// LAPACK
void onemklCgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda,
                   float *d, float *e, float _Complex *tauq, float _Complex *taup, float _Complex
                   *scratchpad, int64_t scratchpad_size);

void onemklDgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *d,
                   double *e, double *tauq, double *taup, double *scratchpad, int64_t
                   scratchpad_size);

void onemklSgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *d,
                   float *e, float *tauq, float *taup, float *scratchpad, int64_t scratchpad_size);

void onemklZgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                   double *d, double *e, double _Complex *tauq, double _Complex *taup, double
                   _Complex *scratchpad, int64_t scratchpad_size);

void onemklSgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau,
                   float *scratchpad, int64_t scratchpad_size);

void onemklDgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau,
                   double *scratchpad, int64_t scratchpad_size);

void onemklCgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda,
                   float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                   double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda,
                   float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklDgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau,
                   double *scratchpad, int64_t scratchpad_size);

void onemklSgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau,
                   float *scratchpad, int64_t scratchpad_size);

void onemklZgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                   double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda,
                   int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklDgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t
                   *ipiv, double *scratchpad, int64_t scratchpad_size);

void onemklSgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t
                   *ipiv, float *scratchpad, int64_t scratchpad_size);

void onemklZgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda,
                   int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCgetri(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv,
                   float _Complex *scratchpad, int64_t scratchpad_size);

void onemklDgetri(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t *ipiv, double
                   *scratchpad, int64_t scratchpad_size);

void onemklSgetri(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t *ipiv, float
                   *scratchpad, int64_t scratchpad_size);

void onemklZgetri(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t
                   *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float
                   _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *b, int64_t ldb, float
                   _Complex *scratchpad, int64_t scratchpad_size);

void onemklDgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double
                   *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad,
                   int64_t scratchpad_size);

void onemklSgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float
                   *a, int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t
                   scratchpad_size);

void onemklZgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double
                   _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *b, int64_t ldb, double
                   _Complex *scratchpad, int64_t scratchpad_size);

void onemklDgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                   int64_t n, double *a, int64_t lda, double *s, double *u, int64_t ldu, double *vt,
                   int64_t ldvt, double *scratchpad, int64_t scratchpad_size);

void onemklSgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                   int64_t n, float *a, int64_t lda, float *s, float *u, int64_t ldu, float *vt, int64_t
                   ldvt, float *scratchpad, int64_t scratchpad_size);

void onemklCgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                   int64_t n, float _Complex *a, int64_t lda, float *s, float _Complex *u, int64_t ldu,
                   float _Complex *vt, int64_t ldvt, float _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklZgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m,
                   int64_t n, double _Complex *a, int64_t lda, double *s, double _Complex *u, int64_t
                   ldu, double _Complex *vt, int64_t ldvt, double _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklCheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float
                   _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklZheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double
                   _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklChegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                   n, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float *w, float
                   _Complex *scratchpad, int64_t scratchpad_size);

void onemklZhegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                   n, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double *w,
                   double _Complex *scratchpad, int64_t scratchpad_size);

void onemklChetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                   lda, float *d, float *e, float _Complex *tau, float _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklZhetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                   lda, double *d, double *e, double _Complex *tau, double _Complex *scratchpad,
                   int64_t scratchpad_size);

void onemklChetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                   lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZhetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                   lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k,
                   float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size);

void onemklDorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k,
                   double *a, int64_t lda, double *tau, double *scratchpad, int64_t
                   scratchpad_size);

void onemklDorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda,
                   double *tau, double *scratchpad, int64_t scratchpad_size);

void onemklSorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda,
                   float *tau, float *scratchpad, int64_t scratchpad_size);

void onemklSorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                   *tau, float *scratchpad, int64_t scratchpad_size);

void onemklDorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                   double *tau, double *scratchpad, int64_t scratchpad_size);

void onemklSormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                   trans, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *c, int64_t
                   ldc, float *scratchpad, int64_t scratchpad_size);

void onemklDormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                   trans, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *c, int64_t
                   ldc, double *scratchpad, int64_t scratchpad_size);

void onemklSormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float
                   *scratchpad, int64_t scratchpad_size);

void onemklDormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc,
                   double *scratchpad, int64_t scratchpad_size);

void onemklDormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc,
                   double *scratchpad, int64_t scratchpad_size);

void onemklSormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float
                   *scratchpad, int64_t scratchpad_size);

void onemklSpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                   *scratchpad, int64_t scratchpad_size);

void onemklDpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                   double *scratchpad, int64_t scratchpad_size);

void onemklCpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                   lda, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                   lda, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                   *scratchpad, int64_t scratchpad_size);

void onemklDpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                   double *scratchpad, int64_t scratchpad_size);

void onemklCpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                   lda, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                   lda, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a,
                   int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size);

void onemklDpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a,
                   int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t
                   scratchpad_size);

void onemklCpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex
                   *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad,
                   int64_t scratchpad_size);

void onemklZpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double
                   _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex
                   *scratchpad, int64_t scratchpad_size);

void onemklDsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double *a,
                   int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size);

void onemklSsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float *a,
                   int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size);

void onemklDsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                   n, double *a, int64_t lda, double *b, int64_t ldb, double *w, double *scratchpad,
                   int64_t scratchpad_size);

void onemklSsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t
                   n, float *a, int64_t lda, float *b, int64_t ldb, float *w, float *scratchpad,
                   int64_t scratchpad_size);

void onemklDsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                   double *d, double *e, double *tau, double *scratchpad, int64_t scratchpad_size);

void onemklSsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float
                   *d, float *e, float *tau, float *scratchpad, int64_t scratchpad_size);

void onemklSsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda,
                   int64_t *ipiv, float *scratchpad, int64_t scratchpad_size);

void onemklDsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda,
                   int64_t *ipiv, double *scratchpad, int64_t scratchpad_size);

void onemklCsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                   lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                   lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                   diag, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b,
                   int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklDtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                   diag, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb,
                   double *scratchpad, int64_t scratchpad_size);

void onemklStrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                   diag, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float
                   *scratchpad, int64_t scratchpad_size);

void onemklZtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag
                   diag, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b,
                   int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k,
                   float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad,
                   int64_t scratchpad_size);

void onemklZungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k,
                   double _Complex *a, int64_t lda, double _Complex *tau, double _Complex
                   *scratchpad, int64_t scratchpad_size);

void onemklCungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a,
                   int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklZungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a,
                   int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklCungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t
                   lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t
                   lda, double _Complex *tau, double _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklCunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float
                   _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double
                   _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float
                   _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m,
                   int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double
                   _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklCunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                   trans, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau,
                   float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklZunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose
                   trans, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau,
                   double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t
                   scratchpad_size);

void onemklSgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda,
                         int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size,
                         float *scratchpad, int64_t scratchpad_size);

void onemklDgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size,
                         double *scratchpad, int64_t scratchpad_size);

void onemklCgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t
                         lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t
                         batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t
                         lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t
                         batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSgetri_batch(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t
                         stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float
                         *scratchpad, int64_t scratchpad_size);

void onemklDgetri_batch(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t
                         stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double
                         *scratchpad, int64_t scratchpad_size);

void onemklCgetri_batch(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t
                         stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float
                         _Complex *scratchpad, int64_t scratchpad_size);

void onemklZgetri_batch(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda,
                         int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                         batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                         stride_ipiv, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                         float *scratchpad, int64_t scratchpad_size);

void onemklDgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                         stride_ipiv, double *b, int64_t ldb, int64_t stride_b, int64_t
                         batch_size, double *scratchpad, int64_t scratchpad_size);

void onemklCgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                         stride_ipiv, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t
                         batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs,
                         double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t
                         stride_ipiv, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t
                         batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda,
                         int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                         batch_size, float *scratchpad, int64_t scratchpad_size);

void onemklDgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                         batch_size, double *scratchpad, int64_t scratchpad_size);

void onemklCgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t
                         lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                         batch_size, float _Complex *scratchpad, int64_t scratchpad_size);

void onemklZgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t
                         lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t
                         batch_size, double _Complex *scratchpad, int64_t scratchpad_size);

void onemklSorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t
                         lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t
                         batch_size, float *scratchpad, int64_t scratchpad_size);

void onemklDorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t
                         lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t
                         batch_size, double *scratchpad, int64_t scratchpad_size);

void onemklSpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda,
                         int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t
                         scratchpad_size);

void onemklDpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t
                         lda, int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t
                         scratchpad_size);

void onemklCpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a,
                         int64_t lda, int64_t stride_a, int64_t batch_size, float _Complex
                         *scratchpad, int64_t scratchpad_size);

void onemklZpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a,
                         int64_t lda, int64_t stride_a, int64_t batch_size, double _Complex
                         *scratchpad, int64_t scratchpad_size);

void onemklSpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float
                         *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b,
                         int64_t batch_size, float *scratchpad, int64_t scratchpad_size);

void onemklDpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double
                         *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t
                         stride_b, int64_t batch_size, double *scratchpad, int64_t
                         scratchpad_size);

void onemklCpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float
                         _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb,
                         int64_t stride_b, int64_t batch_size, float _Complex *scratchpad,
                         int64_t scratchpad_size);

void onemklZpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double
                         _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t
                         ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad,
                         int64_t scratchpad_size);

void onemklCungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a,
                         int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau,
                         int64_t batch_size, float _Complex *scratchpad, int64_t
                         scratchpad_size);

void onemklZungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex
                         *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t
                         stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t
                         scratchpad_size);

int64_t onemklSgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklSgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklSgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklSgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd
                                      jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu,
                                      int64_t ldvt);

int64_t onemklDgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd
                                      jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu,
                                      int64_t ldvt);

int64_t onemklCgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd
                                      jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu,
                                      int64_t ldvt);

int64_t onemklZgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd
                                      jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu,
                                      int64_t ldvt);

int64_t onemklSgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklDgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklCgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklZgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda);

int64_t onemklSgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklDgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklCgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklZgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda);

int64_t onemklSgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklDgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklCgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklZgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklCheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                      int64_t n, int64_t lda);

int64_t onemklZheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                      int64_t n, int64_t lda);

int64_t onemklChegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                      onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int64_t onemklZhegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                      onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int64_t onemklChetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklZhetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklChetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklZhetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklSorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                      int64_t n, int64_t k, int64_t lda);

int64_t onemklDorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                      int64_t n, int64_t k, int64_t lda);

int64_t onemklSorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklDorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklSorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                      int64_t lda);

int64_t onemklDorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                      int64_t lda);

int64_t onemklSormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklDormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklSormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklDormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklSormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                      onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                      int64_t ldc);

int64_t onemklDormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                      onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                      int64_t ldc);

int64_t onemklSpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklDpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklCpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklZpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklSpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklDpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklCpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklZpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t nrhs, int64_t lda, int64_t ldb);

int64_t onemklSpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklDpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklCpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklZpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklSsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklDsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklCsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklZsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklSsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                      int64_t n, int64_t lda);

int64_t onemklDsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo,
                                      int64_t n, int64_t lda);

int64_t onemklSsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                      onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int64_t onemklDsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz,
                                      onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb);

int64_t onemklSsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklDsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklStrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                      trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t
                                      lda, int64_t ldb);

int64_t onemklDtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                      trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t
                                      lda, int64_t ldb);

int64_t onemklCtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                      trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t
                                      lda, int64_t ldb);

int64_t onemklZtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose
                                      trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t
                                      lda, int64_t ldb);

int64_t onemklCungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                      int64_t n, int64_t k, int64_t lda);

int64_t onemklZungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m,
                                      int64_t n, int64_t k, int64_t lda);

int64_t onemklCungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                      int64_t lda);

int64_t onemklZungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k,
                                      int64_t lda);

int64_t onemklCungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklZungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n,
                                      int64_t lda);

int64_t onemklCunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklZunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklCunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklZunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose
                                      trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t
                                      ldc);

int64_t onemklCunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                      onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                      int64_t ldc);

int64_t onemklZunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo,
                                      onemklTranspose trans, int64_t m, int64_t n, int64_t lda,
                                      int64_t ldc);

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

int64_t onemklSgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda,
                                            int64_t stride_a, int64_t stride_ipiv, int64_t
                                            batch_size);

int64_t onemklDgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda,
                                            int64_t stride_a, int64_t stride_ipiv, int64_t
                                            batch_size);

int64_t onemklCgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda,
                                            int64_t stride_a, int64_t stride_ipiv, int64_t
                                            batch_size);

int64_t onemklZgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda,
                                            int64_t stride_a, int64_t stride_ipiv, int64_t
                                            batch_size);

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

#ifdef __cplusplus
}
#endif
