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

typedef struct MatrixHandle_st *MatrixHandle_t;
typedef struct MatmatDescr_st *MatmatDescr_t;

