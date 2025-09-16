#pragma once

#include "sycl.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes (negative values indicate errors):
//  0  : success
// -1  : internal error / exception caught
// -2  : invalid argument (null pointer, bad length, etc.)
// -3  : invalid descriptor state (e.g. uninitialized desc->ptr) or size query failure
#define ONEMKL_DFT_STATUS_SUCCESS            0
#define ONEMKL_DFT_STATUS_ERROR             -1
#define ONEMKL_DFT_STATUS_INVALID_ARGUMENT  -2
#define ONEMKL_DFT_STATUS_BAD_STATE         -3

// DFT precision
typedef enum {
    ONEMKL_DFT_PRECISION_SINGLE = 0,
    ONEMKL_DFT_PRECISION_DOUBLE = 1
} onemklDftPrecision;

// DFT domain
typedef enum {
    ONEMKL_DFT_DOMAIN_REAL = 0,
    ONEMKL_DFT_DOMAIN_COMPLEX = 1
} onemklDftDomain;

// Configuration parameters (subset mirrors oneapi::mkl::dft::config_param)
typedef enum {
    ONEMKL_DFT_PARAM_FORWARD_DOMAIN = 0,
    ONEMKL_DFT_PARAM_DIMENSION,
    ONEMKL_DFT_PARAM_LENGTHS,
    ONEMKL_DFT_PARAM_PRECISION,
    ONEMKL_DFT_PARAM_FORWARD_SCALE,
    ONEMKL_DFT_PARAM_BACKWARD_SCALE,
    ONEMKL_DFT_PARAM_NUMBER_OF_TRANSFORMS,
    ONEMKL_DFT_PARAM_COMPLEX_STORAGE,
    ONEMKL_DFT_PARAM_PLACEMENT,
    ONEMKL_DFT_PARAM_INPUT_STRIDES,
    ONEMKL_DFT_PARAM_OUTPUT_STRIDES,
    ONEMKL_DFT_PARAM_FWD_DISTANCE,
    ONEMKL_DFT_PARAM_BWD_DISTANCE,
    ONEMKL_DFT_PARAM_WORKSPACE,              // size query / placement
    ONEMKL_DFT_PARAM_WORKSPACE_ESTIMATE_BYTES,
    ONEMKL_DFT_PARAM_WORKSPACE_BYTES,
    ONEMKL_DFT_PARAM_FWD_STRIDES,
    ONEMKL_DFT_PARAM_BWD_STRIDES,
    ONEMKL_DFT_PARAM_WORKSPACE_PLACEMENT,
    ONEMKL_DFT_PARAM_WORKSPACE_EXTERNAL_BYTES
} onemklDftConfigParam;

// Configuration values (mirrors oneapi::mkl::dft::config_value)
typedef enum {
    ONEMKL_DFT_VALUE_COMMITTED = 0,
    ONEMKL_DFT_VALUE_UNCOMMITTED,
    ONEMKL_DFT_VALUE_COMPLEX_COMPLEX,
    ONEMKL_DFT_VALUE_REAL_REAL,
    ONEMKL_DFT_VALUE_INPLACE,
    ONEMKL_DFT_VALUE_NOT_INPLACE,
    ONEMKL_DFT_VALUE_WORKSPACE_AUTOMATIC,   // internal
    ONEMKL_DFT_VALUE_ALLOW,
    ONEMKL_DFT_VALUE_AVOID,
    ONEMKL_DFT_VALUE_WORKSPACE_INTERNAL,
    ONEMKL_DFT_VALUE_WORKSPACE_EXTERNAL
} onemklDftConfigValue;

// Opaque descriptor handle
struct onemklDftDescriptor_st;
typedef struct onemklDftDescriptor_st *onemklDftDescriptor_t;

// Creation / destruction
int onemklDftCreate1D(onemklDftDescriptor_t *desc,
                      onemklDftPrecision precision,
                      onemklDftDomain domain,
                      int64_t length);

int onemklDftCreateND(onemklDftDescriptor_t *desc,
                      onemklDftPrecision precision,
                      onemklDftDomain domain,
                      int64_t dim,
                      const int64_t *lengths);

int onemklDftDestroy(onemklDftDescriptor_t desc);

// Commit descriptor to a queue
int onemklDftCommit(onemklDftDescriptor_t desc, syclQueue_t queue);

// Configuration set
int onemklDftSetValueInt64(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t value);
int onemklDftSetValueDouble(onemklDftDescriptor_t desc, onemklDftConfigParam param, double value);
int onemklDftSetValueInt64Array(onemklDftDescriptor_t desc, onemklDftConfigParam param, const int64_t *values, int64_t n);
int onemklDftSetValueConfigValue(onemklDftDescriptor_t desc, onemklDftConfigParam param, onemklDftConfigValue value);

// Configuration get
int onemklDftGetValueInt64(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t *value);
int onemklDftGetValueDouble(onemklDftDescriptor_t desc, onemklDftConfigParam param, double *value);
// For array queries pass *n as available length; on return *n has elements written.
int onemklDftGetValueInt64Array(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t *values, int64_t *n);
int onemklDftGetValueConfigValue(onemklDftDescriptor_t desc, onemklDftConfigParam param, onemklDftConfigValue *value);

// Compute (USM) in-place/out-of-place. Pointers must reference memory
// appropriate for precision/domain. No size checking is performed.
int onemklDftComputeForward(onemklDftDescriptor_t desc, void *inout);
int onemklDftComputeForwardOutOfPlace(onemklDftDescriptor_t desc, void *in, void *out);
int onemklDftComputeBackward(onemklDftDescriptor_t desc, void *inout);
int onemklDftComputeBackwardOutOfPlace(onemklDftDescriptor_t desc, void *in, void *out);

// Compute (buffer API) variants. Host pointers are wrapped in temporary 1D buffers.
int onemklDftComputeForwardBuffer(onemklDftDescriptor_t desc, void *inout);
int onemklDftComputeForwardOutOfPlaceBuffer(onemklDftDescriptor_t desc, void *in, void *out);
int onemklDftComputeBackwardBuffer(onemklDftDescriptor_t desc, void *inout);
int onemklDftComputeBackwardOutOfPlaceBuffer(onemklDftDescriptor_t desc, void *in, void *out);

// Introspection: write out the integral values of selected config_param enums in
// the same order as our public enum declaration above. Returns number written or
// a negative error code if n is insufficient or arguments invalid.
int onemklDftQueryParamIndices(int64_t *out, int64_t n);

#ifdef __cplusplus
}
#endif
