#include "onemkl_dft.h"
#include "sycl.hpp"  // internal struct definitions

#include <oneapi/mkl/dft.hpp>
#include <vector>
#include <complex>
#include <new>
#include <exception>
#include <cstring>

using namespace oneapi::mkl::dft;

struct onemklDftDescriptor_st {
    precision prec;
    domain dom;
    void *ptr; // pointer to concrete descriptor<prec, dom>
};

static inline precision to_prec(onemklDftPrecision p) {
    return (p == ONEMKL_DFT_PRECISION_DOUBLE) ? precision::DOUBLE : precision::SINGLE;
}

static inline domain to_dom(onemklDftDomain d) {
    return (d == ONEMKL_DFT_DOMAIN_COMPLEX) ? domain::COMPLEX : domain::REAL;
}

// Helper to allocate descriptor depending on precision/domain
static int allocate_descriptor(onemklDftDescriptor_t *out, precision p, domain d, const std::vector<int64_t> &lengths) {
    try {
        auto *desc = new onemklDftDescriptor_st();
        desc->prec = p;
        desc->dom = d;
        if (p == precision::SINGLE && d == domain::REAL) {
            desc->ptr = new descriptor<precision::SINGLE, domain::REAL>(lengths);
        } else if (p == precision::SINGLE && d == domain::COMPLEX) {
            desc->ptr = new descriptor<precision::SINGLE, domain::COMPLEX>(lengths);
        } else if (p == precision::DOUBLE && d == domain::REAL) {
            desc->ptr = new descriptor<precision::DOUBLE, domain::REAL>(lengths);
        } else { // DOUBLE COMPLEX
            desc->ptr = new descriptor<precision::DOUBLE, domain::COMPLEX>(lengths);
        }
        *out = desc;
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftCreate1D(onemklDftDescriptor_t *desc,
                      onemklDftPrecision precision,
                      onemklDftDomain domain,
                      int64_t length) {
    std::vector<int64_t> dims{length};
    return allocate_descriptor(desc, to_prec(precision), to_dom(domain), dims);
}

int onemklDftCreateND(onemklDftDescriptor_t *desc,
                      onemklDftPrecision precision,
                      onemklDftDomain domain,
                      int64_t dim,
                      const int64_t *lengths) {
    if (dim <= 0 || lengths == nullptr) return -2;
    std::vector<int64_t> dims(lengths, lengths + dim);
    return allocate_descriptor(desc, to_prec(precision), to_dom(domain), dims);
}

int onemklDftDestroy(onemklDftDescriptor_t desc) {
    if (!desc) return 0;
    try {
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) {
            delete static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc->ptr);
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) {
            delete static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc->ptr);
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) {
            delete static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc->ptr);
        } else {
            delete static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc->ptr);
        }
        delete desc;
        return 0;
    } catch (...) {
        return -1;
    }
}

int onemklDftCommit(onemklDftDescriptor_t desc, syclQueue_t queue) {
    if (!desc || !queue) return -2;
    try {
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) {
            static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc->ptr)->commit(queue->val);
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) {
            static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc->ptr)->commit(queue->val);
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) {
            static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc->ptr)->commit(queue->val);
        } else {
            static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc->ptr)->commit(queue->val);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// Internal mapping helpers. We cannot rely on numeric equality between our
// exported onemklDftConfigParam enumeration values (which are compact and
// stable for Julia) and oneMKL's internal sparse enum values. Provide an
// explicit translation layer.
static inline config_param to_param(onemklDftConfigParam p) {
    switch(p) {
        case ONEMKL_DFT_PARAM_FORWARD_DOMAIN: return config_param::FORWARD_DOMAIN;
        case ONEMKL_DFT_PARAM_DIMENSION: return config_param::DIMENSION;
        case ONEMKL_DFT_PARAM_LENGTHS: return config_param::LENGTHS;
        case ONEMKL_DFT_PARAM_PRECISION: return config_param::PRECISION;
        case ONEMKL_DFT_PARAM_FORWARD_SCALE: return config_param::FORWARD_SCALE;
        case ONEMKL_DFT_PARAM_BACKWARD_SCALE: return config_param::BACKWARD_SCALE;
        case ONEMKL_DFT_PARAM_NUMBER_OF_TRANSFORMS: return config_param::NUMBER_OF_TRANSFORMS;
        case ONEMKL_DFT_PARAM_COMPLEX_STORAGE: return config_param::COMPLEX_STORAGE;
        case ONEMKL_DFT_PARAM_PLACEMENT: return config_param::PLACEMENT;
        case ONEMKL_DFT_PARAM_INPUT_STRIDES: return config_param::INPUT_STRIDES;
        case ONEMKL_DFT_PARAM_OUTPUT_STRIDES: return config_param::OUTPUT_STRIDES;
        case ONEMKL_DFT_PARAM_FWD_DISTANCE: return config_param::FWD_DISTANCE;
        case ONEMKL_DFT_PARAM_BWD_DISTANCE: return config_param::BWD_DISTANCE;
        case ONEMKL_DFT_PARAM_WORKSPACE: return config_param::WORKSPACE;
        case ONEMKL_DFT_PARAM_WORKSPACE_ESTIMATE_BYTES: return config_param::WORKSPACE_ESTIMATE_BYTES;
        case ONEMKL_DFT_PARAM_WORKSPACE_BYTES: return config_param::WORKSPACE_BYTES;
        case ONEMKL_DFT_PARAM_FWD_STRIDES: return config_param::FWD_STRIDES;
        case ONEMKL_DFT_PARAM_BWD_STRIDES: return config_param::BWD_STRIDES;
        case ONEMKL_DFT_PARAM_WORKSPACE_PLACEMENT: return config_param::WORKSPACE_PLACEMENT;
        case ONEMKL_DFT_PARAM_WORKSPACE_EXTERNAL_BYTES: return config_param::WORKSPACE_EXTERNAL_BYTES;
        default: return config_param::FORWARD_DOMAIN; // defensive; shouldn't happen
    }
}
// Explicit value mapping (avoid relying on underlying enum integral values)
static inline config_value to_cvalue(onemklDftConfigValue v) {
    switch (v) {
        case ONEMKL_DFT_VALUE_COMMITTED: return config_value::COMMITTED;
        case ONEMKL_DFT_VALUE_UNCOMMITTED: return config_value::UNCOMMITTED;
        case ONEMKL_DFT_VALUE_COMPLEX_COMPLEX: return config_value::COMPLEX_COMPLEX;
        case ONEMKL_DFT_VALUE_REAL_REAL: return config_value::REAL_REAL;
        case ONEMKL_DFT_VALUE_INPLACE: return config_value::INPLACE;
        case ONEMKL_DFT_VALUE_NOT_INPLACE: return config_value::NOT_INPLACE;
        case ONEMKL_DFT_VALUE_WORKSPACE_AUTOMATIC: return config_value::WORKSPACE_AUTOMATIC;
        case ONEMKL_DFT_VALUE_ALLOW: return config_value::ALLOW;
        case ONEMKL_DFT_VALUE_AVOID: return config_value::AVOID;
        case ONEMKL_DFT_VALUE_WORKSPACE_INTERNAL: return config_value::WORKSPACE_INTERNAL;
        case ONEMKL_DFT_VALUE_WORKSPACE_EXTERNAL: return config_value::WORKSPACE_EXTERNAL;
        default: return config_value::UNCOMMITTED; // defensive fallback
    }
}

static inline onemklDftConfigValue from_cvalue(config_value cv) {
    switch (cv) {
        case config_value::COMMITTED: return ONEMKL_DFT_VALUE_COMMITTED;
        case config_value::UNCOMMITTED: return ONEMKL_DFT_VALUE_UNCOMMITTED;
        case config_value::COMPLEX_COMPLEX: return ONEMKL_DFT_VALUE_COMPLEX_COMPLEX;
        case config_value::REAL_REAL: return ONEMKL_DFT_VALUE_REAL_REAL;
        case config_value::INPLACE: return ONEMKL_DFT_VALUE_INPLACE;
        case config_value::NOT_INPLACE: return ONEMKL_DFT_VALUE_NOT_INPLACE;
        case config_value::WORKSPACE_AUTOMATIC: return ONEMKL_DFT_VALUE_WORKSPACE_AUTOMATIC;
        case config_value::ALLOW: return ONEMKL_DFT_VALUE_ALLOW;
        case config_value::AVOID: return ONEMKL_DFT_VALUE_AVOID;
        case config_value::WORKSPACE_INTERNAL: return ONEMKL_DFT_VALUE_WORKSPACE_INTERNAL;
        case config_value::WORKSPACE_EXTERNAL: return ONEMKL_DFT_VALUE_WORKSPACE_EXTERNAL;
        default: return ONEMKL_DFT_VALUE_UNCOMMITTED; // unknown / unsupported -> safe default
    }
}

// Dispatch macro re-used for configuration
#define ONEMKL_DFT_DISPATCH_CFG(desc_expr, CALL) \
    do { \
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } \
    } while (0)

int onemklDftSetValueInt64(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t value) {
    if (!desc) return -2; if (!desc->ptr) return -3;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftSetValueDouble(onemklDftDescriptor_t desc, onemklDftConfigParam param, double value) {
    if (!desc) return -2; if (!desc->ptr) return -3;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftSetValueInt64Array(onemklDftDescriptor_t desc, onemklDftConfigParam param, const int64_t *values, int64_t n) {
    if (!desc || !values || n < 0) return -2; if (!desc->ptr) return -3;
    try { std::vector<int64_t> v(values, values + n); ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), v)); return 0; } catch (...) { return -1; }
}

int onemklDftSetValueConfigValue(onemklDftDescriptor_t desc, onemklDftConfigParam param, onemklDftConfigValue value) {
    if (!desc) return -2; if (!desc->ptr) return -3;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->set_value(to_param(param), to_cvalue(value))); return 0; } catch (...) { return -1; }
}

int onemklDftGetValueInt64(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t *value) {
    if (!desc || !value) return -2; if (!desc->ptr) return -3;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftGetValueDouble(onemklDftDescriptor_t desc, onemklDftConfigParam param, double *value) {
    if (!desc || !value) return -2; if (!desc->ptr) return -3;
    try { ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), value)); return 0; } catch (...) { return -1; }
}

int onemklDftGetValueInt64Array(onemklDftDescriptor_t desc, onemklDftConfigParam param, int64_t *values, int64_t *n) {
    if (!desc || !values || !n || *n <= 0) return -2; if (!desc->ptr) return -3;
    try {
        std::vector<int64_t> v; ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), &v));
        int64_t to_copy = (*n < (int64_t)v.size()) ? *n : (int64_t)v.size();
        std::memcpy(values, v.data(), sizeof(int64_t)*to_copy);
        *n = to_copy; return 0;
    } catch (...) { return -1; }
}

int onemklDftGetValueConfigValue(onemklDftDescriptor_t desc, onemklDftConfigParam param, onemklDftConfigValue *value) {
    if (!desc || !value) return -2; if (!desc->ptr) return -3;
    try { config_value cv; ONEMKL_DFT_DISPATCH_CFG(desc->ptr, d->get_value(to_param(param), &cv)); *value = from_cvalue(cv); return 0; } catch (...) { return -1; }
}

// Helper macro to dispatch compute operations
#define ONEMKL_DFT_DISPATCH(desc_expr, CALL) \
    do { \
        if (desc->prec == precision::SINGLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::SINGLE && desc->dom == domain::COMPLEX) { \
            auto *d = static_cast< descriptor<precision::SINGLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } else if (desc->prec == precision::DOUBLE && desc->dom == domain::REAL) { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::REAL>* >(desc_expr); \
            CALL; \
        } else { \
            auto *d = static_cast< descriptor<precision::DOUBLE, domain::COMPLEX>* >(desc_expr); \
            CALL; \
        } \
    } while (0)

// Pointer (USM) dispatch with proper element typing rather than using void* directly.
// Using void* caused instantiation of compute_forward/backward with <void> template
// parameters on some oneMKL versions, leading to unresolved symbols at runtime.
int onemklDftComputeForward(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2; if (!desc->ptr) return -3;
    try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) {
                auto *p = static_cast<float*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, p).wait());
            } else {
                auto *p = static_cast<double*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, p).wait());
            }
        } else { // COMPLEX
            if (desc->prec == precision::SINGLE) {
                auto *p = static_cast<std::complex<float>*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, p).wait());
            } else {
                auto *p = static_cast<std::complex<double>*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, p).wait());
            }
        }
        return 0;
    } catch (...) { return -1; }
}

int onemklDftComputeForwardOutOfPlace(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2; if (!desc->ptr) return -3;
    try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) {
                // Real-domain forward transform: real input -> complex output
                auto *pi = static_cast<float*>(in);
                auto *po = static_cast<std::complex<float>*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, pi, po).wait());
            } else {
                auto *pi = static_cast<double*>(in);
                auto *po = static_cast<std::complex<double>*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, pi, po).wait());
            }
        } else { // COMPLEX
            if (desc->prec == precision::SINGLE) {
                auto *pi = static_cast<std::complex<float>*>(in);
                auto *po = static_cast<std::complex<float>*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, pi, po).wait());
            } else {
                auto *pi = static_cast<std::complex<double>*>(in);
                auto *po = static_cast<std::complex<double>*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, pi, po).wait());
            }
        }
        return 0;
    } catch (...) { return -1; }
}

int onemklDftComputeBackward(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2; if (!desc->ptr) return -3;
    try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) {
                auto *p = static_cast<float*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, p).wait());
            } else {
                auto *p = static_cast<double*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, p).wait());
            }
        } else { // COMPLEX
            if (desc->prec == precision::SINGLE) {
                auto *p = static_cast<std::complex<float>*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, p).wait());
            } else {
                auto *p = static_cast<std::complex<double>*>(inout);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, p).wait());
            }
        }
        return 0;
    } catch (...) { return -1; }
}

int onemklDftComputeBackwardOutOfPlace(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2; if (!desc->ptr) return -3;
    try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) {
                // Real-domain backward transform: complex input -> real output
                auto *pi = static_cast<std::complex<float>*>(in);
                auto *po = static_cast<float*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, pi, po).wait());
            } else {
                auto *pi = static_cast<std::complex<double>*>(in);
                auto *po = static_cast<double*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, pi, po).wait());
            }
        } else { // COMPLEX
            if (desc->prec == precision::SINGLE) {
                auto *pi = static_cast<std::complex<float>*>(in);
                auto *po = static_cast<std::complex<float>*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, pi, po).wait());
            } else {
                auto *pi = static_cast<std::complex<double>*>(in);
                auto *po = static_cast<std::complex<double>*>(out);
                ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, pi, po).wait());
            }
        }
        return 0;
    } catch (...) { return -1; }
}

// Keep dispatch macros defined for buffer variants below; undef at end of file.

// Buffer API helpers: create temporary buffers referencing host memory.
// NOTE: This assumes the memory is accessible and sized appropriately.
template <typename T>
static inline sycl::buffer<T,1> make_buffer(T *ptr, int64_t n) {
    return sycl::buffer<T,1>(ptr, sycl::range<1>(static_cast<size_t>(n)));
}

// Query total element count from LENGTHS config (product of lengths).
static int64_t get_element_count(onemklDftDescriptor_t desc) {
    int64_t n = 0; int64_t dims = 0; if (onemklDftGetValueInt64(desc, ONEMKL_DFT_PARAM_DIMENSION, &dims) != 0) return -1; if (dims <= 0 || dims > 8) return -1; int64_t lens[16]; int64_t want = dims; if (onemklDftGetValueInt64Array(desc, ONEMKL_DFT_PARAM_LENGTHS, lens, &want) != 0) return -1; if (want != dims) return -1; int64_t total = 1; for (int i=0;i<dims;i++){ if (lens[i]<=0) return -1; total *= lens[i]; } return total; }

// Select real/complex element size variant for pointers.
int onemklDftComputeForwardBuffer(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2; if (!desc->ptr) return -3; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((float*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
            else { auto buf = make_buffer((double*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
        } else { // COMPLEX
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((std::complex<float>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
            else { auto buf = make_buffer((std::complex<double>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, buf)); }
        }
        return 0; } catch (...) { return -1; }
}

int onemklDftComputeForwardOutOfPlaceBuffer(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2; if (!desc->ptr) return -3; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((float*)in, n); /* complex output size may differ; assume caller sized */ auto bufo = make_buffer((std::complex<float>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((double*)in, n); auto bufo = make_buffer((std::complex<double>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
        } else {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((std::complex<float>*)in, n); auto bufo = make_buffer((std::complex<float>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((std::complex<double>*)in, n); auto bufo = make_buffer((std::complex<double>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_forward(*d, bufi, bufo)); }
        }
        return 0; } catch (...) { return -1; }
}

int onemklDftComputeBackwardBuffer(onemklDftDescriptor_t desc, void *inout) {
    if (!desc || !inout) return -2; if (!desc->ptr) return -3; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((float*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
            else { auto buf = make_buffer((double*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
        } else {
            if (desc->prec == precision::SINGLE) { auto buf = make_buffer((std::complex<float>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
            else { auto buf = make_buffer((std::complex<double>*)inout, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, buf)); }
        }
        return 0; } catch (...) { return -1; }
}

int onemklDftComputeBackwardOutOfPlaceBuffer(onemklDftDescriptor_t desc, void *in, void *out) {
    if (!desc || !in || !out) return -2; if (!desc->ptr) return -3; int64_t n = get_element_count(desc); if (n <= 0) return -3; try {
        if (desc->dom == domain::REAL) {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((std::complex<float>*)in, n); auto bufo = make_buffer((float*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((std::complex<double>*)in, n); auto bufo = make_buffer((double*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
        } else {
            if (desc->prec == precision::SINGLE) { auto bufi = make_buffer((std::complex<float>*)in, n); auto bufo = make_buffer((std::complex<float>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
            else { auto bufi = make_buffer((std::complex<double>*)in, n); auto bufo = make_buffer((std::complex<double>*)out, n); ONEMKL_DFT_DISPATCH(desc->ptr, compute_backward(*d, bufi, bufo)); }
        }
        return 0; } catch (...) { return -1; }
}

#undef ONEMKL_DFT_DISPATCH
#undef ONEMKL_DFT_DISPATCH_CFG

// Introspection helper: capture integral values of config_param enums that we
// rely upon in the Julia layer. We enumerate the sequence present in our C
// header; if oneMKL's internal ordering diverges this will expose it.
int onemklDftQueryParamIndices(int64_t *out, int64_t n) {
    if (!out || n < 20) return -2; // we expose 20 params currently
    try {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        config_param params[] = {
            config_param::FORWARD_DOMAIN,
            config_param::DIMENSION,
            config_param::LENGTHS,
            config_param::PRECISION,
            config_param::FORWARD_SCALE,
            config_param::BACKWARD_SCALE,
            config_param::NUMBER_OF_TRANSFORMS,
            config_param::COMPLEX_STORAGE,
            config_param::PLACEMENT,
            config_param::INPUT_STRIDES,
            config_param::OUTPUT_STRIDES,
            config_param::FWD_DISTANCE,
            config_param::BWD_DISTANCE,
            config_param::WORKSPACE,
            config_param::WORKSPACE_ESTIMATE_BYTES,
            config_param::WORKSPACE_BYTES,
            config_param::FWD_STRIDES,
            config_param::BWD_STRIDES,
            config_param::WORKSPACE_PLACEMENT,
            config_param::WORKSPACE_EXTERNAL_BYTES
        };
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        for (int i=0;i<20;i++) out[i] = static_cast<int64_t>(params[i]);
        return 20;
    } catch (...) { return -1; }
}
