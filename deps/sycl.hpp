#pragma once

#include "sycl.h"

#include <CL/sycl.hpp>

struct syclPlatform_st {
    sycl::platform val;
};

struct syclDevice_st {
    sycl::device val;
};

struct syclContext_st {
    sycl::context val;
};

struct syclQueue_st {
    sycl::queue val;
};

struct syclEvent_st {
    sycl::event val;
};
