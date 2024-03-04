cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(oneAPISupport)

add_library(oneapi_support SHARED src/sycl.h src/sycl.hpp src/sycl.cpp src/onemkl.h src/onemkl.cpp)

target_link_libraries(oneapi_support
  mkl_sycl
  mkl_intel_ilp64
  mkl_sequential
  mkl_core
  sycl
  OpenCL

  # XXX: we don't want to link against this plugin, but otherwise the run-time
  #      loader doesn't find it (since it's located in the non-global Conda
  #      library directory, and we can't set LD_LIBRARY_PATH from within Julia).
  pi_level_zero
)

install(TARGETS oneapi_support
        LIBRARY DESTINATION lib)
