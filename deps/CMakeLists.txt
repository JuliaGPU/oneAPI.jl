cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_COMPILER "dpcpp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(oneAPILib)

find_package(JlCxx)

add_library(oneapilib SHARED sycl.cpp onemkl.cpp)

target_link_libraries(oneapilib
  JlCxx::cxxwrap_julia
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

install(TARGETS oneapilib
        LIBRARY DESTINATION lib)
