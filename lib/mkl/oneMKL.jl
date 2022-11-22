module oneMKL

using ..oneAPI
using ..oneAPI: liboneapi_support

using ..oneAPI.oneL0

using ..oneAPI.SYCL
using ..oneAPI.SYCL: syclQueue_t

using GPUArrays

include("libonemkl.jl")

# Exclude Float16 for now, since many oneMKL functions - copy, scal, do not take Float16
const onemklFloat = Union{Float64,Float32,ComplexF64,ComplexF32}

include("wrappers.jl")
include("linalg.jl")

end
