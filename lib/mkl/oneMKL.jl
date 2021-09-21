module oneMKL

using ..oneAPI
using ..oneAPI: liboneapilib

using ..oneAPI.oneL0

using ..oneAPI.SYCL
using ..oneAPI.SYCL: syclQueue_t

include("libonemkl.jl")

const onemklFloat = Union{Float64,Float32,Float16,ComplexF64,ComplexF32}

include("wrappers.jl")
include("linalg.jl")

end
