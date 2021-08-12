module oneMKL

using ..oneAPI
using ..oneAPI.oneL0

const onemklFloat = Union{Float64,Float32,Float16,ComplexF64,ComplexF32}

include("wrappers.jl")
include("linalg.jl")

end
