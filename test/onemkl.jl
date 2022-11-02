using oneAPI
using oneAPI.oneMKL

using LinearAlgebra

m = 20

A = oneArray(rand(Float64, m))
#maxVal = oneMKL.iamax(m, A)
minVal = oneMKL.iamin(m, A)