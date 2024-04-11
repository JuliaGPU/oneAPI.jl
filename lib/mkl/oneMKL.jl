module oneMKL

using ..oneAPI
using ..oneAPI: unsafe_free!
using ..oneL0

using ..Support

using ..SYCL
using ..SYCL: syclQueue_t

using GPUArrays

using LinearAlgebra
using LinearAlgebra: checksquare
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo

using SparseArrays

# Exclude Float16 for now, since many oneMKL functions do not take Float16
const onemklFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const onemklComplex = Union{ComplexF32,ComplexF64}
const onemklHalf = Float16

include("array.jl")
include("utils.jl")
include("wrappers_blas.jl")
include("wrappers_lapack.jl")
include("wrappers_sparse.jl")
include("linalg.jl")
include("interfaces.jl")

function band(A::StridedArray, kl, ku)
    m, n = size(A)
    AB = zeros(eltype(A),kl+ku+1,n)
    for j = 1:n
        for i = max(1,j-ku):min(m,j+kl)
            AB[ku+1-j+i,j] = A[i,j]
        end
    end
    return AB
end

# convert band storage to general matrix
function unband(AB::StridedArray,m,kl,ku)
    bm, n = size(AB)
    A = zeros(eltype(AB),m,n)
    for j = 1:n
        for i = max(1,j-ku):min(m,j+kl)
            A[i,j] = AB[ku+1-j+i,j]
        end
    end
    return A
end

# zero out elements not on matrix bands
function bandex(A::AbstractMatrix,kl,ku)
    m, n = size(A)
    AB = band(A,kl,ku)
    B = unband(AB,m,kl,ku)
    return B
end

end
