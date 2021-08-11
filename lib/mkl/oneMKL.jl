module oneMKL

using CxxWrap

using ..oneAPI
using ..oneAPI.oneL0

@wrapmodule(joinpath(@__DIR__, "../../deps/liboneapilib.so"), :define_module_mkl)

# XXX: expose ZePtr and the automatic conversions from oneArray to CxxWrap
#      so that we don't need to explicitly pointer, reinterpret or @preserve.
raw_pointer(A::oneArray{T}) where {T} = reinterpret(Ptr{T}, pointer(A))

for fun in [:oneapiHgemm, :oneapiSgemm, :oneapiDgemm, :oneapiCgemm, :oneapiZgemm]
    @eval function $fun(queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        GC.@preserve A B C begin
            oneapiSgemm(queue, transA, transB, m, n, k, alpha, raw_pointer(A), lda,
                        raw_pointer(B), ldb, beta, raw_pointer(C), ldc)
        end
    end
end

function __init__()
  @initcxx
end

end
