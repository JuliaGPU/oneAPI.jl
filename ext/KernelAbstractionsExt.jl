module KernelAbstractionsExt

using oneAPI
using oneAPI: @device_override, method_table

import KernelAbstractions as KA

import Adapt

Adapt.adapt_storage(::KA.CPU, a::oneArray) = convert(Array, a)

# sparse arrays (oneMKL is only available on Linux)
@static if Sys.islinux()
    import SparseArrays
    Adapt.adapt_storage(::KA.CPU, a::oneAPI.oneMKL.oneAbstractSparseMatrix) = SparseArrays.SparseMatrixCSC(a)
end


## scratch memory

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    KA.PrivateArray{T}(undef, Val(Dims), Val(oneAPI.AS.Function))
end


## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::oneDeviceArray) = Base.Experimental.Const(a)

end
