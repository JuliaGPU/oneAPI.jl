module KernelAbstractionsExt

using oneAPI

import KernelAbstractions as KA

import Adapt

Adapt.adapt_storage(::KA.CPU, a::oneArray) = convert(Array, a)

# sparse arrays (oneMKL is only available on Linux)
@static if Sys.islinux()
    import SparseArrays
    Adapt.adapt_storage(::KA.CPU, a::oneAPI.oneMKL.oneAbstractSparseMatrix) = SparseArrays.SparseMatrixCSC(a)
end


## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::oneDeviceArray) = Base.Experimental.Const(a)

end
