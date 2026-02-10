import oneAPI
import oneAPI: oneArray, oneAPIBackend
import AcceleratedKernels as AK

# Use a smaller block size on Intel GPUs to work around a scan correctness issue
# with the Blelloch parallel prefix sum at larger block sizes (>=128).
const _ACCUMULATE_BLOCK_SIZE = 64

# Accumulate operations using AcceleratedKernels
Base.accumulate!(op, B::oneArray, A::oneArray; init = zero(eltype(A)),
                 block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.accumulate!(op, B, A, oneAPIBackend(); init, block_size, kwargs...)

Base.accumulate(op, A::oneArray; init = zero(eltype(A)),
                block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.accumulate(op, A, oneAPIBackend(); init, block_size, kwargs...)

Base.cumsum(src::oneArray; block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.cumsum(src, oneAPIBackend(); block_size, kwargs...)
Base.cumprod(src::oneArray; block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.cumprod(src, oneAPIBackend(); block_size, kwargs...)
