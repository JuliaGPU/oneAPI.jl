import oneAPI
import oneAPI: oneArray, oneAPIBackend
import AcceleratedKernels as AK

# Use a smaller block size on Intel GPUs to work around a scan correctness issue
# with the parallel prefix sum at larger block sizes (>=128).
const _ACCUMULATE_BLOCK_SIZE = 64

# The scan algorithm for the given `dims`: whole-array scans and scans along a dimension
# use different algorithms, so pick the one AcceleratedKernels' `Auto()` would, with our
# block size.
_scan_alg(dims) = dims === nothing ? AK.ScanPrefixes(block_size = _ACCUMULATE_BLOCK_SIZE) :
    AK.SliceScan(block_size = _ACCUMULATE_BLOCK_SIZE)

# Accumulate operations using AcceleratedKernels
Base.accumulate!(op, B::oneArray, A::oneArray; dims = nothing, alg = _scan_alg(dims), kwargs...) =
    AK.accumulate!(op, B, A; dims, alg, kwargs...)

Base.accumulate(op, A::oneArray; dims = nothing, alg = _scan_alg(dims), kwargs...) =
    AK.accumulate(op, A; dims, alg, kwargs...)

Base.cumsum(src::oneArray; dims = nothing, alg = _scan_alg(dims), kwargs...) =
    AK.cumsum(src; dims, alg, kwargs...)
Base.cumprod(src::oneArray; dims = nothing, alg = _scan_alg(dims), kwargs...) =
    AK.cumprod(src; dims, alg, kwargs...)
