import KernelInterface
using oneAPIKernels

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(oneAPIBackend, "oneAPI", oneAPI, oneArray, oneAPI.oneDeviceArray)
