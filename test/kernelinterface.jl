import KernelInterface
using oneAPI.oneAPIInterface

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(oneAPIInterface.oneAPIBackend(), oneArray)
