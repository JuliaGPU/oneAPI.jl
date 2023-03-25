import KernelAbstractions
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
])
Testsuite.testsuite(oneAPIBackend, "oneAPI", oneAPI, oneArray, oneDeviceArray; skip_tests)
