import KernelAbstractions
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
    "Random",
    "CPU synchronization",
    "fallback test: callable types"
])
Testsuite.testsuite(oneAPIBackend, "oneAPI", oneAPI, oneArray, oneDeviceArray; skip_tests)
