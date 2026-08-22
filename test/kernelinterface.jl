import KernelInterface
using oneAPI.oneAPIInterface

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

skip_tests = Set{String}()
# the events test checks that a waiting task blocks on work queued elsewhere, which is not
# observable when every submission synchronizes (the Aurora LTS workaround)
oneAPI.oneL0.sync_each_submission() && push!(skip_tests, "Events")

Testsuite.testsuite(oneAPIInterface.oneAPIBackend, "oneAPI", oneAPI, oneArray, oneAPI.oneDeviceArray; skip_tests)
