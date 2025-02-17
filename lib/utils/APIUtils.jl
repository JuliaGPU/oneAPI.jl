module APIUtils

# helpers that facilitate working with C APIs
using GPUToolbox: @checked, @debug_ccall
export @checked, @debug_ccall
include("enum.jl")

end
