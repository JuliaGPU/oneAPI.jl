Base.sort!(x::oneArray; kwargs...) = (AK.sort!(x; kwargs...); return x)
Base.sortperm!(ix::oneArray, x::oneArray; kwargs...) = (AK.sortperm!(ix, x; kwargs...); return ix)
Base.sortperm(x::oneArray; kwargs...) = sortperm!(oneArray(1:length(x)), x; kwargs...)
