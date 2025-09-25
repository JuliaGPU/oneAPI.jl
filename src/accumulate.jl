Base.accumulate!(op, B::oneArray, A::oneArray; init=zero(eltype(A)), kwargs...) =
    AK.accumulate!(op, B, A, oneAPIBackend(); init, kwargs...)

Base.accumulate(op, A::oneArray; init=zero(eltype(A)), kwargs...) =
    AK.accumulate(op, A, oneAPIBackend(); init, kwargs...)

Base.cumsum(src::oneArray; kwargs...) = AK.cumsum(src, oneAPIBackend(); kwargs...)
Base.cumprod(src::oneArray; kwargs...) = AK.cumprod(src, oneAPIBackend(); kwargs...)