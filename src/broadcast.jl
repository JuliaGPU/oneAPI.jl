import Base.Broadcast: BroadcastStyle, Broadcasted

struct oneArrayStyle{N,B} <: AbstractGPUArrayStyle{N} end
oneArrayStyle{M,B}(::Val{N}) where {N,M,B} = oneArrayStyle{N,B}()

# identify the broadcast style of a (wrapped) oneArray
BroadcastStyle(::Type{<:oneArray{T, N, B}}) where {T, N, B} = oneArrayStyle{N, B}()
BroadcastStyle(W::Type{<:oneWrappedArray{T, N}}) where {T, N} =
    oneArrayStyle{N, buftype(Adapt.unwrap_type(W))}()

# when we are dealing with different buffer styles, we cannot know
# which one is better, so use shared memory
BroadcastStyle(
    ::oneArrayStyle{N, B1},
               ::oneArrayStyle{N, B2}) where {N,B1,B2} =
    oneArrayStyle{N, oneL0.SharedBuffer}()

# allocation of output arrays
Base.similar(bc::Broadcasted{oneArrayStyle{N,B}}, ::Type{T}, dims) where {T,N,B} =
    similar(oneArray{T,length(dims),B}, dims)
