# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct oneArrayStyle{N} <: AbstractGPUArrayStyle{N} end
oneArrayStyle(::Val{N}) where N = oneArrayStyle{N}()
oneArrayStyle{M}(::Val{N}) where {N,M} = oneArrayStyle{N}()

BroadcastStyle(::Type{<:oneArray{T,N}}) where {T,N} = oneArrayStyle{N}()

Base.similar(bc::Broadcasted{oneArrayStyle{N}}, ::Type{T}) where {N,T} =
    similar(oneArray{T}, axes(bc))

Base.similar(bc::Broadcasted{oneArrayStyle{N}}, ::Type{T}, dims...) where {N,T} =
    oneArray{T}(undef, dims...)
