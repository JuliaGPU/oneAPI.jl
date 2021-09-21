# memory operations

function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dst::Union{Ptr{T},ZePtr{T}},
                             src::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    execute!(global_queue(ctx, dev)) do list
        append_copy!(list, dst, src, bytes)
    end
end

function unsafe_fill!(ctx::ZeContext, dev::ZeDevice, ptr::Union{Ptr{T},ZePtr{T}},
                      pattern::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    execute!(global_queue(ctx, dev)) do list
        append_fill!(list, ptr, pattern, sizeof(T), bytes)
    end
end
