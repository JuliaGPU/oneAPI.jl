# memory operations

function Base.unsafe_copyto!(dev::ZeDevice, dst::Union{Ptr{T},ZePtr{T}},
                             src::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    execute!(global_queue(dev)) do list
        append_copy!(list, dst, src, N*sizeof(T))
    end
end

function unsafe_fill!(dev::ZeDevice, ptr::Union{Ptr{T},ZePtr{T}},
                      pattern::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    execute!(global_queue(dev)) do list
        append_fill!(list, ptr, pattern, sizeof(T), N*sizeof(T))
    end
end
