# memory operations

function Base.unsafe_copyto!(dev::ZeDevice, dst::Union{Ptr{T},ZePtr{T}},
                             src::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    execute!(global_queue(dev)) do list
        append_copy!(list, dst, src, N*sizeof(T))
    end

    # memory copies are synchronizing
    # TODO: this is costly; figure out a better programming model
    synchronize(global_queue(device()))
end

function unsafe_fill!(dev::ZeDevice, ptr::Union{Ptr{T},ZePtr{T}},
                      pattern::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    execute!(global_queue(dev)) do list
        append_fill!(list, ptr, pattern, sizeof(T), N*sizeof(T))
    end
end
