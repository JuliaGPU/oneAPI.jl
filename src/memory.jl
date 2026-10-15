# memory operations

"""
    Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dst, src, N)

Low-level memory copy operation on the GPU.

Copies `N` elements of type `T` from `src` to `dst` using the specified context and device.
Both `src` and `dst` can be either host pointers (`Ptr`) or device pointers (`ZePtr`).

# Arguments
- `ctx::ZeContext`: Level Zero context
- `dev::ZeDevice`: Level Zero device
- `dst::Union{Ptr{T},ZePtr{T}}`: Destination pointer
- `src::Union{Ptr{T},ZePtr{T}}`: Source pointer
- `N::Integer`: Number of elements to copy

!!! warning
    This is a low-level function. No bounds checking is performed. For safe array copying,
    use `copyto!` on `oneArray` objects instead.

See also: [`copyto!`](@ref), [`oneArray`](@ref)
"""
function Base.unsafe_copyto!(ctx::ZeContext, dev::ZeDevice, dst::Union{Ptr{T},ZePtr{T}},
                             src::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    execute!(global_stream(ctx, dev)) do list
        append_copy!(list, dst, src, bytes)
    end
end

"""
    unsafe_fill!(ctx::ZeContext, dev::ZeDevice, ptr, pattern, N)

Low-level memory fill operation on the GPU.

Fills `N` elements at `ptr` with the given pattern using the specified context and device.

# Arguments
- `ctx::ZeContext`: Level Zero context
- `dev::ZeDevice`: Level Zero device
- `ptr::Union{Ptr{T},ZePtr{T}}`: Pointer to memory to fill
- `pattern::Union{Ptr{T},ZePtr{T}}`: Pointer to pattern value
- `N::Integer`: Number of elements to fill

!!! warning
    This is a low-level function. For safe array operations, use `fill!` on `oneArray`
    objects instead.

See also: [`fill!`](@ref), [`oneArray`](@ref)
"""
function unsafe_fill!(ctx::ZeContext, dev::ZeDevice, ptr::Union{Ptr{T},ZePtr{T}},
                      pattern::Union{Ptr{T},ZePtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    execute!(global_stream(ctx, dev)) do list
        append_fill!(list, ptr, pattern, sizeof(T), bytes)
    end
end


## managed memory

# An allocation together with the submission target that last used it, so that host
# accesses only wait when there may be outstanding work, and work from another task first
# waits for the previous user. Shared by all arrays derived from the same allocation.
mutable struct Managed{B<:oneL0.AbstractBuffer}
    const mem::B
    const lock::ReentrantLock

    # the stream or explicitly created queue that last used the memory
    owner::Union{Nothing, oneStream, ZeCommandQueue}

    # whether the owner may still have work using the memory. atomic so that host
    # accesses can check it without taking the lock.
    Base.@atomic dirty::Bool

    Managed(mem::B, owner=nothing, dirty::Bool=false) where {B} =
        new{B}(mem, ReentrantLock(), owner, dirty)
end

Base.sizeof(managed::Managed) = sizeof(managed.mem)

# wait for all work on a stream, possibly from another task. unlike `synchronize`, this
# leaves `mkl_dirty` alone, which only the stream's own task may modify.
function drain(s::oneStream)
    oneL0.nonblocking_synchronize(s.list)
    q = Base.@atomic s.queue
    q === nothing || oneL0.nonblocking_synchronize(q)
    return
end
drain(q::ZeCommandQueue) = oneL0.nonblocking_synchronize(q)

# the caller must hold `managed.lock`
function synchronize_owner!(managed::Managed)
    if Base.@atomic managed.dirty
        drain(managed.owner)
        Base.@atomic managed.dirty = false
    end
    return
end

# record `owner` as the user of the memory, waiting for the previous one if it differs.
# the caller must hold `managed.lock` until the operation has been submitted to `owner`,
# so that other tasks never observe the ownership without the work it describes.
function take_ownership!(managed::Managed, owner::Union{oneStream, ZeCommandQueue})
    sizeof(managed) == 0 && return
    if managed.owner !== owner
        synchronize_owner!(managed)
        managed.owner = owner
    end
    Base.@atomic managed.dirty = true
    return
end

# wait for outstanding work before accessing the memory from the host
function maybe_synchronize(managed::Managed)
    Base.@atomic(managed.dirty) || return
    @lock managed.lock synchronize_owner!(managed)
end

# lock the given allocations in a stable order, avoiding deadlocks between tasks that
# submit operations involving the same allocations
function with_managed_locks(f, managed::Vector{Managed})
    locked = sort!(unique!(managed); by=objectid)
    nlocked = 0
    try
        for m in locked
            lock(m.lock)
            nlocked += 1
        end
        f()
    finally
        for i in nlocked:-1:1
            unlock(locked[i].lock)
        end
    end
end

# submit an operation using the memory of `managed` to `owner`, as a single transaction
function with_ownership(f, owner, managed::Vector{Managed})
    with_managed_locks(managed) do
        for m in managed
            take_ownership!(m, owner)
        end
        f()
    end
end
