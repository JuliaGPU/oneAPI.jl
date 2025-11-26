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
    execute!(global_queue(ctx, dev)) do list
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
    execute!(global_queue(ctx, dev)) do list
        append_fill!(list, ptr, pattern, sizeof(T), bytes)
    end
end
