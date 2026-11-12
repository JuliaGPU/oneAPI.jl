# kernel arguments that are too large to pass by value
#
# Level Zero limits the total size of a kernel's arguments to the device's
# `maxArgumentsSize`, 2048 bytes on current Intel GPUs. On PVC that is a hardware limit, as
# the arguments are loaded into a block of registers when a thread starts, and IGC refuses
# to build a module with a kernel that exceeds it. Julia kernels pass their arguments by
# value, though, including large isbits structs like the expression trees of ExaModels.
#
# A kernel whose arguments don't fit is therefore compiled as a wrapper, `SpilledKernel`,
# that takes the largest of them by reference instead: through a device pointer to a copy
# of the value in device memory, which the wrapper loads before calling the kernel with
# the value. The kernel's code is the same either way; kernels whose arguments fit are
# compiled and launched as before. Fields the kernel accesses statically are loaded from
# the copy directly; where it indexes the value dynamically, or passes it to a function
# that isn't inlined, the loaded value is kept in private memory, which counts against
# the scratch space a kernel can use.
#
# IGC can also fail to fit arguments that are within the limit into registers, so kernels
# with large arguments whose build fails are built again with them passed through memory.
#
# The copies are cached by content (see `argument_buffer`), as kernels are typically
# launched over and over with the same large argument, and a copy to device memory waits
# for the stream on discrete GPUs.

# The arguments `zefunction` passes through memory, by position in (f, args...).
struct ArgumentSpill
    mask::Vector{Bool}
end

# The type of the kernel argument that Level Zero receives for a Julia argument of type
# `T`, or `nothing` if it isn't passed (see `call_arguments`).
function kernel_argument_type(@nospecialize(T))
    (isghosttype(T) || Core.Compiler.isconstType(T)) && return nothing
    return isbitstype(T) ? T : Ptr{Any}
end

# The size of the arguments of a kernel taking arguments of the given types, after the
# hidden kernel state. Each argument is aligned to its alignment, as IGC lays them out.
function kernel_arguments_size(types)
    size = sizeof(KernelState)
    for T in types
        T === nothing && continue
        align = Base.datatype_alignment(T)
        size = cld(size, align) * align + sizeof(T)
    end
    return size
end

# A short name for error messages, the full type of a kernel can be huge
function kernel_name(@nospecialize(F))
    name = isdefined(F, :instance) ? nameof(F.instance) : nameof(F)
    return string(name)
end

# Determine which arguments of a kernel with function type `F` and argument types `TT` to
# pass through memory, if their total size exceeds `limit`: as few as possible, the largest
# first. Returns `nothing` if all arguments can be passed by value. Throws if they don't fit
# even then, or with `strict=false` returns `nothing`.
function argument_spill(@nospecialize(F), @nospecialize(TT), limit::Int; strict::Bool=true)
    types = Any[F, TT.parameters...]
    args = Any[kernel_argument_type(T) for T in types]
    size = kernel_arguments_size(args)
    # (a driver that doesn't report a limit leaves it to the compiler)
    (limit <= 0 || size <= limit) && return nothing

    mask = Bool[false for _ in types]
    candidates = filter(i -> args[i] !== nothing && isbitstype(args[i]) &&
                             sizeof(args[i]) > sizeof(Ptr{Cvoid}), eachindex(types))
    sort!(candidates; by = i -> sizeof(args[i]), rev = true)
    for i in candidates
        mask[i] = true
        args[i] = LLVMPtr{types[i], AS.CrossWorkgroup}
        kernel_arguments_size(args) <= limit && return ArgumentSpill(mask)
    end
    strict || return nothing

    sizes = join(("$(i == 1 ? "function" : "argument $(i-1)") ($(types[i])): $(sizeof(args[i])) bytes"
                  for i in eachindex(types) if args[i] !== nothing && sizeof(args[i]) > 0),
                 "\n  ")
    throw(ArgumentError("""The arguments of kernel `$(kernel_name(F))` take $size bytes, which \
                           exceeds the device's limit of $limit bytes, also when passing the \
                           ones larger than a pointer through memory. Sizes after doing so \
                           (not counting the $(sizeof(KernelState))-byte kernel state):
                             $sizes"""))
end

# The function and argument types that a kernel is compiled for when passing arguments
# through memory as `spill` determines.
function spilled_signature(@nospecialize(F), @nospecialize(TT), spill::ArgumentSpill)
    types = Any[F, TT.parameters...]
    for i in eachindex(types)
        spill.mask[i] && (types[i] = LLVMPtr{types[i], AS.CrossWorkgroup})
    end
    return SpilledKernel{types[1], Tuple(spill.mask)}, Tuple{types[2:end]...}
end


## device side

# A kernel `f` with the arguments for which `Mask` is true (the first entry being for `f`
# itself) passed as pointers to their value.
struct SpilledKernel{F, Mask}
    f::F
end

# the alignment of the copies of spilled arguments (see `argument_buffer`)
const SPILL_ALIGNMENT = 64

@generated function (kernel::SpilledKernel{F, Mask})(args::Vararg{Any, N}) where {F, Mask, N}
    load(ex) = :(unsafe_load($ex, 1, Val($SPILL_ALIGNMENT)))
    f = Mask[1] ? load(:(kernel.f)) : :(kernel.f)
    call_args = Any[Mask[i+1] ? load(:(args[$i])) : :(args[$i]) for i in 1:N]
    quote
        Base.@_inline_meta
        f = $f
        @inline f($(call_args...))
        return
    end
end


## host side

# Device copies of spilled arguments, by content. Never written to after the upload, so
# that a launch can use the copy while another launch with the same value is in flight.
struct ArgumentBuffer
    hash::UInt
    bytes::Vector{UInt8}
    buf::oneArray{UInt8, 1, oneL0.DeviceBuffer}
end

# per context and device, least recently used first
const argument_buffers = Dict{Tuple{ZeContext, ZeDevice}, Vector{ArgumentBuffer}}()
const argument_buffers_lock = ReentrantLock()

# how many copies to keep around, and how much memory they may take
const MAX_ARGUMENT_BUFFERS = 256
const MAX_ARGUMENT_BUFFER_BYTES = 64 * 2^20

# A device copy of the value `x`, from the cache or uploaded.
function argument_buffer(ctx::ZeContext, dev::ZeDevice, x::T) where {T}
    bytes = Base.zeros(UInt8, sizeof(T))
    GC.@preserve bytes unsafe_store!(convert(Ptr{T}, pointer(bytes)), x)
    h = hash(bytes)

    @lock argument_buffers_lock begin
        cache = get!(Vector{ArgumentBuffer}, argument_buffers, (ctx, dev))
        for i in length(cache):-1:1
            entry = cache[i]
            if entry.hash == h && entry.bytes == bytes
                deleteat!(cache, i)
                push!(cache, entry)
                return entry.buf
            end
        end
    end

    # upload outside of the lock, as this waits for the stream on discrete GPUs
    data = DataRef(release, alloc_managed(oneL0.DeviceBuffer, ctx, dev, length(bytes),
                                          SPILL_ALIGNMENT))
    buf = oneArray{UInt8, 1}(data, (length(bytes),))
    unsafe_free!(data)
    unsafe_copyto!(ctx, dev, buf, 1, bytes, 1, length(bytes))

    evicted = @lock argument_buffers_lock begin
        cache = argument_buffers[(ctx, dev)]
        push!(cache, ArgumentBuffer(h, bytes, buf))
        total = sum(entry -> length(entry.bytes), cache)
        evicted = ArgumentBuffer[]
        while length(cache) > 1 && (length(cache) > MAX_ARGUMENT_BUFFERS ||
                                    total > MAX_ARGUMENT_BUFFER_BYTES)
            entry = popfirst!(cache)
            total -= length(entry.bytes)
            push!(evicted, entry)
        end
        evicted
    end
    # (freeing memory can wait for the GPU, so not while holding the lock.) launches that
    # still use the memory keep it alive, as they do for any array.
    foreach(entry -> unsafe_free!(entry.buf), evicted)

    return buf
end

# `launch_managed` for a kernel that passes arguments through memory: replace those by
# pointers to their device copy, and launch the wrapper the kernel was compiled as.
# Not specialized, as the argument types depend on which arguments are spilled.
@noinline function launch_spilled(kernel, source, args::Tuple, converted::Tuple,
                                  refs::Vector{DataRef}, queue, kwargs::NamedTuple)
    mask = kernel.spill.mask
    ctx = kernel.fun.mod.context
    dev = kernel.fun.mod.device
    vals = Any[kernel.f, converted...]
    buffers = Any[]
    for i in eachindex(vals)
        mask[i] || continue
        buf = argument_buffer(ctx, dev, vals[i])
        push!(buffers, buf)
        # the launch takes ownership of the copy as it does of the arrays it uses
        push!(refs, buf.data)
        vals[i] = reinterpret(LLVMPtr{typeof(vals[i]), AS.CrossWorkgroup}, pointer(buf))
    end
    f = SpilledKernel{Core.Typeof(vals[1]), Tuple(mask)}(vals[1])
    wrapper_args = Tuple(vals[2:end])
    wrapper = HostKernel{typeof(f), Tuple{map(Core.Typeof, wrapper_args)...}}(f, kernel.fun)

    managed = Managed[ref[] for ref in refs]
    ret = GC.@preserve source args buffers with_ownership(queue, managed) do
        call(wrapper, wrapper_args; queue, roots=refs, kwargs...)
    end
    release_pending!()
    return ret
end
