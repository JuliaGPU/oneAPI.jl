## COV_EXCL_START

# TODO
# - serial version for lower latency
# - group-stride loop to delay need for second kernel launch

# Widen sub-word types to avoid shared memory corruption on Intel GPUs.
# Writing 1/2-byte values to local memory can clobber adjacent bytes.
# Only applies to integer/boolean types where `%` conversion is valid.
@inline _widen_type(::Type{Bool}) = Int32
@inline _widen_type(::Type{Int8}) = Int32
@inline _widen_type(::Type{UInt8}) = Int32
@inline _widen_type(::Type{Int16}) = Int32
@inline _widen_type(::Type{UInt16}) = Int32
@inline _widen_type(::Type{T}) where T = T

# Dispatch-based conversions so the compiler never generates `%` for non-integer types
@inline _to_wide(val, ::Type{W}) where W = val % W
@inline _to_wide(val::T, ::Type{T}) where T = val
@inline _from_wide(val, ::Type{T}) where T = val % T
@inline _from_wide(val::T, ::Type{T}) where T = val

# Reduce a value across a group, using local memory for communication
@inline function reduce_group(op, val::T, neutral, ::Val{maxitems}) where {T, maxitems}
    items = get_local_size()
    item = get_local_id()

    # use a wider type for shared memory to avoid sub-word corruption
    W = _widen_type(T)
    shared = oneLocalArray(W, (maxitems,))
    @inbounds shared[item] = _to_wide(val, W)

    # perform a reduction
    d = 1
    while d < items
        barrier(0)
        index = 2 * d * (item-1) + 1
        @inbounds if index <= items
            other_val = if index + d <= items
                _from_wide(shared[index+d], T)
            else
                neutral
            end
            shared[index] = _to_wide(op(_from_wide(shared[index], T), other_val), W)
        end
        d *= 2
    end

    # load the final value on the first item
    if item == 1
        val = @inbounds _from_wide(shared[item], T)
    end

    return val
end

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
function partial_mapreduce_device(f, op, neutral, maxitems, Rreduce, Rother, R, As...)
    # decompose the 1D hardware indices into separate ones for reduction (across items
    # and possibly groups if it doesn't fit) and other elements (remaining groups)
    localIdx_reduce = get_local_id()
    localDim_reduce = get_local_size()
    groupIdx_reduce, groupIdx_other = fldmod1(get_group_id(), length(Rother))
    groupDim_reduce = get_num_groups() ÷ length(Rother)

    # group-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize items within this group)
    iother = groupIdx_other
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., groupIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a group
        ireduce = localIdx_reduce + (groupIdx_reduce - 1) * localDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(_map_getindex(As, J)...))
            ireduce += localDim_reduce * groupDim_reduce
        end

        val = reduce_group(op, val, neutral, maxitems)

        # write back to memory
        if localIdx_reduce == 1
            R[Iout] = val
        end
    end

    return
end

## COV_EXCL_STOP

function GPUArrays.mapreducedim!(f::F, op::OP, R::oneWrappedArray{T},
                                 A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing) where {F, OP, T}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)

    # allocate an additional, empty dimension to write the reduced value to.
    # this does not affect the actual location in memory of the final values,
    # but allows us to write a generalized kernel supporting partial reductions.
    R′ = reshape(R, (size(R)..., 1))

    # how many items do we want?
    #
    # items in a group work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.
    wanted_items = length(Rreduce)
    function compute_items(max_items)
        if wanted_items > max_items
            max_items
        else
            wanted_items
        end
    end

    # how many items can we launch?
    #
    # we might not be able to launch all those items to reduce each slice in one go.
    # that's why each items also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single item group.

    # group size is restricted by local memory (use widened type for sub-word types)
    max_lmem_elements = compute_properties(device()).maxSharedLocalMemory ÷ sizeof(_widen_type(T))
    max_items = min(compute_properties(device()).maxTotalGroupSize,
                    compute_items(max_lmem_elements ÷ 2))
    # TODO: dynamic local memory to avoid two compilations

    # let the driver suggest a group size
    args = (f, op, init, Val(max_items), Rreduce, Rother, R′, A)
    kernel_args = kernel_convert.(args)
    kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    kernel = zefunction(partial_mapreduce_device, kernel_tt)
    reduce_items = compute_items(launch_configuration(kernel))

    # how many groups should we launch?
    #
    # even though we can always reduce each slice in a single item group, that may not be
    # optimal as it might not saturate the GPU. we already launch some groups to process
    # independent dimensions in parallel; pad that number to ensure full occupancy.
    other_groups = length(Rother)
    reduce_groups = cld(length(Rreduce), reduce_items)

    # determine the launch configuration
    items = reduce_items
    groups = reduce_groups*other_groups

    # perform the actual reduction
    if reduce_groups == 1
        # we can cover the dimensions to reduce using a single group
        @oneapi items groups partial_mapreduce_device(
            f, op, init, Val(items), Rreduce, Rother, R′, A)
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_groups))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            partial .= R
        end
        @oneapi items groups partial_mapreduce_device(
            f, op, init, Val(items), Rreduce, Rother, partial, A)

        GPUArrays.mapreducedim!(identity, op, R′, partial; init=init)
    end

    return R
end
