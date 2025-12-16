Base.to_index(::oneArray, I::AbstractArray{Bool}) = findall(I)

if VERSION >= v"1.11.0-DEV.1157"
    Base.to_indices(x::oneArray, I::Tuple{AbstractArray{Bool}}) =
        (Base.to_index(x, I[1]),)
end

function _ker!(ys, bools, indices)
    i = get_global_id()

    @inbounds if i ≤ length(bools) && bools[i]
        ii = CartesianIndices(bools)[i]
        b = indices[i] # new position
        ys[b] = ii
    end
    return
end

function Base.findall(bools::oneArray{Bool})
    I = keytype(bools)

    indices = cumsum(reshape(bools, prod(size(bools))))

    n = isempty(indices) ? 0 : @allowscalar indices[end]

    ys = oneArray{I}(undef, n)

    if n > 0
        kernel = @oneapi launch = false _ker!(ys, bools, indices)
        group_size = launch_configuration(kernel)
        kernel(ys, bools, indices; items = group_size, groups = cld(length(bools), group_size))
    end
    # unsafe_free!(indices)

    return ys
end
