#
# Auxiliary
#

function Base.convert(::Type{onemklSide}, side::Char)
    if side == 'L'
        return ONEMKL_SIDE_LEFT
    elseif side == 'R'
        return ONEMKL_SIDE_RIGHT
    else
        throw(ArgumentError("Unknown transpose $side"))
    end
end

function Base.convert(::Type{onemklTranspose}, trans::Char)
    if trans == 'N'
        return ONEMKL_TRANSPOSE_NONTRANS
    elseif trans == 'T'
        return ONEMKL_TRANSPOSE_TRANS
    elseif trans == 'C'
        return ONEMLK_TRANSPOSE_CONJTRANS
    else
        throw(ArgumentError("Unknown transpose $trans"))
    end
end

function Base.convert(::Type{onemklUplo}, uplo::Char)
    if uplo == 'U'
        return ONEMKL_UPLO_UPPER
    elseif uplo == 'L'
        return ONEMKL_UPLO_LOWER
    else
        throw(ArgumentError("Unknown transpose $uplo"))
    end
end

function Base.convert(::Type{onemklDiag}, diag::Char)
    if diag == 'N'
        return ONEMKL_DIAG_NONUNIT
    elseif diag == 'U'
        return ONEMKL_DIAG_UNIT
    else
        throw(ArgumentError("Unknown transpose $diag"))
    end
end

function Base.convert(::Type{onemklIndex}, index::Char)
    if index == 'O'
        return ONEMKL_INDEX_ONE
    elseif index == 'Z'
        return ONEMKL_INDEX_ZERO
    else
        throw(ArgumentError("Unknown index $index"))
    end
end

function Base.convert(::Type{onemklLayout}, index::Char)
    if index == 'R'
        return ONEMKL_LAYOUT_ROW
    elseif index == 'C'
        return ONEMKL_LAYOUT_COL
    else
        throw(ArgumentError("Unknown layout $layout"))
    end
end

# create a batch of pointers in device memory from a batch of device arrays
@inline function unsafe_batch(batch::Vector{<:oneArray{T}}) where {T}
    ptrs = pointer.(batch)
    return oneArray(ptrs)
end
