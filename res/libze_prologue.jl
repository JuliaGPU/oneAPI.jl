# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(ZeError(res))
end

macro check(ex)
    is_oom = :(isequal(res, RESULT_ERROR_OUT_OF_HOST_MEMORY) ||
               isequal(res, RESULT_ERROR_OUT_OF_DEVICE_MEMORY))

    quote
        res = retry_reclaim(err -> $is_oom) do
            $(esc(ex))
        end

        if $is_oom
            throw(OutOfMemoryError())
        elseif res != RESULT_SUCCESS
            throw_api_error(res)
        end

        return
    end
end
