# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == RESULT_ERROR_OUT_OF_HOST_MEMORY || res == RESULT_ERROR_OUT_OF_DEVICE_MEMORY
        throw(OutOfGPUMemoryError())
    else
        throw(ZeError(res))
    end
end

function check(f)
    res = retry_reclaim(err -> err == RESULT_ERROR_OUT_OF_HOST_MEMORY ||
                               err == RESULT_ERROR_OUT_OF_DEVICE_MEMORY) do
        f()
    end

    if res != RESULT_SUCCESS
        throw_api_error(res)
    end

    return
end
