# outlined functionality to avoid GC frame allocation
@noinline throw_api_error(res) = throw(PTIError(res))

function check(f)
    res = f()
    if res != PTI_SUCCESS
        throw_api_error(res)
    end
    return
end
