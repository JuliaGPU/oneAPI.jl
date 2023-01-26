# fence

export ZeFence

mutable struct ZeFence
    handle::ze_fence_handle_t
    queue::ZeCommandQueue

    function ZeFence(queue)
        desc_ref = Ref(ze_fence_desc_t())
        handle_ref = Ref{ze_fence_handle_t}()
        zeFenceCreate(queue, desc_ref, handle_ref)
        obj = new(handle_ref[], queue)
        finalizer(obj) do obj
            zeFenceDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_fence_handle_t}, fence::ZeFence) = fence.handle

Base.:(==)(a::ZeFence, b::ZeFence) = a.handle == b.handle
Base.hash(e::ZeFence, h::UInt) = hash(e.handle, h)

Base.wait(fence::ZeFence, timeout::Number=typemax(UInt64)) =
    zeFenceHostSynchronize(fence, timeout)

Base.reset(fence::ZeFence) = zeFenceReset(fence)

function Base.isdone(fence::ZeFence)
    res = unsafe_zeFenceQueryStatus(fence)
    if res == RESULT_NOT_READY
        return false
    elseif res == RESULT_SUCCESS
        return true
    else
        throw_api_error(res)
    end
end
