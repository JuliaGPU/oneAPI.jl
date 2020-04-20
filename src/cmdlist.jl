# list

export ZeCommandList, execute!

mutable struct ZeCommandList
    handle::ze_command_list_handle_t

    function ZeCommandList(dev::ZeDevice)
        desc_ref = Ref(ze_command_list_desc_t(
            ZE_COMMAND_LIST_DESC_VERSION_CURRENT,
            ZE_COMMAND_LIST_FLAG_NONE,
        ))
        handle_ref = Ref{ze_command_list_handle_t}()
        zeCommandListCreate(dev, desc_ref, handle_ref)
        obj = new(handle_ref[])
        finalizer(obj) do obj
            zeCommandListDestroy(obj)
        end
        obj
    end
end

Base.unsafe_convert(::Type{ze_command_list_handle_t}, list::ZeCommandList) = list.handle

Base.close(list::ZeCommandList) = zeCommandListClose(list)

execute!(queue::ZeCommandQueue, lists::Vector{ZeCommandList}, fence=nothing) =
    zeCommandQueueExecuteCommandLists(queue, length(lists), lists, something(fence, C_NULL))

Base.reset(list::ZeCommandList) = zeCommandListReset(list)
