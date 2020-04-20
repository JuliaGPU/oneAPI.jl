# copies

export append_copy!

append_copy!(list::ZeCommandList, dst::Union{Ptr,ZePtr}, src::Union{Ptr,ZePtr},
             size::Integer, event::Union{ZeEvent,Nothing}=nothing) =
    zeCommandListAppendMemoryCopy(list, dst, src, size, something(event, C_NULL))
