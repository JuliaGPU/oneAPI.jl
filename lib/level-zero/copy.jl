# copies

export append_copy!, append_fill!, append_prefetch!, append_advise!

append_copy!(list::AbstractZeCommandList, dst::Union{Ptr,ZePtr}, src::Union{Ptr,ZePtr},
             size::Integer, signal_event::Union{ZeEvent,Nothing}=nothing,
             wait_events::ZeEvent...) =
    zeCommandListAppendMemoryCopy(list, dst, src, size, something(signal_event, C_NULL),
                                   length(wait_events), [wait_events...])

append_fill!(list::AbstractZeCommandList, ptr::Union{Ptr,ZePtr}, pattern::Union{Ptr,ZePtr},
             pattern_size::Integer, size::Integer,
             signal_event::Union{ZeEvent,Nothing}=nothing, wait_events::ZeEvent...) =
    zeCommandListAppendMemoryFill(list, ptr, pattern, pattern_size, size,
                                  something(signal_event, C_NULL), length(wait_events),
                                  [wait_events...])

append_prefetch!(list::AbstractZeCommandList, ptr::Union{Ptr,ZePtr}, size::Integer) =
    zeCommandListAppendMemoryPrefetch(list, ptr, size)

append_advise!(list::AbstractZeCommandList, dev::ZeDevice, ptr::Union{Ptr,ZePtr}, size::Integer,
               advise::ze_memory_advice_t) =
    zeCommandListAppendMemAdvise(list, dev, ptr, size, advise)
