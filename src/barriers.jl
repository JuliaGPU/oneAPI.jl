export append_barrier!, device_barrier

append_barrier!(list::ZeCommandList, signal_event=nothing, wait_events::ZeEvent...) =
    zeCommandListAppendBarrier(list, something(signal_event, C_NULL),
                               length(wait_events), [wait_events...])

device_barrier(dev::ZeDevice) = zeDeviceSystemBarrier(dev)
