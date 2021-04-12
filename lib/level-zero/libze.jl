# Julia wrapper for header: ze_api.h
# Automatically generated using Clang.jl

@checked function zeInit(flags)
    ccall((:zeInit, libze_loader), ze_result_t,
          (ze_init_flags_t,),
          flags)
end

@checked function zeDriverGet(pCount, phDrivers)
    ccall((:zeDriverGet, libze_loader), ze_result_t,
          (Ptr{UInt32}, Ptr{ze_driver_handle_t}),
          pCount, phDrivers)
end

@checked function zeDriverGetApiVersion(hDriver, version)
    ccall((:zeDriverGetApiVersion, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_api_version_t}),
          hDriver, version)
end

@checked function zeDriverGetProperties(hDriver, pDriverProperties)
    ccall((:zeDriverGetProperties, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_driver_properties_t}),
          hDriver, pDriverProperties)
end

@checked function zeDriverGetIpcProperties(hDriver, pIpcProperties)
    ccall((:zeDriverGetIpcProperties, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_driver_ipc_properties_t}),
          hDriver, pIpcProperties)
end

@checked function zeDriverGetExtensionProperties(hDriver, pCount, pExtensionProperties)
    ccall((:zeDriverGetExtensionProperties, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{UInt32}, Ptr{ze_driver_extension_properties_t}),
          hDriver, pCount, pExtensionProperties)
end

@checked function zeDriverGetExtensionFunctionAddress(hDriver, name, ppFunctionAddress)
    ccall((:zeDriverGetExtensionFunctionAddress, libze_loader), ze_result_t,
          (ze_driver_handle_t, Cstring, Ptr{Ptr{Cvoid}}),
          hDriver, name, ppFunctionAddress)
end

@checked function zeDeviceGet(hDriver, pCount, phDevices)
    ccall((:zeDeviceGet, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{UInt32}, Ptr{ze_device_handle_t}),
          hDriver, pCount, phDevices)
end

@checked function zeDeviceGetSubDevices(hDevice, pCount, phSubdevices)
    ccall((:zeDeviceGetSubDevices, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{UInt32}, Ptr{ze_device_handle_t}),
          hDevice, pCount, phSubdevices)
end

@checked function zeDeviceGetProperties(hDevice, pDeviceProperties)
    ccall((:zeDeviceGetProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_properties_t}),
          hDevice, pDeviceProperties)
end

@checked function zeDeviceGetComputeProperties(hDevice, pComputeProperties)
    ccall((:zeDeviceGetComputeProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_compute_properties_t}),
          hDevice, pComputeProperties)
end

@checked function zeDeviceGetModuleProperties(hDevice, pModuleProperties)
    ccall((:zeDeviceGetModuleProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_module_properties_t}),
          hDevice, pModuleProperties)
end

@checked function zeDeviceGetCommandQueueGroupProperties(hDevice, pCount,
                                                         pCommandQueueGroupProperties)
    ccall((:zeDeviceGetCommandQueueGroupProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{UInt32}, Ptr{ze_command_queue_group_properties_t}),
          hDevice, pCount, pCommandQueueGroupProperties)
end

@checked function zeDeviceGetMemoryProperties(hDevice, pCount, pMemProperties)
    ccall((:zeDeviceGetMemoryProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{UInt32}, Ptr{ze_device_memory_properties_t}),
          hDevice, pCount, pMemProperties)
end

@checked function zeDeviceGetMemoryAccessProperties(hDevice, pMemAccessProperties)
    ccall((:zeDeviceGetMemoryAccessProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_memory_access_properties_t}),
          hDevice, pMemAccessProperties)
end

@checked function zeDeviceGetCacheProperties(hDevice, pCount, pCacheProperties)
    ccall((:zeDeviceGetCacheProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{UInt32}, Ptr{ze_device_cache_properties_t}),
          hDevice, pCount, pCacheProperties)
end

@checked function zeDeviceGetImageProperties(hDevice, pImageProperties)
    ccall((:zeDeviceGetImageProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_image_properties_t}),
          hDevice, pImageProperties)
end

@checked function zeDeviceGetExternalMemoryProperties(hDevice, pExternalMemoryProperties)
    ccall((:zeDeviceGetExternalMemoryProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_external_memory_properties_t}),
          hDevice, pExternalMemoryProperties)
end

@checked function zeDeviceGetP2PProperties(hDevice, hPeerDevice, pP2PProperties)
    ccall((:zeDeviceGetP2PProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, ze_device_handle_t, Ptr{ze_device_p2p_properties_t}),
          hDevice, hPeerDevice, pP2PProperties)
end

@checked function zeDeviceCanAccessPeer(hDevice, hPeerDevice, value)
    ccall((:zeDeviceCanAccessPeer, libze_loader), ze_result_t,
          (ze_device_handle_t, ze_device_handle_t, Ptr{ze_bool_t}),
          hDevice, hPeerDevice, value)
end

@checked function zeDeviceGetStatus(hDevice)
    ccall((:zeDeviceGetStatus, libze_loader), ze_result_t,
          (ze_device_handle_t,),
          hDevice)
end

@checked function zeDeviceGetGlobalTimestamps(hDevice, hostTimestamp, deviceTimestamp)
    ccall((:zeDeviceGetGlobalTimestamps, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{UInt64}, Ptr{UInt64}),
          hDevice, hostTimestamp, deviceTimestamp)
end

@checked function zeContextCreate(hDriver, desc, phContext)
    ccall((:zeContextCreate, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_context_desc_t}, Ptr{ze_context_handle_t}),
          hDriver, desc, phContext)
end

@checked function zeContextCreateEx(hDriver, desc, numDevices, phDevices, phContext)
    ccall((:zeContextCreateEx, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_context_desc_t}, UInt32, Ptr{ze_device_handle_t},
           Ptr{ze_context_handle_t}),
          hDriver, desc, numDevices, phDevices, phContext)
end

@checked function zeContextDestroy(hContext)
    ccall((:zeContextDestroy, libze_loader), ze_result_t,
          (ze_context_handle_t,),
          hContext)
end

@checked function zeContextGetStatus(hContext)
    ccall((:zeContextGetStatus, libze_loader), ze_result_t,
          (ze_context_handle_t,),
          hContext)
end

@checked function zeCommandQueueCreate(hContext, hDevice, desc, phCommandQueue)
    ccall((:zeCommandQueueCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_command_queue_desc_t},
           Ptr{ze_command_queue_handle_t}),
          hContext, hDevice, desc, phCommandQueue)
end

@checked function zeCommandQueueDestroy(hCommandQueue)
    ccall((:zeCommandQueueDestroy, libze_loader), ze_result_t,
          (ze_command_queue_handle_t,),
          hCommandQueue)
end

@checked function zeCommandQueueExecuteCommandLists(hCommandQueue, numCommandLists,
                                                    phCommandLists, hFence)
    ccall((:zeCommandQueueExecuteCommandLists, libze_loader), ze_result_t,
          (ze_command_queue_handle_t, UInt32, Ptr{ze_command_list_handle_t},
           ze_fence_handle_t),
          hCommandQueue, numCommandLists, phCommandLists, hFence)
end

@checked function zeCommandQueueSynchronize(hCommandQueue, timeout)
    ccall((:zeCommandQueueSynchronize, libze_loader), ze_result_t,
          (ze_command_queue_handle_t, UInt64),
          hCommandQueue, timeout)
end

@checked function zeCommandListCreate(hContext, hDevice, desc, phCommandList)
    ccall((:zeCommandListCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_command_list_desc_t},
           Ptr{ze_command_list_handle_t}),
          hContext, hDevice, desc, phCommandList)
end

@checked function zeCommandListCreateImmediate(hContext, hDevice, altdesc, phCommandList)
    ccall((:zeCommandListCreateImmediate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_command_queue_desc_t},
           Ptr{ze_command_list_handle_t}),
          hContext, hDevice, altdesc, phCommandList)
end

@checked function zeCommandListDestroy(hCommandList)
    ccall((:zeCommandListDestroy, libze_loader), ze_result_t,
          (ze_command_list_handle_t,),
          hCommandList)
end

@checked function zeCommandListClose(hCommandList)
    ccall((:zeCommandListClose, libze_loader), ze_result_t,
          (ze_command_list_handle_t,),
          hCommandList)
end

@checked function zeCommandListReset(hCommandList)
    ccall((:zeCommandListReset, libze_loader), ze_result_t,
          (ze_command_list_handle_t,),
          hCommandList)
end

@checked function zeCommandListAppendWriteGlobalTimestamp(hCommandList, dstptr,
                                                          hSignalEvent, numWaitEvents,
                                                          phWaitEvents)
    ccall((:zeCommandListAppendWriteGlobalTimestamp, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{UInt64}, ze_event_handle_t, UInt32,
           Ptr{ze_event_handle_t}),
          hCommandList, dstptr, hSignalEvent, numWaitEvents, phWaitEvents)
end

@checked function zeCommandListAppendBarrier(hCommandList, hSignalEvent, numWaitEvents,
                                             phWaitEvents)
    ccall((:zeCommandListAppendBarrier, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, hSignalEvent, numWaitEvents, phWaitEvents)
end

@checked function zeCommandListAppendMemoryRangesBarrier(hCommandList, numRanges,
                                                         pRangeSizes, pRanges,
                                                         hSignalEvent, numWaitEvents,
                                                         phWaitEvents)
    ccall((:zeCommandListAppendMemoryRangesBarrier, libze_loader), ze_result_t,
          (ze_command_list_handle_t, UInt32, Ptr{Csize_t}, Ptr{Ptr{Cvoid}},
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, numRanges, pRangeSizes, pRanges, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeContextSystemBarrier(hContext, hDevice)
    ccall((:zeContextSystemBarrier, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t),
          hContext, hDevice)
end

@checked function zeCommandListAppendMemoryCopy(hCommandList, dstptr, srcptr, size,
                                                hSignalEvent, numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendMemoryCopy, libze_loader), ze_result_t,
          (ze_command_list_handle_t, PtrOrZePtr{Cvoid}, PtrOrZePtr{Cvoid}, Csize_t, ze_event_handle_t,
           UInt32, Ptr{ze_event_handle_t}),
          hCommandList, dstptr, srcptr, size, hSignalEvent, numWaitEvents, phWaitEvents)
end

@checked function zeCommandListAppendMemoryFill(hCommandList, ptr, pattern, pattern_size,
                                                size, hSignalEvent, numWaitEvents,
                                                phWaitEvents)
    ccall((:zeCommandListAppendMemoryFill, libze_loader), ze_result_t,
          (ze_command_list_handle_t, PtrOrZePtr{Cvoid}, PtrOrZePtr{Cvoid}, Csize_t, Csize_t,
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, ptr, pattern, pattern_size, size, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendMemoryCopyRegion(hCommandList, dstptr, dstRegion,
                                                      dstPitch, dstSlicePitch, srcptr,
                                                      srcRegion, srcPitch, srcSlicePitch,
                                                      hSignalEvent, numWaitEvents,
                                                      phWaitEvents)
    ccall((:zeCommandListAppendMemoryCopyRegion, libze_loader), ze_result_t,
          (ze_command_list_handle_t, PtrOrZePtr{Cvoid}, Ptr{ze_copy_region_t}, UInt32, UInt32,
           PtrOrZePtr{Cvoid}, Ptr{ze_copy_region_t}, UInt32, UInt32, ze_event_handle_t, UInt32,
           Ptr{ze_event_handle_t}),
          hCommandList, dstptr, dstRegion, dstPitch, dstSlicePitch, srcptr, srcRegion,
          srcPitch, srcSlicePitch, hSignalEvent, numWaitEvents, phWaitEvents)
end

@checked function zeCommandListAppendMemoryCopyFromContext(hCommandList, dstptr,
                                                           hContextSrc, srcptr, size,
                                                           hSignalEvent, numWaitEvents,
                                                           phWaitEvents)
    ccall((:zeCommandListAppendMemoryCopyFromContext, libze_loader), ze_result_t,
          (ze_command_list_handle_t, PtrOrZePtr{Cvoid}, ze_context_handle_t, PtrOrZePtr{Cvoid}, Csize_t,
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, dstptr, hContextSrc, srcptr, size, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendImageCopy(hCommandList, hDstImage, hSrcImage,
                                               hSignalEvent, numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendImageCopy, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t,
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, hDstImage, hSrcImage, hSignalEvent, numWaitEvents, phWaitEvents)
end

@checked function zeCommandListAppendImageCopyRegion(hCommandList, hDstImage, hSrcImage,
                                                     pDstRegion, pSrcRegion, hSignalEvent,
                                                     numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendImageCopyRegion, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t,
           Ptr{ze_image_region_t}, Ptr{ze_image_region_t}, ze_event_handle_t, UInt32,
           Ptr{ze_event_handle_t}),
          hCommandList, hDstImage, hSrcImage, pDstRegion, pSrcRegion, hSignalEvent,
          numWaitEvents, phWaitEvents)
end

@checked function zeCommandListAppendImageCopyToMemory(hCommandList, dstptr, hSrcImage,
                                                       pSrcRegion, hSignalEvent,
                                                       numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendImageCopyToMemory, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{Cvoid}, ze_image_handle_t,
           Ptr{ze_image_region_t}, ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, dstptr, hSrcImage, pSrcRegion, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendImageCopyFromMemory(hCommandList, hDstImage, srcptr,
                                                         pDstRegion, hSignalEvent,
                                                         numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendImageCopyFromMemory, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_image_handle_t, Ptr{Cvoid},
           Ptr{ze_image_region_t}, ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, hDstImage, srcptr, pDstRegion, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendMemoryPrefetch(hCommandList, ptr, size)
    ccall((:zeCommandListAppendMemoryPrefetch, libze_loader), ze_result_t,
          (ze_command_list_handle_t, PtrOrZePtr{Cvoid}, Csize_t),
          hCommandList, ptr, size)
end

@checked function zeCommandListAppendMemAdvise(hCommandList, hDevice, ptr, size, advice)
    ccall((:zeCommandListAppendMemAdvise, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_device_handle_t, PtrOrZePtr{Cvoid}, Csize_t,
           ze_memory_advice_t),
          hCommandList, hDevice, ptr, size, advice)
end

@checked function zeEventPoolCreate(hContext, desc, numDevices, phDevices, phEventPool)
    ccall((:zeEventPoolCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{ze_event_pool_desc_t}, UInt32,
           Ptr{ze_device_handle_t}, Ptr{ze_event_pool_handle_t}),
          hContext, desc, numDevices, phDevices, phEventPool)
end

@checked function zeEventPoolDestroy(hEventPool)
    ccall((:zeEventPoolDestroy, libze_loader), ze_result_t,
          (ze_event_pool_handle_t,),
          hEventPool)
end

@checked function zeEventCreate(hEventPool, desc, phEvent)
    ccall((:zeEventCreate, libze_loader), ze_result_t,
          (ze_event_pool_handle_t, Ptr{ze_event_desc_t}, Ptr{ze_event_handle_t}),
          hEventPool, desc, phEvent)
end

@checked function zeEventDestroy(hEvent)
    ccall((:zeEventDestroy, libze_loader), ze_result_t,
          (ze_event_handle_t,),
          hEvent)
end

@checked function zeEventPoolGetIpcHandle(hEventPool, phIpc)
    ccall((:zeEventPoolGetIpcHandle, libze_loader), ze_result_t,
          (ze_event_pool_handle_t, Ptr{ze_ipc_event_pool_handle_t}),
          hEventPool, phIpc)
end

@checked function zeEventPoolOpenIpcHandle(hContext, hIpc, phEventPool)
    ccall((:zeEventPoolOpenIpcHandle, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_ipc_event_pool_handle_t, Ptr{ze_event_pool_handle_t}),
          hContext, hIpc, phEventPool)
end

@checked function zeEventPoolCloseIpcHandle(hEventPool)
    ccall((:zeEventPoolCloseIpcHandle, libze_loader), ze_result_t,
          (ze_event_pool_handle_t,),
          hEventPool)
end

@checked function zeCommandListAppendSignalEvent(hCommandList, hEvent)
    ccall((:zeCommandListAppendSignalEvent, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_event_handle_t),
          hCommandList, hEvent)
end

@checked function zeCommandListAppendWaitOnEvents(hCommandList, numEvents, phEvents)
    ccall((:zeCommandListAppendWaitOnEvents, libze_loader), ze_result_t,
          (ze_command_list_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, numEvents, phEvents)
end

@checked function zeEventHostSignal(hEvent)
    ccall((:zeEventHostSignal, libze_loader), ze_result_t,
          (ze_event_handle_t,),
          hEvent)
end

@checked function zeEventHostSynchronize(hEvent, timeout)
    ccall((:zeEventHostSynchronize, libze_loader), ze_result_t,
          (ze_event_handle_t, UInt64),
          hEvent, timeout)
end

@checked function zeEventQueryStatus(hEvent)
    ccall((:zeEventQueryStatus, libze_loader), ze_result_t,
          (ze_event_handle_t,),
          hEvent)
end

@checked function zeCommandListAppendEventReset(hCommandList, hEvent)
    ccall((:zeCommandListAppendEventReset, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_event_handle_t),
          hCommandList, hEvent)
end

@checked function zeEventHostReset(hEvent)
    ccall((:zeEventHostReset, libze_loader), ze_result_t,
          (ze_event_handle_t,),
          hEvent)
end

@checked function zeEventQueryKernelTimestamp(hEvent, dstptr)
    ccall((:zeEventQueryKernelTimestamp, libze_loader), ze_result_t,
          (ze_event_handle_t, Ptr{ze_kernel_timestamp_result_t}),
          hEvent, dstptr)
end

@checked function zeCommandListAppendQueryKernelTimestamps(hCommandList, numEvents,
                                                           phEvents, dstptr, pOffsets,
                                                           hSignalEvent, numWaitEvents,
                                                           phWaitEvents)
    ccall((:zeCommandListAppendQueryKernelTimestamps, libze_loader), ze_result_t,
          (ze_command_list_handle_t, UInt32, Ptr{ze_event_handle_t}, Ptr{Cvoid},
           Ptr{Csize_t}, ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, numEvents, phEvents, dstptr, pOffsets, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeFenceCreate(hCommandQueue, desc, phFence)
    ccall((:zeFenceCreate, libze_loader), ze_result_t,
          (ze_command_queue_handle_t, Ptr{ze_fence_desc_t}, Ptr{ze_fence_handle_t}),
          hCommandQueue, desc, phFence)
end

@checked function zeFenceDestroy(hFence)
    ccall((:zeFenceDestroy, libze_loader), ze_result_t,
          (ze_fence_handle_t,),
          hFence)
end

@checked function zeFenceHostSynchronize(hFence, timeout)
    ccall((:zeFenceHostSynchronize, libze_loader), ze_result_t,
          (ze_fence_handle_t, UInt64),
          hFence, timeout)
end

@checked function zeFenceQueryStatus(hFence)
    ccall((:zeFenceQueryStatus, libze_loader), ze_result_t,
          (ze_fence_handle_t,),
          hFence)
end

@checked function zeFenceReset(hFence)
    ccall((:zeFenceReset, libze_loader), ze_result_t,
          (ze_fence_handle_t,),
          hFence)
end

@checked function zeImageGetProperties(hDevice, desc, pImageProperties)
    ccall((:zeImageGetProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_image_desc_t}, Ptr{ze_image_properties_t}),
          hDevice, desc, pImageProperties)
end

@checked function zeImageCreate(hContext, hDevice, desc, phImage)
    ccall((:zeImageCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_image_desc_t},
           Ptr{ze_image_handle_t}),
          hContext, hDevice, desc, phImage)
end

@checked function zeImageDestroy(hImage)
    ccall((:zeImageDestroy, libze_loader), ze_result_t,
          (ze_image_handle_t,),
          hImage)
end

@checked function zeMemAllocShared(hContext, device_desc, host_desc, size, alignment,
                                   hDevice, pptr)
    ccall((:zeMemAllocShared, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{ze_device_mem_alloc_desc_t},
           Ptr{ze_host_mem_alloc_desc_t}, Csize_t, Csize_t, ze_device_handle_t,
           Ptr{Ptr{Cvoid}}),
          hContext, device_desc, host_desc, size, alignment, hDevice, pptr)
end

@checked function zeMemAllocDevice(hContext, device_desc, size, alignment, hDevice, pptr)
    ccall((:zeMemAllocDevice, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{ze_device_mem_alloc_desc_t}, Csize_t, Csize_t,
           ze_device_handle_t, Ptr{Ptr{Cvoid}}),
          hContext, device_desc, size, alignment, hDevice, pptr)
end

@checked function zeMemAllocHost(hContext, host_desc, size, alignment, pptr)
    ccall((:zeMemAllocHost, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{ze_host_mem_alloc_desc_t}, Csize_t, Csize_t,
           Ptr{Ptr{Cvoid}}),
          hContext, host_desc, size, alignment, pptr)
end

@checked function zeMemFree(hContext, ptr)
    ccall((:zeMemFree, libze_loader), ze_result_t,
          (ze_context_handle_t, PtrOrZePtr{Cvoid}),
          hContext, ptr)
end

@checked function zeMemGetAllocProperties(hContext, ptr, pMemAllocProperties, phDevice)
    ccall((:zeMemGetAllocProperties, libze_loader), ze_result_t,
          (ze_context_handle_t, PtrOrZePtr{Cvoid}, Ptr{ze_memory_allocation_properties_t},
           Ptr{ze_device_handle_t}),
          hContext, ptr, pMemAllocProperties, phDevice)
end

@checked function zeMemGetAddressRange(hContext, ptr, pBase, pSize)
    ccall((:zeMemGetAddressRange, libze_loader), ze_result_t,
          (ze_context_handle_t, PtrOrZePtr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}),
          hContext, ptr, pBase, pSize)
end

@checked function zeMemGetIpcHandle(hContext, ptr, pIpcHandle)
    ccall((:zeMemGetIpcHandle, libze_loader), ze_result_t,
          (ze_context_handle_t, PtrOrZePtr{Cvoid}, Ptr{ze_ipc_mem_handle_t}),
          hContext, ptr, pIpcHandle)
end

@checked function zeMemOpenIpcHandle(hContext, hDevice, handle, flags, pptr)
    ccall((:zeMemOpenIpcHandle, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t,
           ze_ipc_memory_flags_t, PtrOrZePtr{Ptr{Cvoid}}),
          hContext, hDevice, handle, flags, pptr)
end

@checked function zeMemCloseIpcHandle(hContext, ptr)
    ccall((:zeMemCloseIpcHandle, libze_loader), ze_result_t,
          (ze_context_handle_t, PtrOrZePtr{Cvoid}),
          hContext, ptr)
end

@checked function zeModuleCreate(hContext, hDevice, desc, phModule, phBuildLog)
    ccall((:zeModuleCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_module_desc_t},
           Ptr{ze_module_handle_t}, Ptr{ze_module_build_log_handle_t}),
          hContext, hDevice, desc, phModule, phBuildLog)
end

@checked function zeModuleDestroy(hModule)
    ccall((:zeModuleDestroy, libze_loader), ze_result_t,
          (ze_module_handle_t,),
          hModule)
end

@checked function zeModuleDynamicLink(numModules, phModules, phLinkLog)
    ccall((:zeModuleDynamicLink, libze_loader), ze_result_t,
          (UInt32, Ptr{ze_module_handle_t}, Ptr{ze_module_build_log_handle_t}),
          numModules, phModules, phLinkLog)
end

@checked function zeModuleBuildLogDestroy(hModuleBuildLog)
    ccall((:zeModuleBuildLogDestroy, libze_loader), ze_result_t,
          (ze_module_build_log_handle_t,),
          hModuleBuildLog)
end

@checked function zeModuleBuildLogGetString(hModuleBuildLog, pSize, pBuildLog)
    ccall((:zeModuleBuildLogGetString, libze_loader), ze_result_t,
          (ze_module_build_log_handle_t, Ptr{Csize_t}, Cstring),
          hModuleBuildLog, pSize, pBuildLog)
end

@checked function zeModuleGetNativeBinary(hModule, pSize, pModuleNativeBinary)
    ccall((:zeModuleGetNativeBinary, libze_loader), ze_result_t,
          (ze_module_handle_t, Ptr{Csize_t}, Ptr{UInt8}),
          hModule, pSize, pModuleNativeBinary)
end

@checked function zeModuleGetGlobalPointer(hModule, pGlobalName, pSize, pptr)
    ccall((:zeModuleGetGlobalPointer, libze_loader), ze_result_t,
          (ze_module_handle_t, Cstring, Ptr{Csize_t}, Ptr{Ptr{Cvoid}}),
          hModule, pGlobalName, pSize, pptr)
end

@checked function zeModuleGetKernelNames(hModule, pCount, pNames)
    ccall((:zeModuleGetKernelNames, libze_loader), ze_result_t,
          (ze_module_handle_t, Ptr{UInt32}, Ptr{Cstring}),
          hModule, pCount, pNames)
end

@checked function zeModuleGetProperties(hModule, pModuleProperties)
    ccall((:zeModuleGetProperties, libze_loader), ze_result_t,
          (ze_module_handle_t, Ptr{ze_module_properties_t}),
          hModule, pModuleProperties)
end

@checked function zeKernelCreate(hModule, desc, phKernel)
    ccall((:zeKernelCreate, libze_loader), ze_result_t,
          (ze_module_handle_t, Ptr{ze_kernel_desc_t}, Ptr{ze_kernel_handle_t}),
          hModule, desc, phKernel)
end

@checked function zeKernelDestroy(hKernel)
    ccall((:zeKernelDestroy, libze_loader), ze_result_t,
          (ze_kernel_handle_t,),
          hKernel)
end

@checked function zeModuleGetFunctionPointer(hModule, pFunctionName, pfnFunction)
    ccall((:zeModuleGetFunctionPointer, libze_loader), ze_result_t,
          (ze_module_handle_t, Cstring, Ptr{Ptr{Cvoid}}),
          hModule, pFunctionName, pfnFunction)
end

@checked function zeKernelSetGroupSize(hKernel, groupSizeX, groupSizeY, groupSizeZ)
    ccall((:zeKernelSetGroupSize, libze_loader), ze_result_t,
          (ze_kernel_handle_t, UInt32, UInt32, UInt32),
          hKernel, groupSizeX, groupSizeY, groupSizeZ)
end

@checked function zeKernelSuggestGroupSize(hKernel, globalSizeX, globalSizeY, globalSizeZ,
                                           groupSizeX, groupSizeY, groupSizeZ)
    ccall((:zeKernelSuggestGroupSize, libze_loader), ze_result_t,
          (ze_kernel_handle_t, UInt32, UInt32, UInt32, Ptr{UInt32}, Ptr{UInt32},
           Ptr{UInt32}),
          hKernel, globalSizeX, globalSizeY, globalSizeZ, groupSizeX, groupSizeY,
          groupSizeZ)
end

@checked function zeKernelSuggestMaxCooperativeGroupCount(hKernel, totalGroupCount)
    ccall((:zeKernelSuggestMaxCooperativeGroupCount, libze_loader), ze_result_t,
          (ze_kernel_handle_t, Ptr{UInt32}),
          hKernel, totalGroupCount)
end

@checked function zeKernelSetArgumentValue(hKernel, argIndex, argSize, pArgValue)
    ccall((:zeKernelSetArgumentValue, libze_loader), ze_result_t,
          (ze_kernel_handle_t, UInt32, Csize_t, Ptr{Cvoid}),
          hKernel, argIndex, argSize, pArgValue)
end

@checked function zeKernelSetIndirectAccess(hKernel, flags)
    ccall((:zeKernelSetIndirectAccess, libze_loader), ze_result_t,
          (ze_kernel_handle_t, ze_kernel_indirect_access_flags_t),
          hKernel, flags)
end

@checked function zeKernelGetIndirectAccess(hKernel, pFlags)
    ccall((:zeKernelGetIndirectAccess, libze_loader), ze_result_t,
          (ze_kernel_handle_t, Ptr{ze_kernel_indirect_access_flags_t}),
          hKernel, pFlags)
end

@checked function zeKernelGetSourceAttributes(hKernel, pSize, pString)
    ccall((:zeKernelGetSourceAttributes, libze_loader), ze_result_t,
          (ze_kernel_handle_t, Ptr{UInt32}, Ptr{Cstring}),
          hKernel, pSize, pString)
end

@checked function zeKernelSetCacheConfig(hKernel, flags)
    ccall((:zeKernelSetCacheConfig, libze_loader), ze_result_t,
          (ze_kernel_handle_t, ze_cache_config_flags_t),
          hKernel, flags)
end

@checked function zeKernelGetProperties(hKernel, pKernelProperties)
    ccall((:zeKernelGetProperties, libze_loader), ze_result_t,
          (ze_kernel_handle_t, Ptr{ze_kernel_properties_t}),
          hKernel, pKernelProperties)
end

@checked function zeKernelGetName(hKernel, pSize, pName)
    ccall((:zeKernelGetName, libze_loader), ze_result_t,
          (ze_kernel_handle_t, Ptr{Csize_t}, Cstring),
          hKernel, pSize, pName)
end

@checked function zeCommandListAppendLaunchKernel(hCommandList, hKernel, pLaunchFuncArgs,
                                                  hSignalEvent, numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendLaunchKernel, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_kernel_handle_t, Ptr{ze_group_count_t},
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, hKernel, pLaunchFuncArgs, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendLaunchCooperativeKernel(hCommandList, hKernel,
                                                             pLaunchFuncArgs, hSignalEvent,
                                                             numWaitEvents, phWaitEvents)
    ccall((:zeCommandListAppendLaunchCooperativeKernel, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_kernel_handle_t, Ptr{ze_group_count_t},
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, hKernel, pLaunchFuncArgs, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendLaunchKernelIndirect(hCommandList, hKernel,
                                                          pLaunchArgumentsBuffer,
                                                          hSignalEvent, numWaitEvents,
                                                          phWaitEvents)
    ccall((:zeCommandListAppendLaunchKernelIndirect, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_kernel_handle_t, Ptr{ze_group_count_t},
           ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, hKernel, pLaunchArgumentsBuffer, hSignalEvent, numWaitEvents,
          phWaitEvents)
end

@checked function zeCommandListAppendLaunchMultipleKernelsIndirect(hCommandList,
                                                                   numKernels, phKernels,
                                                                   pCountBuffer,
                                                                   pLaunchArgumentsBuffer,
                                                                   hSignalEvent,
                                                                   numWaitEvents,
                                                                   phWaitEvents)
    ccall((:zeCommandListAppendLaunchMultipleKernelsIndirect, libze_loader), ze_result_t,
          (ze_command_list_handle_t, UInt32, Ptr{ze_kernel_handle_t}, Ptr{UInt32},
           Ptr{ze_group_count_t}, ze_event_handle_t, UInt32, Ptr{ze_event_handle_t}),
          hCommandList, numKernels, phKernels, pCountBuffer, pLaunchArgumentsBuffer,
          hSignalEvent, numWaitEvents, phWaitEvents)
end

@checked function zeContextMakeMemoryResident(hContext, hDevice, ptr, size)
    ccall((:zeContextMakeMemoryResident, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, PtrOrZePtr{Cvoid}, Csize_t),
          hContext, hDevice, ptr, size)
end

@checked function zeContextEvictMemory(hContext, hDevice, ptr, size)
    ccall((:zeContextEvictMemory, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, PtrOrZePtr{Cvoid}, Csize_t),
          hContext, hDevice, ptr, size)
end

@checked function zeContextMakeImageResident(hContext, hDevice, hImage)
    ccall((:zeContextMakeImageResident, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, ze_image_handle_t),
          hContext, hDevice, hImage)
end

@checked function zeContextEvictImage(hContext, hDevice, hImage)
    ccall((:zeContextEvictImage, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, ze_image_handle_t),
          hContext, hDevice, hImage)
end

@checked function zeSamplerCreate(hContext, hDevice, desc, phSampler)
    ccall((:zeSamplerCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_sampler_desc_t},
           Ptr{ze_sampler_handle_t}),
          hContext, hDevice, desc, phSampler)
end

@checked function zeSamplerDestroy(hSampler)
    ccall((:zeSamplerDestroy, libze_loader), ze_result_t,
          (ze_sampler_handle_t,),
          hSampler)
end

@checked function zeVirtualMemReserve(hContext, pStart, size, pptr)
    ccall((:zeVirtualMemReserve, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}),
          hContext, pStart, size, pptr)
end

@checked function zeVirtualMemFree(hContext, ptr, size)
    ccall((:zeVirtualMemFree, libze_loader), ze_result_t,
          (ze_context_handle_t, PtrOrZePtr{Cvoid}, Csize_t),
          hContext, ptr, size)
end

@checked function zeVirtualMemQueryPageSize(hContext, hDevice, size, pagesize)
    ccall((:zeVirtualMemQueryPageSize, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Csize_t, Ptr{Csize_t}),
          hContext, hDevice, size, pagesize)
end

@checked function zePhysicalMemCreate(hContext, hDevice, desc, phPhysicalMemory)
    ccall((:zePhysicalMemCreate, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_device_handle_t, Ptr{ze_physical_mem_desc_t},
           Ptr{ze_physical_mem_handle_t}),
          hContext, hDevice, desc, phPhysicalMemory)
end

@checked function zePhysicalMemDestroy(hContext, hPhysicalMemory)
    ccall((:zePhysicalMemDestroy, libze_loader), ze_result_t,
          (ze_context_handle_t, ze_physical_mem_handle_t),
          hContext, hPhysicalMemory)
end

@checked function zeVirtualMemMap(hContext, ptr, size, hPhysicalMemory, offset, access)
    ccall((:zeVirtualMemMap, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{Cvoid}, Csize_t, ze_physical_mem_handle_t, Csize_t,
           ze_memory_access_attribute_t),
          hContext, ptr, size, hPhysicalMemory, offset, access)
end

@checked function zeVirtualMemUnmap(hContext, ptr, size)
    ccall((:zeVirtualMemUnmap, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{Cvoid}, Csize_t),
          hContext, ptr, size)
end

@checked function zeVirtualMemSetAccessAttribute(hContext, ptr, size, access)
    ccall((:zeVirtualMemSetAccessAttribute, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{Cvoid}, Csize_t, ze_memory_access_attribute_t),
          hContext, ptr, size, access)
end

@checked function zeVirtualMemGetAccessAttribute(hContext, ptr, size, access, outSize)
    ccall((:zeVirtualMemGetAccessAttribute, libze_loader), ze_result_t,
          (ze_context_handle_t, Ptr{Cvoid}, Csize_t, Ptr{ze_memory_access_attribute_t},
           Ptr{Csize_t}),
          hContext, ptr, size, access, outSize)
end

@checked function zeKernelSetGlobalOffsetExp(hKernel, offsetX, offsetY, offsetZ)
    ccall((:zeKernelSetGlobalOffsetExp, libze_loader), ze_result_t,
          (ze_kernel_handle_t, UInt32, UInt32, UInt32),
          hKernel, offsetX, offsetY, offsetZ)
end
