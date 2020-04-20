# Julia wrapper for header: ze_api.h
# Automatically generated using Clang.jl


@checked function zeInit(flags)
    ccall((:zeInit, libze_loader), ze_result_t,
          (ze_init_flag_t,),
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

@checked function zeDriverGetIPCProperties(hDriver, pIPCProperties)
    ccall((:zeDriverGetIPCProperties, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_driver_ipc_properties_t}),
          hDriver, pIPCProperties)
end

@checked function zeDriverGetExtensionFunctionAddress(hDriver, pFuncName, pfunc)
    ccall((:zeDriverGetExtensionFunctionAddress, libze_loader), ze_result_t,
          (ze_driver_handle_t, Cstring, Ptr{Ptr{Cvoid}}),
          hDriver, pFuncName, pfunc)
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

@checked function zeDeviceGetKernelProperties(hDevice, pKernelProperties)
    ccall((:zeDeviceGetKernelProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_kernel_properties_t}),
          hDevice, pKernelProperties)
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

@checked function zeDeviceGetCacheProperties(hDevice, pCacheProperties)
    ccall((:zeDeviceGetCacheProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_cache_properties_t}),
          hDevice, pCacheProperties)
end

@checked function zeDeviceGetImageProperties(hDevice, pImageProperties)
    ccall((:zeDeviceGetImageProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_device_image_properties_t}),
          hDevice, pImageProperties)
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

@checked function zeDeviceSetLastLevelCacheConfig(hDevice, CacheConfig)
    ccall((:zeDeviceSetLastLevelCacheConfig, libze_loader), ze_result_t,
          (ze_device_handle_t, ze_cache_config_t),
          hDevice, CacheConfig)
end

@checked function zeCommandQueueCreate(hDevice, desc, phCommandQueue)
    ccall((:zeCommandQueueCreate, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_command_queue_desc_t},
           Ptr{ze_command_queue_handle_t}),
          hDevice, desc, phCommandQueue)
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
          (ze_command_queue_handle_t, UInt32),
          hCommandQueue, timeout)
end

@checked function zeCommandListCreate(hDevice, desc, phCommandList)
    ccall((:zeCommandListCreate, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_command_list_desc_t}, Ptr{ze_command_list_handle_t}),
          hDevice, desc, phCommandList)
end

@checked function zeCommandListCreateImmediate(hDevice, altdesc, phCommandList)
    ccall((:zeCommandListCreateImmediate, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_command_queue_desc_t}, Ptr{ze_command_list_handle_t}),
          hDevice, altdesc, phCommandList)
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

@checked function zeImageGetProperties(hDevice, desc, pImageProperties)
    ccall((:zeImageGetProperties, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_image_desc_t}, Ptr{ze_image_properties_t}),
          hDevice, desc, pImageProperties)
end

@checked function zeImageCreate(hDevice, desc, phImage)
    ccall((:zeImageCreate, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_image_desc_t}, Ptr{ze_image_handle_t}),
          hDevice, desc, phImage)
end

@checked function zeImageDestroy(hImage)
    ccall((:zeImageDestroy, libze_loader), ze_result_t,
          (ze_image_handle_t,),
          hImage)
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

@checked function zeDeviceSystemBarrier(hDevice)
    ccall((:zeDeviceSystemBarrier, libze_loader), ze_result_t,
          (ze_device_handle_t,),
          hDevice)
end

@checked function zeModuleCreate(hDevice, desc, phModule, phBuildLog)
    ccall((:zeModuleCreate, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_module_desc_t}, Ptr{ze_module_handle_t},
           Ptr{ze_module_build_log_handle_t}),
          hDevice, desc, phModule, phBuildLog)
end

@checked function zeModuleDestroy(hModule)
    ccall((:zeModuleDestroy, libze_loader), ze_result_t,
          (ze_module_handle_t,),
          hModule)
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

@checked function zeModuleGetGlobalPointer(hModule, pGlobalName, pptr)
    ccall((:zeModuleGetGlobalPointer, libze_loader), ze_result_t,
          (ze_module_handle_t, Cstring, Ptr{Ptr{Cvoid}}),
          hModule, pGlobalName, pptr)
end

@checked function zeModuleGetKernelNames(hModule, pCount, pNames)
    ccall((:zeModuleGetKernelNames, libze_loader), ze_result_t,
          (ze_module_handle_t, Ptr{UInt32}, Ptr{Cstring}),
          hModule, pCount, pNames)
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

@checked function zeKernelSetAttribute(hKernel, attr, size, pValue)
    ccall((:zeKernelSetAttribute, libze_loader), ze_result_t,
          (ze_kernel_handle_t, ze_kernel_attribute_t, UInt32, Ptr{Cvoid}),
          hKernel, attr, size, pValue)
end

@checked function zeKernelGetAttribute(hKernel, attr, pSize, pValue)
    ccall((:zeKernelGetAttribute, libze_loader), ze_result_t,
          (ze_kernel_handle_t, ze_kernel_attribute_t, Ptr{UInt32}, Ptr{Cvoid}),
          hKernel, attr, pSize, pValue)
end

@checked function zeKernelSetIntermediateCacheConfig(hKernel, CacheConfig)
    ccall((:zeKernelSetIntermediateCacheConfig, libze_loader), ze_result_t,
          (ze_kernel_handle_t, ze_cache_config_t),
          hKernel, CacheConfig)
end

@checked function zeKernelGetProperties(hKernel, pKernelProperties)
    ccall((:zeKernelGetProperties, libze_loader), ze_result_t,
          (ze_kernel_handle_t, Ptr{ze_kernel_properties_t}),
          hKernel, pKernelProperties)
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

@checked function zeDeviceMakeMemoryResident(hDevice, ptr, size)
    ccall((:zeDeviceMakeMemoryResident, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{Cvoid}, Csize_t),
          hDevice, ptr, size)
end

@checked function zeDeviceEvictMemory(hDevice, ptr, size)
    ccall((:zeDeviceEvictMemory, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{Cvoid}, Csize_t),
          hDevice, ptr, size)
end

@checked function zeDeviceMakeImageResident(hDevice, hImage)
    ccall((:zeDeviceMakeImageResident, libze_loader), ze_result_t,
          (ze_device_handle_t, ze_image_handle_t),
          hDevice, hImage)
end

@checked function zeDeviceEvictImage(hDevice, hImage)
    ccall((:zeDeviceEvictImage, libze_loader), ze_result_t,
          (ze_device_handle_t, ze_image_handle_t),
          hDevice, hImage)
end

@checked function zeEventPoolCreate(hDriver, desc, numDevices, phDevices, phEventPool)
    ccall((:zeEventPoolCreate, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_event_pool_desc_t}, UInt32, Ptr{ze_device_handle_t},
           Ptr{ze_event_pool_handle_t}),
          hDriver, desc, numDevices, phDevices, phEventPool)
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

@checked function zeEventPoolOpenIpcHandle(hDriver, hIpc, phEventPool)
    ccall((:zeEventPoolOpenIpcHandle, libze_loader), ze_result_t,
          (ze_driver_handle_t, ze_ipc_event_pool_handle_t, Ptr{ze_event_pool_handle_t}),
          hDriver, hIpc, phEventPool)
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
          (ze_event_handle_t, UInt32),
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

@checked function zeEventGetTimestamp(hEvent, timestampType, dstptr)
    ccall((:zeEventGetTimestamp, libze_loader), ze_result_t,
          (ze_event_handle_t, ze_event_timestamp_type_t, Ptr{Cvoid}),
          hEvent, timestampType, dstptr)
end

@checked function zeSamplerCreate(hDevice, desc, phSampler)
    ccall((:zeSamplerCreate, libze_loader), ze_result_t,
          (ze_device_handle_t, Ptr{ze_sampler_desc_t}, Ptr{ze_sampler_handle_t}),
          hDevice, desc, phSampler)
end

@checked function zeSamplerDestroy(hSampler)
    ccall((:zeSamplerDestroy, libze_loader), ze_result_t,
          (ze_sampler_handle_t,),
          hSampler)
end

@checked function zeDriverAllocSharedMem(hDriver, device_desc, host_desc, size, alignment,
                                         hDevice, pptr)
    ccall((:zeDriverAllocSharedMem, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_device_mem_alloc_desc_t},
           Ptr{ze_host_mem_alloc_desc_t}, Csize_t, Csize_t, ze_device_handle_t,
           Ptr{Ptr{Cvoid}}),
          hDriver, device_desc, host_desc, size, alignment, hDevice, pptr)
end

@checked function zeDriverAllocDeviceMem(hDriver, device_desc, size, alignment, hDevice,
                                         pptr)
    ccall((:zeDriverAllocDeviceMem, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_device_mem_alloc_desc_t}, Csize_t, Csize_t,
           ze_device_handle_t, Ptr{Ptr{Cvoid}}),
          hDriver, device_desc, size, alignment, hDevice, pptr)
end

@checked function zeDriverAllocHostMem(hDriver, host_desc, size, alignment, pptr)
    ccall((:zeDriverAllocHostMem, libze_loader), ze_result_t,
          (ze_driver_handle_t, Ptr{ze_host_mem_alloc_desc_t}, Csize_t, Csize_t,
           Ptr{Ptr{Cvoid}}),
          hDriver, host_desc, size, alignment, pptr)
end

@checked function zeDriverFreeMem(hDriver, ptr)
    ccall((:zeDriverFreeMem, libze_loader), ze_result_t,
          (ze_driver_handle_t, PtrOrZePtr{Cvoid}),
          hDriver, ptr)
end

@checked function zeDriverGetMemAllocProperties(hDriver, ptr, pMemAllocProperties, phDevice)
    ccall((:zeDriverGetMemAllocProperties, libze_loader), ze_result_t,
          (ze_driver_handle_t, PtrOrZePtr{Cvoid}, Ptr{ze_memory_allocation_properties_t},
           Ptr{ze_device_handle_t}),
          hDriver, ptr, pMemAllocProperties, phDevice)
end

@checked function zeDriverGetMemAddressRange(hDriver, ptr, pBase, pSize)
    ccall((:zeDriverGetMemAddressRange, libze_loader), ze_result_t,
          (ze_driver_handle_t, PtrOrZePtr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}),
          hDriver, ptr, pBase, pSize)
end

@checked function zeDriverGetMemIpcHandle(hDriver, ptr, pIpcHandle)
    ccall((:zeDriverGetMemIpcHandle, libze_loader), ze_result_t,
          (ze_driver_handle_t, PtrOrZePtr{Cvoid}, Ptr{ze_ipc_mem_handle_t}),
          hDriver, ptr, pIpcHandle)
end

@checked function zeDriverOpenMemIpcHandle(hDriver, hDevice, handle, flags, pptr)
    ccall((:zeDriverOpenMemIpcHandle, libze_loader), ze_result_t,
          (ze_driver_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t,
           ze_ipc_memory_flag_t, Ptr{Ptr{Cvoid}}),
          hDriver, hDevice, handle, flags, pptr)
end

@checked function zeDriverCloseMemIpcHandle(hDriver, ptr)
    ccall((:zeDriverCloseMemIpcHandle, libze_loader), ze_result_t,
          (ze_driver_handle_t, PtrOrZePtr{Cvoid}),
          hDriver, ptr)
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
          (ze_fence_handle_t, UInt32),
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

@checked function zeCommandListAppendMemoryCopy(hCommandList, dstptr, srcptr, size, hEvent)
    ccall((:zeCommandListAppendMemoryCopy, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, ze_event_handle_t),
          hCommandList, dstptr, srcptr, size, hEvent)
end

@checked function zeCommandListAppendMemoryFill(hCommandList, ptr, pattern, pattern_size,
                                                size, hEvent)
    ccall((:zeCommandListAppendMemoryFill, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t,
           ze_event_handle_t),
          hCommandList, ptr, pattern, pattern_size, size, hEvent)
end

@checked function zeCommandListAppendMemoryCopyRegion(hCommandList, dstptr, dstRegion,
                                                      dstPitch, dstSlicePitch, srcptr,
                                                      srcRegion, srcPitch, srcSlicePitch,
                                                      hEvent)
    ccall((:zeCommandListAppendMemoryCopyRegion, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{Cvoid}, Ptr{ze_copy_region_t}, UInt32, UInt32,
           Ptr{Cvoid}, Ptr{ze_copy_region_t}, UInt32, UInt32, ze_event_handle_t),
          hCommandList, dstptr, dstRegion, dstPitch, dstSlicePitch, srcptr, srcRegion,
          srcPitch, srcSlicePitch, hEvent)
end

@checked function zeCommandListAppendImageCopy(hCommandList, hDstImage, hSrcImage, hEvent)
    ccall((:zeCommandListAppendImageCopy, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t,
           ze_event_handle_t),
          hCommandList, hDstImage, hSrcImage, hEvent)
end

@checked function zeCommandListAppendImageCopyRegion(hCommandList, hDstImage, hSrcImage,
                                                     pDstRegion, pSrcRegion, hEvent)
    ccall((:zeCommandListAppendImageCopyRegion, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t,
           Ptr{ze_image_region_t}, Ptr{ze_image_region_t}, ze_event_handle_t),
          hCommandList, hDstImage, hSrcImage, pDstRegion, pSrcRegion, hEvent)
end

@checked function zeCommandListAppendImageCopyToMemory(hCommandList, dstptr, hSrcImage,
                                                       pSrcRegion, hEvent)
    ccall((:zeCommandListAppendImageCopyToMemory, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{Cvoid}, ze_image_handle_t,
           Ptr{ze_image_region_t}, ze_event_handle_t),
          hCommandList, dstptr, hSrcImage, pSrcRegion, hEvent)
end

@checked function zeCommandListAppendImageCopyFromMemory(hCommandList, hDstImage, srcptr,
                                                         pDstRegion, hEvent)
    ccall((:zeCommandListAppendImageCopyFromMemory, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_image_handle_t, Ptr{Cvoid},
           Ptr{ze_image_region_t}, ze_event_handle_t),
          hCommandList, hDstImage, srcptr, pDstRegion, hEvent)
end

@checked function zeCommandListAppendMemoryPrefetch(hCommandList, ptr, size)
    ccall((:zeCommandListAppendMemoryPrefetch, libze_loader), ze_result_t,
          (ze_command_list_handle_t, Ptr{Cvoid}, Csize_t),
          hCommandList, ptr, size)
end

@checked function zeCommandListAppendMemAdvise(hCommandList, hDevice, ptr, size, advice)
    ccall((:zeCommandListAppendMemAdvise, libze_loader), ze_result_t,
          (ze_command_list_handle_t, ze_device_handle_t, Ptr{Cvoid}, Csize_t,
           ze_memory_advice_t),
          hCommandList, hDevice, ptr, size, advice)
end
