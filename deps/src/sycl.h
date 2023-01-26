#pragma once

#include <stddef.h>

#include <level_zero/ze_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syclPlatform_st *syclPlatform_t;
int syclPlatformCreate(syclPlatform_t *obj, ze_driver_handle_t driver);
int syclPlatformDestroy(syclPlatform_t obj);

typedef struct syclDevice_st *syclDevice_t;
int syclDeviceCreate(syclDevice_t *obj, syclPlatform_t platform,
                     ze_device_handle_t device);
int syclDeviceDestroy(syclDevice_t obj);

typedef struct syclContext_st *syclContext_t;
int syclContextCreate(syclContext_t *obj, syclDevice_t *devices, size_t ndevices,
                      ze_context_handle_t context, int keep_ownership);
int syclContextDestroy(syclContext_t obj);

typedef struct syclQueue_st *syclQueue_t;
int syclQueueCreate(syclQueue_t *obj, syclContext_t context, syclDevice_t device,
                    ze_command_queue_handle_t queue, int keep_ownership);
int syclQueueDestroy(syclQueue_t obj);

typedef struct syclEvent_st *syclEvent_t;
int syclEventCreate(syclEvent_t *obj, syclContext_t context,
                    ze_event_handle_t event, int keep_ownership);
int syclEventDestroy(syclEvent_t obj);

#ifdef __cplusplus
}
#endif
