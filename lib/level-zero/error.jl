# Error type and decoding functionality

export ZeError


struct ZeError <: Exception
    code::ze_result_t
end

Base.convert(::Type{ze_result_t}, err::ZeError) = err.code

Base.showerror(io::IO, err::ZeError) =
    print(io, "ZeError: ", description(err), " (code $(reinterpret(Int32, err.code)), $(name(err)))")

Base.show(io::IO, ::MIME"text/plain", err::ZeError) = print(io, "ZeError($(err.code))")

name(err::ZeError) = string(err.code)

## COV_EXCL_START
function description(err::ZeError)
    if err.code == RESULT_SUCCESS
        "success"
    elseif err.code == RESULT_NOT_READY
        "synchronization primitive not signaled"
    elseif err.code == RESULT_ERROR_DEVICE_LOST
        "device hung, reset, was removed, or driver update occurred"
    elseif err.code == RESULT_ERROR_OUT_OF_HOST_MEMORY
        "insufficient host memory to satisfy call"
    elseif err.code == RESULT_ERROR_OUT_OF_DEVICE_MEMORY
        "insufficient device memory to satisfy call"
    elseif err.code == RESULT_ERROR_MODULE_BUILD_FAILURE
        "error occurred when building module, see build log for details"
    elseif err.code == RESULT_ERROR_INSUFFICIENT_PERMISSIONS
        "access denied due to permission level"
    elseif err.code == RESULT_ERROR_NOT_AVAILABLE
        "resource already in use and simultaneous access not allowed"
    elseif err.code == RESULT_ERROR_UNINITIALIZED
        "driver is not initialized"
    elseif err.code == RESULT_ERROR_UNSUPPORTED_VERSION
        "generic error code for unsupported versions"
    elseif err.code == RESULT_ERROR_UNSUPPORTED_FEATURE
        "generic error code for unsupported features"
    elseif err.code == RESULT_ERROR_INVALID_ARGUMENT
        "generic error code for invalid arguments"
    elseif err.code == RESULT_ERROR_INVALID_NULL_HANDLE
        "handle argument is not valid"
    elseif err.code == RESULT_ERROR_HANDLE_OBJECT_IN_USE
        "object pointed to by handle still in-use by device"
    elseif err.code == RESULT_ERROR_INVALID_NULL_POINTER
        "pointer argument may not be nullptr"
    elseif err.code == RESULT_ERROR_INVALID_SIZE
        "size argument is invalid (e.g., must not be zero)"
    elseif err.code == RESULT_ERROR_UNSUPPORTED_SIZE
        "size argument is not supported by the device (e.g., too large)"
    elseif err.code == RESULT_ERROR_UNSUPPORTED_ALIGNMENT
        "alignment argument is not supported by the device (e.g., too small)"
    elseif err.code == RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
        "synchronization object in invalid state"
    elseif err.code == RESULT_ERROR_INVALID_ENUMERATION
        "enumerator argument is not valid"
    elseif err.code == RESULT_ERROR_UNSUPPORTED_ENUMERATION
        "enumerator argument is not supported by the device"
    elseif err.code == RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT
        "image format is not supported by the device"
    elseif err.code == RESULT_ERROR_INVALID_NATIVE_BINARY
        "native binary is not supported by the device"
    elseif err.code == RESULT_ERROR_INVALID_GLOBAL_NAME
        "global variable is not found in the module"
    elseif err.code == RESULT_ERROR_INVALID_KERNEL_NAME
        "kernel name is not found in the module"
    elseif err.code == RESULT_ERROR_INVALID_FUNCTION_NAME
        "function name is not found in the module"
    elseif err.code == RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION
        "group size dimension is not valid for the kernel or device"
    elseif err.code == RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION
        "global width dimension is not valid for the kernel or device"
    elseif err.code == RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX
        "kernel argument index is not valid for kernel"
    elseif err.code == RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE
        "kernel argument size does not match kernel"
    elseif err.code == RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE
        "value of kernel attribute is not valid for the kernel or device"
    elseif err.code == RESULT_ERROR_INVALID_COMMAND_LIST_TYPE
        "command list type does not match command queue type"
    elseif err.code == RESULT_ERROR_OVERLAPPING_REGIONS
        "copy operations do not support overlapping regions of memory"
    elseif err.code == RESULT_ERROR_UNKNOWN
        "unknown or internal error"
    else
        "no description for this error"
    end
end
## COV_EXCL_STOP

@enum_without_prefix _ze_result_t ZE_


## API call wrapper

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(ZeError(res))
end
