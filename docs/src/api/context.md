# Context and Device Management

This page documents the API for managing Level Zero drivers, devices, and contexts in oneAPI.jl.

## Overview

oneAPI.jl uses task-local state to manage GPU resources. This allows different Julia tasks to
work with different drivers, devices, or contexts without interfering with each other.

The typical hierarchy is:
- **Driver**: Represents a Level Zero driver (usually one per GPU vendor/installation)
- **Device**: Represents a physical GPU device
- **Context**: Manages resources like memory allocations and command queues

## Driver Management

### `driver() -> ZeDriver`

Get the current Level Zero driver for the calling task. If no driver has been explicitly
set with `driver!`, returns the first available driver. The driver selection is task-local.

### `driver!(drv::ZeDriver)`

Set the current Level Zero driver for the calling task. This also clears the current
device selection, as devices are associated with specific drivers.

### `drivers() -> Vector{ZeDriver}`

Return a list of all available Level Zero drivers.

## Device Management

### `device() -> ZeDevice`

Get the current Level Zero device for the calling task. If no device has been explicitly
set with `device!`, returns the first available device for the current driver. The device
selection is task-local.

### `device!(dev::ZeDevice)` / `device!(i::Int)`

Set the current Level Zero device for the calling task. Can pass either a device object or
a 1-based device index.

### `devices() -> Vector{ZeDevice}` / `devices(drv::ZeDriver)`

Return a list of available Level Zero devices. Without arguments, returns devices for
the current driver.

## Context Management

### `context() -> ZeContext`

Get the current Level Zero context for the calling task. If no context has been explicitly
set with `context!`, returns a global context for the current driver. Contexts manage the
lifetime of resources like memory allocations and command queues.

### `context!(ctx::ZeContext)`

Set the current Level Zero context for the calling task.

## Command Queues

### `global_queue(ctx::ZeContext, dev::ZeDevice) -> ZeCommandQueue`

Get the global command queue for the given context and device. This queue is used as the
default queue for executing operations. The queue is created with in-order execution flags.

### `synchronize()`

Block the host thread until all operations on the global command queue for the current
context and device have completed.


## Example Workflow

```julia
using oneAPI

# List available drivers
drv_list = drivers()
println("Available drivers: ", length(drv_list))

# Select a specific driver
driver!(drv_list[1])

# List devices for current driver
dev_list = devices()
println("Available devices: ", length(dev_list))

# Select a specific device
device!(dev_list[1])

# Get the current context (created automatically)
ctx = context()

# Perform GPU operations...
a = oneArray(rand(Float32, 100))

# Wait for all operations to complete
synchronize()
```

## Multi-Device Programming

You can use different devices in different Julia tasks:

```julia
using oneAPI

# Task 1: Use first device
Threads.@spawn begin
    device!(1)
    a = oneArray(rand(Float32, 100))
    # ... operations on device 1 ...
end

# Task 2: Use second device
Threads.@spawn begin
    device!(2)
    b = oneArray(rand(Float32, 100))
    # ... operations on device 2 ...
end
```
