# Level Zero Interface

The `oneL0` submodule provides low-level access to the Level Zero API, which gives you fine-grained control over the hardware.

## Drivers and Devices

You can enumerate available drivers and devices:

```julia
using oneAPI.oneL0

# Get available drivers
drvs = drivers()

# Get devices for a driver
devs = devices(first(drvs))

# Inspect device properties
props = compute_properties(first(devs))
println("Max workgroup size: ", props.maxTotalGroupSize)
```

## Contexts and Queues

Manage contexts and command queues for executing operations:

```julia
# Create a context
ctx = ZeContext(first(drvs))

# Create a command queue
queue = ZeCommandQueue(ctx, first(devs))

# Execute a command list
execute!(queue) do list
    append_barrier!(list)
end
```

## Memory Operations

You can perform low-level memory operations using command lists:

```julia
execute!(queue) do list
    append_copy!(list, dst_ptr, src_ptr, size)
end
```

