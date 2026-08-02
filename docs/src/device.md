# Device Intrinsics

When writing custom kernels, you have access to a set of device intrinsics that map to underlying hardware instructions. These are provided by [SPIRVIntrinsics.jl](https://github.com/JuliaGPU/SPIRVIntrinsics.jl) and re-exported by oneAPI.jl.

## Indexing

These functions allow you to determine the current thread's position in the execution grid.

- `get_global_id(dim=1)`: Global index of the work item.
- `get_local_id(dim=1)`: Local index of the work item within the workgroup.
- `get_group_id(dim=1)`: Index of the workgroup.
- `get_global_size(dim=1)`: Global size of the ND-range.
- `get_local_size(dim=1)`: Size of the workgroup.
- `get_num_groups(dim=1)`: Number of workgroups.

Unlike their OpenCL counterparts, both the dimension argument and the returned indices are
1-based, so the result can be used to index a Julia array directly.

## Synchronization

- `barrier(flags)`: Synchronizes all work items in a workgroup. The `flags` argument selects
  which memory operations are fenced: `oneAPI.LOCAL_MEM_FENCE`, `oneAPI.GLOBAL_MEM_FENCE`, or
  their bitwise or. There is no default; a value must be passed.

## Atomics

Atomic operations are supported for thread-safe updates to memory.

- `atomic_add!(ptr, val)`
- `atomic_sub!(ptr, val)`
- `atomic_inc!(ptr)`
- `atomic_dec!(ptr)`
- `atomic_min!(ptr, val)`
- `atomic_max!(ptr, val)`
- `atomic_and!(ptr, val)`
- `atomic_or!(ptr, val)`
- `atomic_xor!(ptr, val)`
- `atomic_cmpxchg!(ptr, cmp, val)`

Supported types for atomics generally include `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, and `Float64`.

## Math Functions

Standard math functions from Julia's `Base` are supported within kernels (e.g., `sin`, `cos`, `exp`, `sqrt`).

