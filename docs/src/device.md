# Device Intrinsics

When writing custom kernels, you have access to a set of device intrinsics that map to underlying hardware instructions.

## Indexing

These functions allow you to determine the current thread's position in the execution grid.

- `get_global_id(dim=0)`: Global index of the work item.
- `get_local_id(dim=0)`: Local index of the work item within the workgroup.
- `get_group_id(dim=0)`: Index of the workgroup.
- `get_global_size(dim=0)`: Global size of the ND-range.
- `get_local_size(dim=0)`: Size of the workgroup.
- `get_num_groups(dim=0)`: Number of workgroups.

## Synchronization

- `barrier(flags=0)`: Synchronizes all work items in a workgroup.

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

