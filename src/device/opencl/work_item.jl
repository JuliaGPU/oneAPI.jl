# Work-Item Functions

export get_work_dim,
       get_global_size, get_global_id,
       get_local_size, get_enqueued_local_size, get_local_id,
       get_num_groups, get_group_id,
       get_global_offset,
       get_global_linear_id, get_local_linear_id

# NOTE: these functions now unsafely truncate to Int to avoid top bit checks.
#       we should probably use range metadata instead.

# TODO: 1-indexed dimension selection?

get_work_dim() = @builtin_ccall("get_work_dim", Cuint, ()) % Int

get_global_size(dimindx::Integer=0) = @builtin_ccall("get_global_size", Csize_t, (Cuint,), dimindx) % Int
get_global_id(dimindx::Integer=0) = @builtin_ccall("get_global_id", Csize_t, (Cuint,), dimindx) % Int + 1

get_local_size(dimindx::Integer=0) = @builtin_ccall("get_local_size", Csize_t, (Cuint,), dimindx) % Int
get_enqueued_local_size(dimindx::Integer=0) = @builtin_ccall("get_enqueued_local_size", Csize_t, (Cuint,), dimindx) % Int
get_local_id(dimindx::Integer=0) = @builtin_ccall("get_local_id", Csize_t, (Cuint,), dimindx) % Int + 1

get_num_groups(dimindx::Integer=0) = @builtin_ccall("get_num_groups", Csize_t, (Cuint,), dimindx) % Int
get_group_id(dimindx::Integer=0) = @builtin_ccall("get_group_id", Csize_t, (Cuint,), dimindx) % Int + 1

get_global_offset(dimindx::Integer=0) = @builtin_ccall("get_global_offset", Csize_t, (Cuint,), dimindx) % Int + 1

get_global_linear_id() = @builtin_ccall("get_global_linear_id", Csize_t, ()) % Int + 1
get_local_linear_id() = @builtin_ccall("get_local_linear_id", Csize_t, ()) % Int + 1
