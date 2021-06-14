# Atomic Functions

# "atomic operations on 32-bit signed, unsigned integers and single precision
#  floating-point to locations in __global or __local memory"

const atomic_integer_types = [UInt32, Int32]
# TODO: 64-bit atomics with ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS
# TODO: additional floating-point atomics with ZE_extension_float_atomics
const atomic_memory_types = [AS.Local, AS.Global]


# generically typed

for gentype in atomic_integer_types, as in atomic_memory_types
@eval begin

@device_function atomic_add!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_add", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_sub!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_sub", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_inc!(p::LLVMPtr{$gentype,$as}) =
    @builtin_ccall("atomic_inc", $gentype, (LLVMPtr{$gentype,$as},), p)

@device_function atomic_dec!(p::LLVMPtr{$gentype,$as}) =
    @builtin_ccall("atomic_dec", $gentype, (LLVMPtr{$gentype,$as},), p)

@device_function atomic_min!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_min", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_max!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_max", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_and!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_and", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_or!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_or", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_xor!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_xor", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_xchg!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("atomic_xchg", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype), p, val)

@device_function atomic_cmpxchg!(p::LLVMPtr{$gentype,$as}, cmp::$gentype, val::$gentype) =
    @builtin_ccall("atomic_cmpxchg", $gentype,
                   (LLVMPtr{$gentype,$as}, $gentype, $gentype), p, cmp, val)

end
end


# specifically typed

@device_function atomic_xchg!(p::LLVMPtr{Float32,AS.Local}, val::Float32) =
    @builtin_ccall("atomic_xchg", Float32, (LLVMPtr{Float32,AS.Local}, Float32,), p, val)
@device_function atomic_xchg!(p::LLVMPtr{Float32,AS.Global}, val::Float32) =
    @builtin_ccall("atomic_xchg", Float32, (LLVMPtr{Float32,AS.Global}, Float32,), p, val)


# documentation

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old + val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_add!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old - val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_sub!

"""
Swaps the old value stored at location `p` with new value given by `val`.
Returns old value.
"""
atomic_xchg!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute (`old` + 1) and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_inc!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute (`old` - 1) and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_dec!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `(old == cmp) ? val : old` and store result at location pointed by `p`.
The function returns `old`.
"""
atomic_cmpxchg!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `min(old, val)` and store minimum value at location pointed by `p`. The
function returns `old`.
"""
atomic_min!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `max(old, val)` and store maximum value at location pointed by `p`. The
function returns `old`.
"""
atomic_max

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old & val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_and!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old | val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_or!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old ^ val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_xor!
