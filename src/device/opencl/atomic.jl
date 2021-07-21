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



#
# High-level interface
#

# prototype of a high-level interface for performing atomic operations on arrays
#
# this design could be generalized by having atomic {field,array}{set,ref} accessors, as
# well as acquire/release operations to implement the fallback functionality where any
# operation can be applied atomically.

const inplace_ops = Dict(
    :(+=) => :(+),
    :(-=) => :(-),
    :(*=) => :(*),
    :(/=) => :(/),
    :(&=) => :(&),
    :(|=) => :(|),
    :(⊻=) => :(⊻)
)

struct AtomicError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, err::AtomicError) =
    print(io, "AtomicError: ", err.msg)

"""
    @atomic a[I] = op(a[I], val)
    @atomic a[I] ...= val

Atomically perform a sequence of operations that loads an array element `a[I]`, performs the
operation `op` on that value and a second value `val`, and writes the result back to the
array. This sequence can be written out as a regular assignment, in which case the same
array element should be used in the left and right hand side of the assignment, or as an
in-place application of a known operator. In both cases, the array reference should be pure
and not induce any side-effects.

!!! warn
    This interface is experimental, and might change without warning.  Use the lower-level
    `atomic_...!` functions for a stable API.
"""
macro atomic(ex)
    # decode assignment and call
    if ex.head == :(=)
        ref = ex.args[1]
        rhs = ex.args[2]
        Meta.isexpr(rhs, :call) || throw(AtomicError("right-hand side of an @atomic assignment should be a call"))
        op = rhs.args[1]
        if rhs.args[2] != ref
            throw(AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side"))
        end
        val = rhs.args[3]
    elseif haskey(inplace_ops, ex.head)
        op = inplace_ops[ex.head]
        ref = ex.args[1]
        val = ex.args[2]
    else
        throw(AtomicError("unknown @atomic expression"))
    end

    # decode array expression
    Meta.isexpr(ref, :ref) || throw(AtomicError("@atomic should be applied to an array reference expression"))
    array = ref.args[1]
    indices = Expr(:tuple, ref.args[2:end]...)

    esc(quote
        $atomic_arrayset($array, $indices, $op, $val)
    end)
end

# FIXME: make this respect the indexing style
@inline atomic_arrayset(A::AbstractArray{T}, Is::Tuple, op::Function, val) where {T} =
    atomic_arrayset(A, Base._to_linear_index(A, Is...), op, convert(T, val))

function atomic_arrayset(A::AbstractArray, I::Integer, op::Function, val)
    error("Don't know how to atomically perform $op on $(typeof(A))")
    # TODO: while { acquire, op, cmpxchg }
end

# oneAPI.jl atomics
for (op,impl) in [(+)      => atomic_add!,
                  (-)      => atomic_sub!,
                  (&)      => atomic_and!,
                  (|)      => atomic_or!,
                  (⊻)      => atomic_xor!,
                  Base.max => atomic_max!,
                  Base.min => atomic_min!]
    @eval @inline atomic_arrayset(A::oneDeviceArray, I::Integer, ::typeof($op), val) =
        $impl(pointer(A, I), val)
end
