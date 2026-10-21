# Sparse descriptors borrow buffers; their owner retains the complete execution domain.
# The support ABI completes bind/optimize/execute/release synchronously.
mutable struct SparseHandleOwner{A,B,Q,S,C,D}
    handle::Union{Nothing,matrix_handle_t}
    matrix::A
    buffers::B
    queue::Q
    stream::S
    context::C
    device::D
    task::Task
    signature::Tuple
    lock::ReentrantLock
    closed::Bool
    uncertain::Bool
end

"""
    PreparedSparse

An explicitly prepared oneMKL operation. It borrows fixed sparse storage: do not edit
indices, resize, rebind or free any component until `close` completes. Values may be
updated in place between completed executions. Reuse is limited to the originating
task, stream, context and device and the recorded operation parameters. Native analysis
binds an independent numerical snapshot. Each execution compares current values bitwise
on the device, requiring O(nnz) work and a host-visible decision. Unchanged values reuse
analysis; changed values allocate a new snapshot and rebuild analysis before execution.
Preparation does not make a sparse array itself stateful. Failed disposal does not lift
the borrowing restrictions; uncertain native release retains the resource in quarantine.
"""
mutable struct PreparedSparse{O,A,P,F}
    owner::O
    source::A
    operation::P
    optimizer::F
    source_signature::Tuple
    lock::ReentrantLock
    invalid::Bool
end
Base.close(P::PreparedSparse) = lock(P.lock) do
    close(P.owner)
end

const _deferred_sparse_owners = Any[]
const _deferred_sparse_owners_lock = ReentrantLock()
const _quarantined_sparse_owners = Any[]

function defer_sparse_owner!(owner)
    (owner.closed || owner.uncertain) && return nothing
    lock(_deferred_sparse_owners_lock) do
        push!(_deferred_sparse_owners,owner)
    end
    return nothing
end

check_sparse_status(status, operation) =
    iszero(status) ? nothing : error("oneMKL sparse $operation failed (support status $status)")

function sparse_storage_signature(A,buffers)
    (size(A), map(b -> (objectid(b),length(b),UInt(oneAPI.device_pointer(b))),buffers))
end

function check_sparse_owner(owner; execution=true)
    owner.closed && throw(ArgumentError("sparse handle is closed"))
    owner.uncertain && throw(ArgumentError("sparse handle release has an uncertain outcome"))
    if execution
        current_task() === owner.task && context() == owner.context &&
            device() == owner.device &&
            oneAPI.global_stream(owner.context,owner.device) === owner.stream ||
            throw(ArgumentError("prepared sparse execution requires its originating task, stream, context and device"))
        current = _storage(owner.matrix)
        all(pair -> pair[1] === pair[2],zip(current,owner.buffers)) &&
            sparse_storage_signature(owner.matrix,current) == owner.signature ||
            throw(ArgumentError("prepared sparse storage changed; close and prepare again"))
    end
    return nothing
end

function sparse_transaction(f,owner,arrays...;execution=true)
    lock(owner.lock) do
        check_sparse_owner(owner;execution)
        all(A -> context(A) == owner.context && device(A) == owner.device,arrays) ||
            throw(ArgumentError("oneMKL sparse operands require the same context and device"))
        roots = map(A -> A isa SubArray ? parent(A) : A,(owner.buffers...,arrays...))
        managed = oneAPI.Managed[A.data[] for A in roots]
        oneAPI.with_ownership(owner.stream,managed) do
            lock(owner.stream.lock) do
                oneAPI.mkl_boundary!(owner.stream)
                GC.@preserve owner arrays f(owner.handle,owner.queue)
            end
        end
    end
end

function Base.close(owner::SparseHandleOwner)
    lock(owner.lock) do
        owner.closed && return nothing
        owner.uncertain && throw(ArgumentError("sparse handle release has an uncertain outcome; it cannot be retried"))
        if owner.handle === nothing
            owner.closed = true
            return nothing
        end
        managed = oneAPI.Managed[A.data[] for A in owner.buffers]
        oneAPI.with_ownership(owner.stream,managed) do
            lock(owner.stream.lock) do
                # Disposal can run in a foreign task. Drain producers without changing
                # the originating task's mkl_dirty bookkeeping or obtaining its TLS.
                oneAPI.drain(owner.stream)
                handle = Ref(owner.handle)
                owner.uncertain = true
                status = try
                    GC.@preserve owner onemklXsparse_release_matrix_handle(owner.queue,handle)
                catch
                    lock(_deferred_sparse_owners_lock) do
                        push!(_quarantined_sparse_owners,owner)
                    end
                    rethrow()
                end
                if !iszero(status)
                    lock(_deferred_sparse_owners_lock) do
                        push!(_quarantined_sparse_owners,owner)
                    end
                    check_sparse_status(status,:release)
                end
                owner.handle = nothing
                owner.closed = true
                owner.uncertain = false
            end
        end
    end
    return nothing
end

function flush_deferred_sparse_owners()
    owners = lock(_deferred_sparse_owners_lock) do
        pending = copy(_deferred_sparse_owners)
        empty!(_deferred_sparse_owners)
        pending
    end
    for owner in owners
        try
            close(owner)
        catch err
            defer_sparse_owner!(owner)
            @error "Sparse handle cleanup failed" exception=(err,catch_backtrace())
        end
    end
    return nothing
end

function SparseHandleOwner(A::oneAbstractSparseMatrix)
    Support._check_sparse_abi()
    T = eltype(A); Ti = SparseArrays.indtype(A)
    T <: onemklFloat && Ti in (Int32,Int64) ||
        throw(ArgumentError("oneMKL sparse requires BLAS floating values and Int32/Int64 indices"))
    buffers = _storage(A)
    eltype(first(buffers)) === Ti ||
        throw(ArgumentError("oneMKL sparse requires uniform pointer and coordinate widths"))
    ctx,dev = context(last(buffers)),device(last(buffers))
    all(b -> context(b)==ctx && device(b)==dev,buffers) ||
        throw(ArgumentError("sparse component placement differs"))
    context()==ctx && device()==dev ||
        throw(ArgumentError("sparse preparation requires the matrix context and device to be current"))
    flush_deferred_sparse_owners()
    stream = oneAPI.global_stream(ctx,dev)
    queue = sycl_queue(global_queue(ctx,dev))
    owner = SparseHandleOwner(nothing,A,buffers,queue,stream,ctx,dev,current_task(),
                              sparse_storage_signature(A,buffers),ReentrantLock(),false,false)
    finalizer(defer_sparse_owner!,owner)
    if !_mkl_empty(A)
        try
            handle = Ref{matrix_handle_t}(C_NULL)
            status = onemklXsparse_init_matrix_handle(handle)
            handle[] == C_NULL || (owner.handle = handle[])
            check_sparse_status(status,:initialize)
            sparse_transaction(owner) do native,queue
                bind_sparse_owner!(queue,native,A)
            end
        catch
            try
                close(owner)
            catch cleanup_error
                @error "Sparse construction cleanup failed" exception=(cleanup_error,catch_backtrace())
            end
            rethrow()
        end
    end
    return owner
end

function with_sparse_owner(f,A::oneAbstractSparseMatrix)
    owner = SparseHandleOwner(A)
    try
        result = f(owner)
        close(owner)
        return result
    catch
        try
            close(owner)
        catch cleanup_error
            @error "Sparse execution cleanup failed" exception=(cleanup_error,catch_backtrace())
        end
        rethrow()
    end
end

for (T,prefix) in ((Float32,:S),(Float64,:D),(ComplexF32,:C),(ComplexF64,:Z)), Ti in (Int32,Int64)
    suffix = Ti === Int64 ? "_64" : ""
    csr = Symbol("onemkl",prefix,"sparse_set_csr_data",suffix)
    coo = Symbol("onemkl",prefix,"sparse_set_coo_data",suffix)
    @eval begin
        function bind_sparse_owner!(queue,handle,A::Union{oneSparseMatrixCSR{$T,$Ti},oneSparseMatrixCSC{$T,$Ti}})
            m,n = A isa oneSparseMatrixCSC ? reverse(size(A)) : size(A)
            p,i,v = _storage(A)
            check_sparse_status($csr(queue,handle,m,n,nnz(A),'O',
                                oneAPI.device_pointer(p),oneAPI.device_pointer(i),oneAPI.device_pointer(v)),:bind)
        end
        function bind_sparse_owner!(queue,handle,A::oneSparseMatrixCOO{$T,$Ti})
            m,n = size(A); p,i,v = _storage(A)
            check_sparse_status($coo(queue,handle,m,n,nnz(A),'O',
                                oneAPI.device_pointer(p),oneAPI.device_pointer(i),oneAPI.device_pointer(v)),:bind)
        end
    end
end
