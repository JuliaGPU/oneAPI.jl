# Checked synchronous vendor operations. Automatic selection lives in GPUArrays.
const onemklSparseFloat = onemklFloat
const onemklSparseInt = Union{Int32,Int64}
_mkl_empty(A) = nnz(A)==0 || any(iszero,size(A))
csc_supported() = version() >= v"2025.3"
sparse_matrix(A::oneAbstractSparseMatrix) = A
sparse_matrix(P::PreparedSparse) = P.source
const SparseOperand = Union{oneAbstractSparseMatrix,PreparedSparse}

function _scale_output!(beta,y)
    iszero(beta) ? fill!(y,zero(eltype(y))) : (y .*= beta)
    y
end
_clear_ignored_output!(beta,y) = (iszero(beta) && fill!(y,zero(eltype(y))); y)
_empty_triangular_is_identity(diag,A) = diag=='U' || any(iszero,size(A))

function check_sparse_parameters(trans::Char,uplo::Char='L',diag::Char='N')
    trans in ('N','T','C') || throw(ArgumentError("invalid sparse transpose character"))
    uplo in ('L','U') || throw(ArgumentError("invalid sparse triangle character"))
    diag in ('N','U') || throw(ArgumentError("invalid sparse diagonal character"))
end
function check_sparse_dense(A,arrays...)
    all(X -> eltype(X)===eltype(A),arrays) ||
        throw(ArgumentError("oneMKL sparse requires matching dense and sparse value types"))
    all(arrays) do X
        stride(X,1)==1 && (ndims(X)==1 || stride(X,2)>=max(1,size(X,1)))
    end || throw(ArgumentError("oneMKL sparse requires contiguous vectors and column-major matrices"))
    all(X -> context(X)==context(A.nzVal) && device(X)==device(A.nzVal),arrays) ||
        throw(ArgumentError("oneMKL sparse operand placement differs"))
end

function with_sparse_execution(f,input::SparseOperand,operation,arrays...)
    A = sparse_matrix(input)
    check_sparse_dense(A,arrays...)
    run(owner) = lock(owner.lock) do
        check_sparse_owner(owner)
        f(owner)
    end
    if input isa PreparedSparse
        return lock(input.lock) do
            input.invalid && throw(ArgumentError("prepared sparse refresh failed during disposal; close and reconstruct it"))
            all(p -> get(operation,first(p),nothing)==last(p),pairs(input.operation)) ||
                throw(ArgumentError("prepared sparse operation parameters do not match"))
            sparse_storage_signature(input.source,_storage(input.source)) == input.source_signature ||
                throw(ArgumentError("prepared sparse source storage changed; close and prepare again"))
            lock(input.owner.lock) do
                check_sparse_owner(input.owner)
            end
            # Native handles may cache numerical data, even for a fixed pattern.
            # Never modify their bound snapshot. Unchanged values reuse analysis;
            # changed values require a new snapshot and native optimization.
            if !GPUArrays.same_value_bits(nonzeros(input.source),nonzeros(input.owner.matrix))
                candidate=prepare_sparse_owner(input.source,input.optimizer)
                try
                    close(input.owner)
                catch
                    input.invalid=true
                    try close(candidate) catch cleanup_error
                        @error "Sparse refresh cleanup failed" exception=(cleanup_error,catch_backtrace())
                    end
                    rethrow()
                end
                input.owner=candidate
            end
            run(input.owner)
        end
    end
    return with_sparse_owner(run,A)
end

# Explicit pointers avoid recording a different task's stream during ccall conversion.
sparse_pointer(X::oneArray) = oneAPI.device_pointer(X)
sparse_pointer(X::SubArray) = oneAPI.device_pointer(parent(X)) +
    Base._memory_offset(X.parent,map(first,X.indices)...)

for (T,prefix) in ((Float32,:S),(Float64,:D),(ComplexF32,:C),(ComplexF64,:Z)),
    operation in (:gemv,:gemm,:symv,:trmv,:trsv,:trsm)
    name = Symbol("onemkl",prefix,"sparse_",operation)
    @eval native_sparse(::Type{$T},::Val{$(QuoteNode(operation))},args...) =
        check_sparse_status($name(args...),$(QuoteNode(operation)))
end

function sparse_gemv!(trans::Char,alpha::Number,input::SparseOperand,
                      x::oneStridedVector,beta::Number,y::oneStridedVector)
    check_sparse_parameters(trans)
    A = sparse_matrix(input)
    alpha,beta = convert(eltype(A),alpha),convert(eltype(A),beta)
    GPUArrays.check_sparse_product(y,trans,'N',A,x)
    params = (;kind=:gemv,trans)
    with_sparse_execution(input,params,x,y) do owner
        _mkl_empty(A) && return _scale_output!(beta,y)
        csc = A isa oneSparseMatrixCSC
        conjugate = csc && trans=='C' && eltype(A)<:Complex
        rhs = conjugate ? conj.(x) : x
        out = conjugate ? (iszero(beta) ? similar(y) : conj.(y)) : y
        _clear_ignored_output!(beta,out)
        op = csc ? (trans=='N' ? 'T' : 'N') : trans
        a,b = conjugate ? (conj(alpha),conj(beta)) : (alpha,beta)
        sparse_transaction(owner,rhs,out) do handle,queue
            native_sparse(eltype(A),Val(:gemv),queue,op,a,handle,sparse_pointer(rhs),b,sparse_pointer(out))
        end
        conjugate && (y .= conj.(out))
        y
    end
end

function sparse_gemm!(trans::Char,transB::Char,alpha::Number,input::SparseOperand,
                      B::oneStridedMatrix,beta::Number,C::oneStridedMatrix)
    check_sparse_parameters(trans); check_sparse_parameters(transB)
    transB=='N' || throw(ArgumentError("oneMKL sparse GEMM requires a nontransposed dense operand"))
    A = sparse_matrix(input)
    alpha,beta = convert(eltype(A),alpha),convert(eltype(A),beta)
    A isa oneSparseMatrixCOO && throw(ArgumentError("oneMKL sparse GEMM requires CSR or CSC"))
    GPUArrays.check_sparse_product(C,trans,transB,A,B)
    params = (;kind=:gemm,trans,transB,nrhs=size(C,2))
    with_sparse_execution(input,params,B,C) do owner
        _mkl_empty(A) && return _scale_output!(beta,C)
        csc = A isa oneSparseMatrixCSC
        conjugate = csc && trans=='C' && eltype(A)<:Complex
        rhs = conjugate ? conj.(LinearAlgebra.wrap(B,transB)) : B
        out = conjugate ? (iszero(beta) ? similar(C) : conj.(C)) : C
        _clear_ignored_output!(beta,out)
        op = csc ? (trans=='N' ? 'T' : 'N') : trans
        opB = conjugate ? 'N' : transB
        a,b = conjugate ? (conj(alpha),conj(beta)) : (alpha,beta)
        sparse_transaction(owner,rhs,out) do handle,queue
            native_sparse(eltype(A),Val(:gemm),queue,'C',op,opB,a,handle,
                          sparse_pointer(rhs),size(C,2),max(1,stride(rhs,2)),b,
                          sparse_pointer(out),max(1,stride(out,2)))
        end
        conjugate && (C .= conj.(out))
        C
    end
end

function sparse_symv!(uplo::Char,alpha::Number,input::SparseOperand,
                      x::oneStridedVector,beta::Number,y::oneStridedVector)
    check_sparse_parameters('N',uplo)
    A = sparse_matrix(input); checksquare(A)
    alpha,beta = convert(eltype(A),alpha),convert(eltype(A),beta)
    A isa oneSparseMatrixCOO && throw(ArgumentError("oneMKL symmetric multiply requires CSR or CSC"))
    GPUArrays.check_sparse_product(y,'N','N',A,x)
    params=(;kind=:symv,uplo)
    with_sparse_execution(input,params,x,y) do owner
        _mkl_empty(A) && return _scale_output!(beta,y)
        _clear_ignored_output!(beta,y)
        native_uplo = A isa oneSparseMatrixCSC ? flip_uplo(uplo) : uplo
        sparse_transaction(owner,x,y) do handle,queue
            native_sparse(eltype(A),Val(:symv),queue,native_uplo,alpha,handle,
                          sparse_pointer(x),beta,sparse_pointer(y))
        end
        y
    end
end

function check_sparse_triangle(A,trans,uplo,diag)
    check_sparse_parameters(trans,uplo,diag)
    checksquare(A)
    !_mkl_empty(A) && (!(A isa oneSparseMatrixCSR) || trans!='N') &&
        throw(ArgumentError("oneMKL triangular sparse operations require nontransposed CSR"))
end

function sparse_trmv!(uplo::Char,trans::Char,diag::Char,alpha::Number,input::SparseOperand,
                      x::oneStridedVector,beta::Number,y::oneStridedVector)
    A=sparse_matrix(input)
    alpha = convert(eltype(A),alpha)
    check_sparse_triangle(A,trans,uplo,diag)
    GPUArrays.check_sparse_product(y,trans,'N',A,x)
    beta = convert(eltype(A),beta)
    params=(;kind=:trmv,uplo,trans,diag)
    with_sparse_execution(input,params,x,y) do owner
        if _mkl_empty(A)
            _scale_output!(beta,y)
            diag=='U' && (y .+= alpha .* x)
            return y
        end
        _clear_ignored_output!(beta,y)
        sparse_transaction(owner,x,y) do handle,queue
            native_sparse(eltype(A),Val(:trmv),queue,uplo,trans,diag,alpha,handle,
                          sparse_pointer(x),beta,sparse_pointer(y))
        end
        y
    end
end

function sparse_trsv!(uplo::Char,trans::Char,diag::Char,alpha::Number,input::SparseOperand,
                      x::oneStridedVector,y::oneStridedVector)
    A=sparse_matrix(input)
    alpha=convert(eltype(A),alpha)
    check_sparse_triangle(A,trans,uplo,diag)
    size(A,1)==length(x)==length(y) || throw(DimensionMismatch("triangular vector solve dimensions differ"))
    check_sparse_solve_alias(A,x,y)
    params=(;kind=:trsv,uplo,trans,diag)
    with_sparse_execution(input,params,x,y) do owner
        if _mkl_empty(A)
            _empty_triangular_is_identity(diag,A) || throw(SingularException(1))
            y .= alpha .* x
            return y
        end
        sparse_transaction(owner,x,y) do handle,queue
            native_sparse(eltype(A),Val(:trsv),queue,uplo,trans,diag,alpha,handle,
                          sparse_pointer(x),sparse_pointer(y))
        end
        y
    end
end

function sparse_trsm!(uplo::Char,trans::Char,transX::Char,diag::Char,alpha::Number,
                      input::SparseOperand,X::oneStridedMatrix,Y::oneStridedMatrix)
    A=sparse_matrix(input)
    alpha=convert(eltype(A),alpha)
    check_sparse_triangle(A,trans,uplo,diag); check_sparse_parameters(transX)
    shape = GPUArrays.op_size(X,transX)
    size(Y)==shape && size(Y,1)==size(A,1) || throw(DimensionMismatch("triangular matrix solve dimensions differ"))
    check_sparse_solve_alias(A,X,Y)
    params=(;kind=:trsm,uplo,trans,diag,nrhs=size(Y,2))
    with_sparse_execution(input,params,X,Y) do owner
        if _mkl_empty(A)
            _empty_triangular_is_identity(diag,A) || throw(SingularException(1))
            Y .= alpha .* LinearAlgebra.wrap(X,transX)
            return Y
        end
        # The vendor's transposed in-place RHS contract is not established. Snapshot
        # that input, preserving ordinary exact-alias nontransposed solves.
        rhs=X===Y && transX!='N' ? copy(X) : X
        sparse_transaction(owner,rhs,Y) do handle,queue
            native_sparse(eltype(A),Val(:trsm),queue,'C',trans,transX,uplo,diag,alpha,
                          handle,sparse_pointer(rhs),size(Y,2),max(1,stride(rhs,2)),
                          sparse_pointer(Y),max(1,stride(Y,2)))
        end
        Y
    end
end

function prepare_sparse_owner(A,optimize)
    owner=SparseHandleOwner(GPUArrays.snapshot_sparse_values(A))
    try
        if !_mkl_empty(A)
            sparse_transaction(owner) do handle,queue
                optimize(handle,queue)
            end
        end
        return owner
    catch
        try close(owner) catch cleanup_error
            @error "Sparse optimization cleanup failed" exception=(cleanup_error,catch_backtrace())
        end
        rethrow()
    end
end

function prepare_sparse(A,operation,optimize)
    owner=prepare_sparse_owner(A,optimize)
    PreparedSparse(owner,A,operation,optimize,sparse_storage_signature(A,_storage(A)),ReentrantLock(),false)
end

function sparse_optimize_gemv!(trans::Char,A::oneAbstractSparseMatrix)
    check_sparse_parameters(trans)
    op=A isa oneSparseMatrixCSC ? (trans=='N' ? 'T' : 'N') : trans
    prepare_sparse(A,(;kind=:gemv,trans),(h,q)->
        check_sparse_status(onemklXsparse_optimize_gemv(q,op,h),:optimize_gemv))
end
function sparse_optimize_gemm!(trans::Char,A::oneAbstractSparseMatrix)
    check_sparse_parameters(trans)
    A isa oneSparseMatrixCOO && throw(ArgumentError("oneMKL sparse GEMM requires CSR or CSC"))
    op=A isa oneSparseMatrixCSC ? (trans=='N' ? 'T' : 'N') : trans
    prepare_sparse(A,(;kind=:gemm,trans),(h,q)->
        check_sparse_status(onemklXsparse_optimize_gemm(q,op,h),:optimize_gemm))
end
function sparse_optimize_gemm!(trans::Char,transB::Char,nrhs::Int,A::oneAbstractSparseMatrix)
    transB=='N' || throw(ArgumentError("oneMKL sparse GEMM requires a nontransposed dense operand"))
    check_sparse_parameters(trans); check_sparse_parameters(transB)
    A isa oneSparseMatrixCOO && throw(ArgumentError("oneMKL sparse GEMM requires CSR or CSC"))
    nrhs>=0 || throw(ArgumentError("negative right-hand-side count"))
    op=A isa oneSparseMatrixCSC ? (trans=='N' ? 'T' : 'N') : trans
    nativeB=A isa oneSparseMatrixCSC && trans=='C' && eltype(A)<:Complex ? 'N' : transB
    prepare_sparse(A,(;kind=:gemm,trans,transB,nrhs),(h,q)->
        check_sparse_status(onemklXsparse_optimize_gemm_advanced(q,'C',op,nativeB,h,nrhs),:optimize_gemm))
end
for kind in (:trmv,:trsv,:trsm)
    name=Symbol("sparse_optimize_",kind,"!")
    native=Symbol("onemklXsparse_optimize_",kind)
    @eval function $name(uplo::Char,trans::Char,diag::Char,A::oneAbstractSparseMatrix)
        check_sparse_triangle(A,trans,uplo,diag)
        prepare_sparse(A,(;kind=$(QuoteNode(kind)),uplo,trans,diag),(h,q)->
            check_sparse_status($native(q,uplo,trans,diag,h),$(QuoteNode(native))))
    end
end
function sparse_optimize_trsm!(uplo::Char,trans::Char,diag::Char,nrhs::Int,A::oneAbstractSparseMatrix)
    check_sparse_triangle(A,trans,uplo,diag)
    nrhs>=0 || throw(ArgumentError("negative right-hand-side count"))
    prepare_sparse(A,(;kind=:trsm,uplo,trans,diag,nrhs),(h,q)->
        check_sparse_status(onemklXsparse_optimize_trsm_advanced(q,'C',uplo,trans,diag,h,nrhs),:optimize_trsm))
end

function check_sparse_solve_alias(A,X,Y)
    any(b -> Base.mightalias(Y,b),_storage(A)) &&
        throw(ArgumentError("sparse solve output must not overlap sparse component storage"))
    X !== Y && Base.mightalias(X,Y) &&
        throw(ArgumentError("sparse solve permits identical RHS/output, not partial overlap"))
    nothing
end
