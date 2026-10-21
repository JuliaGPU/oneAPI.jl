using oneAPI,GPUArrays,SparseArrays,LinearAlgebra,Test
using oneAPI.oneMKL: oneSparseMatrixCSR

if haskey(ENV,"GPUARRAYS_V13_STACK_SUPPORT")
    @testset "stack support native exception containment" begin
        expected=ENV["GPUARRAYS_V13_STACK_SUPPORT"]
        @test realpath(oneAPI.Support.liboneapi_support)==realpath(expected)
        # Stop before the native probe if an artifact override was not applied:
        # the old artifact aborts instead of reporting this unsupported operation.
        realpath(oneAPI.Support.liboneapi_support)==realpath(expected) || error("wrong support library")
        A=oneSparseMatrixCSR(sparse(Float32[2 0;1 3]))
        B=oneArray(Float32[1 2;3 4]); C=similar(B)
        owner=oneMKL.SparseHandleOwner(A)
        try
            @test_throws ErrorException oneMKL.sparse_transaction(owner,B,C) do h,q
                oneMKL.native_sparse(Float32,Val(:gemm),q,'C','N','T',1f0,h,
                    oneMKL.sparse_pointer(B),2,2,0f0,oneMKL.sparse_pointer(C),2)
            end
        finally
            close(owner)
        end
        @test owner.closed
    end
end

@testset "unsupported vendor layout preflight" begin
    A=oneSparseMatrixCSR(sparse(Float32[2 0;1 3]))
    B=oneArray(Float32[1 2;3 4]); C=oneArray(fill(7f0,2,2))
    for trans in ('T','C')
        @test_throws ArgumentError GPUArrays.spmm!(C,'N',trans,A,B,1f0,0f0;algorithm=GPUArrays.Vendor())
        @test Array(C)==fill(7f0,2,2)
        @test_throws ArgumentError oneMKL.sparse_gemm!('N',trans,1f0,A,B,0f0,C)
        @test Array(C)==fill(7f0,2,2)
        @test_throws ArgumentError oneMKL.sparse_optimize_gemm!('N',trans,2,A)
    end
end

@testset "explicit sparse resource lifetime" begin
    host=sparse(Float32[2 0 3; 0 4 0; 1 0 5])
    A=oneSparseMatrixCSR(host)
    x=oneArray(Float32[1,2,3]); y=similar(x)
    original=copy(nonzeros(A))
    P=oneMKL.sparse_optimize_gemv!('N',A)
    try
        for iteration in 1:20
            nonzeros(A) .= original .* Float32(iteration)
            GC.gc()
            oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
            @test Array(y) ≈ Float32(iteration).*(host*Array(x))
        end
        before=Array(y)
        @test_throws ArgumentError oneMKL.sparse_gemv!('T',1f0,P,x,0f0,y)
        @test Array(y)==before
        result=fetch(@async try
            oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
            nothing
        catch err
            err
        end)
        @test result isa ArgumentError
        @test Array(y)==before
    finally
        fetch(@async close(P))
    end
    @test P.owner.closed
    close(P)
    @test_throws ArgumentError oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)

    P=oneMKL.sparse_optimize_gemv!('N',A)
    native=P.owner.handle
    stream=P.owner.stream
    saved_error=stream.error
    try
        lock(stream.lock) do
            stream.error=ErrorException("injected pre-release failure")
        end
        @test_throws ErrorException close(P)
        @test !P.owner.closed
        @test !P.owner.uncertain
        @test P.owner.handle==native
    finally
        lock(stream.lock) do
            stream.error=saved_error
        end
        close(P)
    end

    P=oneMKL.sparse_optimize_gemv!('N',A)
    finalize(P.owner)
    @test !P.owner.closed
    fetch(@async oneMKL.flush_deferred_sparse_owners())
    @test P.owner.closed

    # Diagnose signature mismatch without actually invalidating a live native binding.
    P=oneMKL.sparse_optimize_gemv!('N',A)
    before=Array(y)
    signature=P.owner.signature
    P.owner.signature=(size(A),())
    try
        @test_throws ArgumentError oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
        @test Array(y)==before
    finally
        P.owner.signature=signature
        close(P)
    end
end

@testset "prepared operation value refresh" begin
    host=sparse(ComplexF32[2 0 3im;0 4 0;1+im 0 5])
    x=oneArray(ComplexF32[1,2,3]); B=oneArray(reshape(ComplexF32.(1:6),3,2))
    for S in (oneMKL.oneSparseMatrixCSR,oneMKL.oneSparseMatrixCSC), trans in ('N','T','C')
        A=S(host)
        P=oneMKL.sparse_optimize_gemv!(trans,A)
        y=similar(x)
        original=copy(nonzeros(A))
        try
            for scale in (1f0,2f0)
                nonzeros(A) .= scale.*original
                oneMKL.sparse_gemv!(trans,1f0,P,x,0f0,y)
                @test Array(y) ≈ scale.*(LinearAlgebra.wrap(host,trans)*Array(x))
            end
        finally
            close(P)
        end
        A=S(host)
        P=oneMKL.sparse_optimize_gemm!(trans,'N',2,A)
        C=similar(B)
        original=copy(nonzeros(A))
        try
            for scale in (1f0,2f0)
                nonzeros(A) .= scale.*original
                oneMKL.sparse_gemm!(trans,'N',1f0,P,B,0f0,C)
                @test Array(C) ≈ scale.*(LinearAlgebra.wrap(host,trans)*Array(B))
            end
        finally
            close(P)
        end
    end
    host=sparse(Float32[2 0 0;1 4 0;3 1 5])
    A=oneMKL.oneSparseMatrixCSR(host)
    x=oneArray(Float32[1,2,3]); B=oneArray(reshape(Float32.(1:6),3,2))
    for kind in (:trsv,:trsm)
        original=copy(nonzeros(A))
        P=kind===:trsv ? oneMKL.sparse_optimize_trsv!('L','N','N',A) :
                        oneMKL.sparse_optimize_trsm!('L','N','N',2,A)
        try
            for scale in (1f0,2f0)
                nonzeros(A) .= scale.*original
                if kind===:trsv
                    y=similar(x)
                    oneMKL.sparse_trsv!('L','N','N',1f0,P,x,y)
                    @test Array(y) ≈ LowerTriangular(scale.*host)\Array(x)
                else
                    C=similar(B)
                    oneMKL.sparse_trsm!('L','N','N','N',1f0,P,B,C)
                    @test Array(C) ≈ LowerTriangular(scale.*host)\Array(B)
                end
            end
        finally
            close(P)
            copyto!(nonzeros(A),original)
        end
    end
end

@testset "prepared numerical snapshots" begin
    host=sparse(Float32[2 0 3; 0 4 0; 1 0 5])
    x=oneArray(Float32[1,2,3]); y=similar(x)
    for S in (oneMKL.oneSparseMatrixCSR,oneMKL.oneSparseMatrixCSC,oneMKL.oneSparseMatrixCOO)
        A=S(host)
        P=oneMKL.sparse_optimize_gemv!('N',A)
        original=P.owner
        try
            @test !Base.mightalias(nonzeros(A),nonzeros(P.owner.matrix))
            @test all(a === b for (a,b) in zip(Base.front(GPUArrays.sparse_buffers(A)),
                                               Base.front(GPUArrays.sparse_buffers(P.owner.matrix))))
            for iteration in 1:3
                oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
                @test P.owner === original
                @test Array(y) ≈ host*Array(x)
            end
            nonzeros(A) .*= 2f0
            oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
            @test P.owner !== original
            @test original.closed
            @test Array(y) ≈ 2f0.*(host*Array(x))
            replacement=P.owner
            oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
            @test P.owner === replacement
        finally
            close(P)
        end
    end

    # Compare representations, not floating equality: NaN payloads and signed zeros
    # may affect vendor arithmetic even though isequal/equality can hide differences.
    nan1=reinterpret(Float32,0x7fc00001)
    nan2=reinterpret(Float32,0x7fc00002)
    A=oneMKL.oneSparseMatrixCSR(oneArray(Int32[1,2]),oneArray(Int32[1]),
                               oneArray(Float32[nan1]),(1,1))
    P=oneMKL.sparse_optimize_gemv!('N',A)
    sx=oneArray(Float32[1]); sy=similar(sx)
    try
        old=P.owner
        oneMKL.sparse_gemv!('N',1f0,P,sx,0f0,sy)
        @test P.owner === old
        copyto!(nonzeros(A),Float32[nan2])
        oneMKL.sparse_gemv!('N',1f0,P,sx,0f0,sy)
        @test P.owner !== old
        @test old.closed
        copyto!(nonzeros(A),Float32[0])
        oneMKL.sparse_gemv!('N',1f0,P,sx,0f0,sy)
        old=P.owner
        copyto!(nonzeros(A),Float32[-0.0])
        oneMKL.sparse_gemv!('N',1f0,P,sx,0f0,sy)
        @test P.owner !== old
        @test old.closed
    finally
        close(P)
    end

    A=oneMKL.oneSparseMatrixCSR(host)
    fail=Ref(false)
    optimizer=(h,q)->begin
        fail[] && error("injected optimization failure")
        oneMKL.check_sparse_status(oneMKL.onemklXsparse_optimize_gemv(q,'N',h),:optimize_gemv)
    end
    P=oneMKL.prepare_sparse(A,(;kind=:gemv,trans='N'),optimizer)
    old=P.owner
    try
        oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
        before=Array(y)
        nonzeros(A) .*= 3f0
        fail[]=true
        @test_throws ErrorException oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
        @test P.owner === old
        @test !old.closed && !P.invalid
        @test Array(y)==before
        fail[]=false
        oneMKL.sparse_gemv!('N',1f0,P,x,0f0,y)
        @test old.closed
        @test Array(y) ≈ 3f0.*(host*Array(x))
    finally
        close(P)
    end
end
