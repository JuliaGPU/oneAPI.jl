using oneAPI,GPUArrays,SparseArrays,LinearAlgebra,Test
using oneAPI.oneMKL: oneSparseMatrixCSR

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
