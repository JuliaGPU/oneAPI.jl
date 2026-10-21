using oneAPI, Test
const Support=oneAPI.Support

sparse_wrapper_version_one()=UInt32(1)
sparse_wrapper_version_two()=UInt32(2)
sparse_wrapper_version_three()=UInt32(3)

@testset "sparse support artifact contract" begin
    marker=:onemkl_sparse_wrapper_abi_version
    csc=:onemklSsparse_set_csc_data
    one=@cfunction(sparse_wrapper_version_one,UInt32,())
    two=@cfunction(sparse_wrapper_version_two,UInt32,())
    three=@cfunction(sparse_wrapper_version_three,UInt32,())
    probe(entries)=Support._sparse_wrapper_error(name->get(entries,name,nothing))
    @test occursin("missing",probe(Dict(csc=>two)))
    @test occursin("found 1",probe(Dict(marker=>one,csc=>two)))
    @test occursin("found 3",probe(Dict(marker=>three,csc=>two)))
    @test occursin("CSC",probe(Dict(marker=>two)))
    @test probe(Dict(marker=>two,csc=>two))===nothing

    # Audit every generated raw sparse wrapper, not just resource-owner entrypoints.
    wrappers=filter(name->occursin(r"^onemkl[A-Z]sparse_",String(name)),names(Support;all=true))
    @test !isempty(wrappers)
    for name in wrappers
        f=getfield(Support,name)
        method=only(methods(f))
        args=Tuple{ntuple(_->Nothing,method.nargs-1)...}
        code=only(code_lowered(f,args)).code
        first_statement=first(code)
        @test first_statement isa Expr && first_statement.head===:call &&
              first_statement.args[1] isa GlobalRef &&
              first_statement.args[1].mod===Support &&
              first_statement.args[1].name===:_check_sparse_abi
    end

    saved_ok=Support._sparse_abi_ok[]
    saved_reason=Support._sparse_abi_error[]
    handle=Ref{Support.matrix_handle_t}(C_NULL)
    try
        Support._sparse_abi_ok[]=false
        Support._sparse_abi_error[]="injected incompatible artifact"
        @test_throws ErrorException Support.onemklXsparse_init_matrix_handle(handle)
        @test handle[]==C_NULL
    finally
        Support._sparse_abi_ok[]=saved_ok
        Support._sparse_abi_error[]=saved_reason
    end
end
