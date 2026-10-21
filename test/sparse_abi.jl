using oneAPI, Test
const Support=oneAPI.Support

sparse_wrapper_version_one()=UInt32(1)
sparse_wrapper_version_two()=UInt32(2)
sparse_wrapper_version_three()=UInt32(3)

struct ForbiddenSparseArgument end
const sparse_argument_conversions=Ref(0)
function Base.cconvert(::Type{T},::ForbiddenSparseArgument) where T
    sparse_argument_conversions[]+=1
    throw(ArgumentError("native sparse argument conversion was reached"))
end

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

    saved_ok=Support._sparse_abi_ok[]
    saved_reason=Support._sparse_abi_error[]
    handle=Ref{Support.matrix_handle_t}(C_NULL)
    try
        Support._sparse_abi_ok[]=false
        Support._sparse_abi_error[]="injected incompatible artifact"
        # Every raw wrapper must reject before converting or passing native arguments.
        wrappers=filter(name->occursin(r"^onemkl[A-Z]sparse_",String(name)),names(Support;all=true))
        @test !isempty(wrappers)
        for name in wrappers
            f=getfield(Support,name)
            method=only(methods(f))
            args=ntuple(_->ForbiddenSparseArgument(),method.nargs-1)
            result=try
                f(args...)
                nothing
            catch err
                err
            end
            @test result isa ErrorException &&
                  occursin("injected incompatible artifact",sprint(showerror,result))
        end
        @test sparse_argument_conversions[]==0
        @test_throws ErrorException Support.onemklXsparse_init_matrix_handle(handle)
        @test handle[]==C_NULL
    finally
        Support._sparse_abi_ok[]=saved_ok
        Support._sparse_abi_error[]=saved_reason
    end
end
