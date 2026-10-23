module PTI

using ..APIUtils

using CEnum: @cenum

using PTI_jll

# PTI_jll does not load its libraries: libpti_view starts a thread once used, and libpti
# hooks into the Level Zero loader. They are only loaded by the first call into them.
is_available() = PTI_jll.is_available()

include("libpti.jl")

export PTIError

struct PTIError <: Exception
    code::pti_result
end

Base.showerror(io::IO, err::PTIError) =
    print(io, "PTIError: ", unsafe_string(ptiResultTypeToString(err.code)),
          " (code $(Int(err.code)))")


## buffer management

# PTI requests and returns buffers from its own threads, including while the process exits,
# where running Julia code would hang. libpti_buffers implements the callbacks in C, and
# queues the completed buffers for `foreach_record` to process.
const callbacks_set = Ref(false)

function set_callbacks()
    callbacks_set[] && return
    ptiViewSetCallbacks(cglobal((:pti_buffers_request, libpti_buffers)),
                        cglobal((:pti_buffers_complete, libpti_buffers)))
    callbacks_set[] = true
    return
end

# Call `f` on a pointer to every record in the completed buffers, which are released
# afterwards. PTI does not document how long the strings that records point to stay valid,
# so `f` should copy anything it needs.
function foreach_record(f)
    buf = Ref{Ptr{UInt8}}()
    used = Ref{Csize_t}()
    record = Ref{Ptr{pti_view_record_base}}()
    while ccall((:pti_buffers_pop, libpti_buffers), Cint,
                (Ptr{Ptr{UInt8}}, Ptr{Csize_t}), buf, used) == 1
        try
            record[] = C_NULL
            while true
                res = unchecked_ptiViewGetNextRecord(buf[], used[], record)
                res == PTI_STATUS_END_OF_BUFFER && break
                res == PTI_SUCCESS || throw_api_error(res)
                f(record[])
            end
        finally
            Libc.free(buf[])
        end
    end
end

# Number of buffers whose records were lost because they could not be allocated.
lost_buffers() = ccall((:pti_buffers_lost, libpti_buffers), UInt64, ())

function api_name(group, id)
    name = Ref{Ptr{Cchar}}()
    res = unchecked_ptiViewGetApiIdName(group, id, name)
    res == PTI_SUCCESS ? unsafe_string(name[]) : "unknown API $id"
end

end
