# XXX: we currently need to call this function from a SYCL destructor
function onemklDestroy()
    @ccall liboneapi_support.onemklDestroy()::Cvoid
end
