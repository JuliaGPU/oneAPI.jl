using oneAPI, Test

function vadd(a, b, c)
    i = get_global_id()
    @inbounds c[i] = a[i] + b[i]
    return
end

dims = (2,)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = oneArray(a)
d_b = oneArray(b)
d_c = oneArray(c)

len = prod(dims)
@oneapi items=len vadd(d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c
