# Compiler and Reflection

This page documents the compiler interface and code reflection tools for oneAPI.jl.

## Code Reflection

oneAPI.jl provides macros for inspecting code generation at various stages:

- `@device_code_lowered` - Show lowered IR (desugared Julia code)
- `@device_code_typed` - Show type-inferred IR
- `@device_code_warntype` - Show type-inferred IR with type stability warnings
- `@device_code_llvm` - Show LLVM IR
- `@device_code_spirv` - Show SPIR-V assembly
- `@device_code` - Show all compilation stages interactively

These macros are re-exported from GPUCompiler.jl. See the [GPUCompiler documentation](https://github.com/JuliaGPU/GPUCompiler.jl) for detailed usage.

### `return_type(f, tt) -> Type`

Return the inferred return type of function `f` when called with argument types `tt` in a GPU kernel context.

**Arguments:**
- `f`: Function to analyze
- `tt`: Tuple type of arguments

**Returns:**
- Type that `f(args...)` would return where `args::tt`

**Example:**
```julia
function compute(x::Float32)
    return x * 2.0f0
end

rt = oneAPI.return_type(compute, Tuple{Float32})
@assert rt == Float32
```


## Inspecting Generated Code

Code reflection tools help you understand how your Julia code is compiled to GPU code:

### LLVM IR

View the LLVM intermediate representation:

```julia
using oneAPI

function kernel(a, b)
    i = get_global_id()
    @inbounds a[i] = b[i] + 1.0f0
    return
end

a = oneArray(zeros(Float32, 10))
b = oneArray(rand(Float32, 10))

@device_code_llvm @oneapi groups=1 items=10 kernel(a, b)
```

### SPIR-V Assembly

View the final SPIR-V assembly that runs on the GPU:

```julia
@device_code_spirv @oneapi groups=1 items=10 kernel(a, b)
```

### Type Inference

Check for type instabilities that hurt performance:

```julia
@device_code_warntype @oneapi groups=1 items=10 kernel(a, b)
```

### Type-Inferred IR

See the typed intermediate representation:

```julia
@device_code_typed @oneapi groups=1 items=10 kernel(a, b)
```

### Interactive Inspection

Use `@device_code` for an interactive menu:

```julia
@device_code @oneapi groups=1 items=10 kernel(a, b)
# Opens a menu to select which compilation stage to view
```

## Return Type Inference

Query the return type of a kernel:

```julia
function compute(x::Float32)
    return x * 2.0f0
end

# Infer return type
rt = oneAPI.return_type(compute, Tuple{Float32})
@assert rt == Float32
```

## Debugging Type Issues

### Common Type Instability Sources

```julia
# ❌ Type instability: Conditional returns different types
function bad_kernel(x, flag)
    if flag
        return x        # Float32
    else
        return 0        # Int
    end
end

# ✅ Type stable: Consistent return type
function good_kernel(x, flag)
    if flag
        return x        # Float32
    else
        return 0.0f0    # Float32
    end
end
```

### Using @device_code_warntype

```julia
function mystery_kernel!(output, input)
    i = get_global_id()
    @inbounds output[i] = some_complex_function(input[i])
    return
end

# Check for type issues
@device_code_warntype @oneapi groups=1 items=10 mystery_kernel!(a, b)

# Look for red warnings indicating type instability
```

## Compilation Options

### Kernel vs Device Function

```julia
# Compile as kernel (default for @oneapi)
@device_code_llvm @oneapi kernel=true kernel(a, b)

# Compile as device function (callable from other kernels)
@device_code_llvm @oneapi kernel=false helper_function(x)
```

### Always Inline

Force inlining of device functions:

```julia
@oneapi always_inline=true kernel(a, b)
```

### Custom Kernel Name

Specify a custom name for the kernel:

```julia
@oneapi name="my_custom_kernel" kernel(a, b)
```

## Example: Optimizing a Kernel

Here's a workflow for optimizing a kernel using reflection tools:

```julia
using oneAPI

# Initial version
function sum_kernel_v1!(result, data)
    i = get_global_id()
    if i == 1
        sum = 0
        for j in 1:length(data)
            sum += data[j]
        end
        result[1] = sum
    end
    return
end

data = oneArray(rand(Float32, 1000))
result = oneArray(zeros(Float32, 1))

# Check for type issues
@device_code_warntype @oneapi groups=1 items=1 sum_kernel_v1!(result, data)
# Notice: `sum` might be Int instead of Float32!

# Fixed version
function sum_kernel_v2!(result, data)
    i = get_global_id()
    if i == 1
        sum = 0.0f0  # Explicitly Float32
        for j in 1:length(data)
            sum += data[j]
        end
        result[1] = sum
    end
    return
end

# Verify the fix
@device_code_warntype @oneapi groups=1 items=1 sum_kernel_v2!(result, data)
# Should be type-stable now!

# Check the generated code
@device_code_llvm @oneapi groups=1 items=1 sum_kernel_v2!(result, data)
```

## Profiling

For performance profiling, see the [Performance Guide](@ref).

## Troubleshooting

### Compilation Errors

If you encounter compilation errors:

1. **Check type stability**: Use `@device_code_warntype`
2. **Inspect LLVM IR**: Use `@device_code_llvm` to see if the issue is in LLVM generation
3. **Simplify the kernel**: Comment out sections to isolate the problematic code
4. **Check argument types**: Ensure arguments are GPU-compatible (isbits types)

### SPIR-V Issues

If SPIR-V generation fails:

1. **Update dependencies**: Ensure SPIRV-LLVM-Translator is up to date
2. **Check device capabilities**: Some operations require specific hardware features
3. **Reduce complexity**: Very complex kernels might hit compiler limits

### Performance Issues

If your kernel is slow:

1. **Profile memory access patterns**: Coalesced access is crucial
2. **Check occupancy**: Are you launching enough work-items?
3. **Minimize barriers**: Synchronization has overhead
4. **Use local memory wisely**: It's faster than global memory but limited in size
