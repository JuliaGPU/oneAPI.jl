# API Reference

This page provides an overview of the oneAPI.jl API. For detailed documentation, see the specific API reference pages:

- [Context & Device Management](api/context.md) - Managing drivers, devices, and contexts
- [Array Operations](api/arrays.md) - Working with GPU arrays
- [Kernel Programming](api/kernels.md) - Writing and launching custom kernels
- [Memory Management](api/memory.md) - Memory allocation and transfer
- [Compiler & Reflection](api/compiler.md) - Code generation and introspection

## Core Functions

```@autodocs
Modules = [oneAPI]
Pages   = ["src/context.jl", "src/utils.jl"]
Filter = t -> t !== oneAPI.synchronize
```

## Compiler Functions

```@autodocs
Modules = [oneAPI]
Pages = ["src/compiler/execution.jl", "src/compiler/reflection.jl"]
```

## oneL0 (Level Zero)

Low-level bindings to the Level Zero API. See the [Level Zero page](level_zero.md) for details.

```@autodocs
Modules = [oneAPI.oneL0]
Filter = t -> t !== oneAPI.oneL0.synchronize
```

## oneMKL

Intel oneAPI Math Kernel Library bindings. See the [oneMKL page](onemkl.md) for details.

```@autodocs
Modules = [oneAPI.oneMKL]
```

