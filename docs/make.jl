using Pkg

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
# # when first running instantiate
Pkg.instantiate()
using Documenter
using Documenter.Remotes
using oneAPI

oneAPI.versioninfo()

makedocs(
    sitename = "oneAPI.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        canonical = "https://exanauts.github.io/ExaPF.jl/stable/",
        mathengine = Documenter.KaTeX(),
    ),
    modules = [oneAPI],
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Getting Started" => "getting_started.md",
        "Usage" => [
            "Array Programming" => "arrays.md",
            "Kernel Programming" => "kernels.md",
            "Memory Management" => "memory.md",
            "Device Intrinsics" => "device.md",
            "Performance Guide" => "usage/performance.md",
        ],
        "API Reference" => [
            "Overview" => "api.md",
            "Context & Device Management" => "api/context.md",
            "Array Operations" => "api/arrays.md",
            "Kernel Programming" => "api/kernels.md",
            "Memory Management" => "api/memory.md",
            "Compiler & Reflection" => "api/compiler.md",
            "Level Zero (oneL0)" => "level_zero.md",
            "oneMKL" => "onemkl.md",
        ],
        "Troubleshooting" => "troubleshooting.md",
    ],
    checkdocs = :none,  # Don't error on missing docstrings
    warnonly = [:cross_references, :missing_docs],  # Only warn, don't error
)

deploydocs(
    repo = "github.com/JuliaGPU/oneAPI.jl.git",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    push_preview = true,
)

