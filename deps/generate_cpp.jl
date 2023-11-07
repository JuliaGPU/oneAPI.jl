# https://github.com/oneapi-src/oneMKL
# https://raw.githubusercontent.com/oneapi-src/oneMKL/develop/include/oneapi/mkl/blas/detail/onemkl_blas_backends.hxx
# https://raw.githubusercontent.com/oneapi-src/oneMKL/develop/include/oneapi/mkl/lapack/detail/mkl_common/onemkl_lapack_backends.hxx

# library = "blas"
# filename = "onemkl_blas_backends.hxx"
# output = "onemkl_blas.h"
# output2 = "onemkl_blas.cpp"

library = "lapack"
filename = "onemkl_lapack_backends.hxx"
output = "onemkl_lapack.h"
output2 = "onemkl_lapack.cpp"

dict_version = Dict{Int, Char}(1 => 'S', 2 => 'D', 3 => 'C', 4 => 'Z')

version_types = Dict{Char, String}('S' => "float",
                                   'D' => "double",
                                   'C' => "std::complex<float>",
                                   'Z' => "std::complex<double>")

function generate_headers(filename::String, output::String)
  routines = Dict{String,Int}()
  headers = read(filename, String)
  signatures = []
  headers = split(headers, "*/")[2]
  headers = split(headers, ";", keepempty=false)
  for (i, header) in enumerate(headers)
    template = occursin("template", header)
    !occursin("ONEMKL_EXPORT", header) && (header = "")
    occursin("sycl::event", header) && (header = "")
    header = replace(header, "ONEMKL_EXPORT " => "")

    header = replace(header, "sycl::queue &queue" => "syclQueue_t device_queue")
    header = replace(header, "std::int64_t" => "int64_t")

    header = replace(header, "sycl::buffer<float> &" => "float *")
    header = replace(header, "sycl::buffer<double> &" => "double *")
    header = replace(header, "sycl::buffer<std::complex<float>> &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<double>> &" => "double _Complex *")
    header = replace(header, "sycl::buffer<int64_t> &" => "int64_t *")
  
    header = replace(header, "template <>\n" => "")
    header = replace(header, "<std::complex<float>>" => "")
    header = replace(header, "<std::complex<double>>" => "")
    header = replace(header, "<float>" => "")
    header = replace(header, "<double>" => "")

    header = replace(header, "oneapi::mkl::uplo" => "onemklUplo")
    header = replace(header, "oneapi::mkl::side" => "onemklSide")
    header = replace(header, "oneapi::mkl::transpose" => "onemklTranspose")
    header = replace(header, "oneapi::mkl::diag" => "onemklDiag")
    header = replace(header, "oneapi::mkl::job" => "onemklJob")
    header = replace(header, "oneapi::mkl::jobsvd" => "onemklJobsvd")
    header = replace(header, "oneapi::mkl::generate" => "onemklGenerate")
    
    header = replace(header, "\n" => "")
    for i = 1:20
      header = replace(header, "  " => " ")
    end
    header = replace(header, "( " => "(")

    if header ≠ ""
      ind1 = findfirst(' ', header)
      ind2 = findfirst('(', header)
      name_routine = header[ind1+1:ind2-1]
      if occursin("group_", header)
        header = replace(header, name_routine => "group_$(name_routine)")
        name_routine = "group_$(name_routine)"
      end
      !haskey(routines, name_routine) && (routines[name_routine] = 0)
      routines[name_routine] += 1

      version = occursin("float", header) ? 'S' : 'X'
      version = occursin("double", header) ? 'D' : version
      version = occursin("float _Complex", header) ? 'C' : version
      version = occursin("double _Complex", header) ? 'Z' : version
      version = version == 'X' ? dict_version[routines[name_routine]] : version

      header = replace(header, name_routine => "onemkl$(version)$(name_routine)")
      push!(signatures, (header, name_routine, version, template))
    end
  end

  for routine in keys(routines)
    if routines[routine] ≥ 5
      @warn "$routine has more than 4 versions."
    end
  end
  path_oneapi_headers = joinpath(@__DIR__, output)
  oneapi_headers = open(path_oneapi_headers, "w")
  for (header, name_routine, version) in signatures
    write(oneapi_headers, header)
    write(oneapi_headers, ";\n")
  end
  close(oneapi_headers)
  return signatures
end

function generate_cpp(library::String, filename::String, output::String)
  signatures = generate_headers(filename, output)
  path_oneapi_cpp = joinpath(@__DIR__, output)
  oneapi_cpp = open(path_oneapi_cpp, "w")
  for (header, name, version, template) in signatures
    pararameters = split(header, "(")[2]
    pararameters = split(pararameters, ")")[1]
    pararameters = replace(pararameters, "syclQueue_t device_queue" => "device_queue->val")
    pararameters = replace(pararameters, "int64_t " => "")
    pararameters = replace(pararameters, "float _Complex *" => "reinterpret_cast<std::complex<float> *>")
    pararameters = replace(pararameters, "double _Complex *" => "reinterpret_cast<std::complex<double> *>")
    pararameters = replace(pararameters, ", float *" => ", ")
    pararameters = replace(pararameters, ", double *" => ", ")
    pararameters = replace(pararameters, ", float " => ", ")
    pararameters = replace(pararameters, ", double " => ", ")
    pararameters = replace(pararameters, ", *" => ", ")

    pararameters = replace(pararameters, "onemklTranspose trans" => "convert(trans)")
    pararameters = replace(pararameters, "onemklUplo uplo" => "convert(uplo)")
    pararameters = replace(pararameters, "onemklDiag diag" => "convert(diag)")

    # TO DO: use a regex
    pararameters = replace(pararameters, ">a," => ">(a),")
    pararameters = replace(pararameters, ">b," => ">(b),")
    pararameters = replace(pararameters, ">taup," => ">(taup),")
    pararameters = replace(pararameters, ">tauq," => ">(tauq),")
    pararameters = replace(pararameters, ">tau," => ">(tau),")
    pararameters = replace(pararameters, ">scratchpad," => ">(scratchpad),")

    write(oneapi_cpp, "extern \"C\" $header {\n")
    if template
      type = version_types[version]
      !occursin("scratchpad_size", name) && write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$name<$type>($pararameters);\n")
      occursin("scratchpad_size", name)  && write(oneapi_cpp, "   int64_t scratchpad_size = oneapi::mkl::$library::$name<$type>($pararameters);\n")
    else
      write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$name($pararameters);\n")
    end
    !occursin("scratchpad_size", name) && write(oneapi_cpp, "   __FORCE_MKL_FLUSH__(status);\n")
    occursin("scratchpad_size", name) && write(oneapi_cpp, "   __FORCE_MKL_FLUSH__(scratchpad_size);\n")
    write(oneapi_cpp, "}")
    write(oneapi_cpp, "\n\n")
  end
  close(oneapi_cpp)
end

generate_headers(filename, output)
generate_cpp(library, filename, output2)
