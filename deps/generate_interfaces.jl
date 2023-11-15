using oneAPI_Support_Headers_jll

blas = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include", "oneapi", "mkl", "blas", "buffer_decls.hpp")
lapack = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include", "oneapi", "mkl", "lapack.hpp")
sparse = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include", "oneapi", "mkl", "spblas.hpp")

dict_version = Dict{Int, Char}(1 => 'S', 2 => 'D', 3 => 'C', 4 => 'Z')

version_types = Dict{Char, String}('S' => "float",
                                   'D' => "double",
                                   'C' => "std::complex<float>",
                                   'Z' => "std::complex<double>")

version_types_header = Dict{Char, String}('S' => "float",
                                          'D' => "double",
                                          'C' => "float _Complex",
                                          'Z' => "double _Complex")

function generate_headers(filename::String, output::String)
  routines = Dict{String,Int}()
  signatures = []
  signatures2 = []
  cpp_headers = read(lapack, String)
  headers = ""

  # Remove comments
  for header in split(cpp_headers, '\n')
    !startswith(header, "//") && (headers *= header)
  end

  # Analyse each header
  headers = split(headers, ';')
  for (i, header) in enumerate(headers)
    # We only generate C interfaces for exported symbols
    !occursin("DLL_EXPORT", header) && !occursin("_scratchpad_size", header) && continue

    # We don't want to interface routines with the following types or parameters
    occursin("class", header) && continue
    occursin("sycl::half", header) && continue
    occursin("bfloat16", header) && continue
    occursin("int8_t", header) && continue
    occursin("sycl::event", header) && continue  # USM API
    occursin("group_count", header) && occursin("group_sizes", header) && continue  # USM API

    # Check if the routine is a template
    template = occursin("template", header)
    if template
      header = replace(header, "template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>" => "")
      header = replace(header, "template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>" => "")
      header = replace(header, "template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>" => "")
      header = replace(header, "template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>" => "")
      header = replace(header, "template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>" => "")
      header = replace(header, "template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>" => "")
    end

    # Replace the types
    header = replace(header, "void onemkl" => "int onemkl")
    header = replace(header, "sycl::queue &queue" => "syclQueue_t device_queue")
    header = replace(header, "std::int64_t" => "int64_t")

    header = replace(header, "sycl::buffer<float> &" => "float *")
    header = replace(header, "sycl::buffer<float>  &" => "float *")
    header = replace(header, "sycl::buffer<double> &" => "double *")
    header = replace(header, "sycl::buffer<std::complex<float>> &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<float>>  &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<double>> &" => "double _Complex *")
    header = replace(header, "sycl::buffer<int64_t> &" => "int64_t *")
  
    header = replace(header, "sycl::buffer<float, 1> &" => "float *")
    header = replace(header, "sycl::buffer<double, 1> &" => "double *")
    header = replace(header, "sycl::buffer<std::complex<float>, 1> &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<double>, 1> &" => "double _Complex *")
    header = replace(header, "sycl::buffer<int64_t, 1> &" => "int64_t *")

    header = replace(header, "template <>\n" => "")
    header = replace(header, "<std::complex<float>>" => "")
    header = replace(header, "<std::complex<double>>" => "")
    header = replace(header, "<float>" => "")
    header = replace(header, "<double>" => "")

    header = replace(header, "oneapi::mkl::transpose" => "onemklTranspose")
    header = replace(header, "oneapi::mkl::uplo" => "onemklUplo")
    header = replace(header, "oneapi::mkl::diag" => "onemklDiag")
    header = replace(header, "oneapi::mkl::side" => "onemklSide")
    header = replace(header, "oneapi::mkl::offset" => "onemklOffset")
    header = replace(header, "oneapi::mkl::job" => "onemklJob")
    header = replace(header, "oneapi::mkl::generate" => "onemklGenerate")
    header = replace(header, "oneapi::mkl::compz" => "onemklCompz")
    header = replace(header, "oneapi::mkl::direct" => "onemklDirect")
    header = replace(header, "oneapi::mkl::storev" => "onemklStorev")
    header = replace(header, "oneapi::mkl::rangev" => "onemklRangev")
    header = replace(header, "oneapi::mkl::order" => "onemklOrder")
    header = replace(header, "oneapi::mkl::jobsvd" => "onemklJobsvd")
    header = replace(header, "oneapi::mkl::layout" => "onemklLayout")
    header = replace(header, "oneapi::mkl::index" => "onemklIndex")

    # Sanytize the header
    header = replace(header, "\n" => "")
    header = replace(header, "DLL_EXPORT " => "")
    for i = 1:20
      header = replace(header, "  " => " ")
    end
    header = replace(header, "( " => "(")
    header = replace(header, ", )" => ")")

    ind1 = findfirst(' ', header)
    ind2 = findfirst('(', header)
    name_routine = header[ind1+1:ind2-1]
    !haskey(routines, name_routine) && (routines[name_routine] = 0)
    routines[name_routine] += 1

    version = occursin("double", header) ? 'D' : 'X'
    version = occursin("float", header) ? 'S' : version
    version = occursin("float _Complex", header) ? 'C' : version
    version = occursin("double _Complex", header) ? 'Z' : version
    version = occursin("_scratchpad_size", header) ? 'W' : version
    if version == 'X'
      if name_routine == "scal"
        (routines[name_routine] ≤ 4) && (version = dict_version[routines[name_routine]])
        (routines[name_routine] == 5) && (version = "SC")
        (routines[name_routine] == 6) && (version = "DZ")
      elseif routines[name_routine] ≤ 4
        version = dict_version[routines[name_routine]]
      else
        @warn("The routine $(name_routine) has more than 4 methods.")
      end
    end

    if version == 'W'
      # The version 'W' is used for routines with suffix "_scratchpad_size"
      versions = mapreduce(x -> startswith(name_routine, x), |, ["un", "he"]) ? ('C', 'Z') : ('S', 'D', 'C', 'Z')
      for blas_version in versions
        copy_header = header
        copy_header = replace(copy_header, "typename fp_type::value_type" => version_types_header[blas_version])
        copy_header = replace(copy_header, name_routine => "onemkl$(blas_version)$(name_routine)")
        copy_header = replace(copy_header, "void onemkl" => "int onemkl")
        push!(signatures, (copy_header, name_routine, blas_version, template))
      end
    else
      header = replace(header, name_routine => "onemkl$(version)$(name_routine)")
      header = replace(header, "void onemkl" => "int onemkl")
      push!(signatures, (header, name_routine, version, template))
    end
  end

  path_oneapi_headers = joinpath(@__DIR__, output)
  oneapi_headers = open(path_oneapi_headers, "w")
  # write(oneapi_headers, header)
  for (header, name_routine, version, template) in signatures
    # Don't wrap just a "_scratchpad_size"
    name_routine2 = replace(name_routine, "_scratchpad_size" => "")
    !haskey(routines, name_routine2) && continue
    push!(signatures2, (header, name_routine, version, template))

    pos = findfirst('(', header)
    fun = split(header, " ")
    len = 0
    for (i, part) in enumerate(fun)
      len += length(part)
      if len ≤ 90
        (i ≠ 1) && write(oneapi_headers, " ")
        write(oneapi_headers, part)
      else
        write(oneapi_headers, "\n")
        for i = 1:pos
          write(oneapi_headers, " ")
        end
        write(oneapi_headers, part)
        len = pos + length(part)
      end
    end
    write(oneapi_headers, ";\n\n")
  end
  close(oneapi_headers)
  return signatures2
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
    pararameters = replace(pararameters, "float _Complex " => "static_cast<std::complex<float> >")
    pararameters = replace(pararameters, "double _Complex " => "static_cast<std::complex<double> >")
    pararameters = replace(pararameters, ", float *" => ", ")
    pararameters = replace(pararameters, ", double *" => ", ")
    pararameters = replace(pararameters, ", float " => ", ")
    pararameters = replace(pararameters, ", double " => ", ")
    pararameters = replace(pararameters, ", *" => ", ")

    # TO DO: use a regex
    pararameters = replace(pararameters, "onemklTranspose transa" => "convert(transa)")
    pararameters = replace(pararameters, "onemklTranspose transb" => "convert(transb)")
    pararameters = replace(pararameters, "onemklSide left_right" => "convert(left_right)")
    pararameters = replace(pararameters, "onemklUplo upper_lower" => "convert(upper_lower)")
    pararameters = replace(pararameters, "onemklDiag unit_diag" => "convert(unit_diag)")
    pararameters = replace(pararameters, "onemklTranspose trans" => "convert(trans)")
    pararameters = replace(pararameters, "onemklUplo uplo" => "convert(uplo)")
    pararameters = replace(pararameters, "onemklDiag diag" => "convert(diag)")
    pararameters = replace(pararameters, "onemklSide side" => "convert(side)")
    pararameters = replace(pararameters, "onemklGenerate vect" => "convert(vect)")
    pararameters = replace(pararameters, "onemklGenerate vec" => "convert(vec)")
    pararameters = replace(pararameters, "onemklJob jobz" => "convert(jobz)")
    pararameters = replace(pararameters, "onemklJobsvd jobu" => "convert(jobu)")
    pararameters = replace(pararameters, "onemklJobsvd jobvt" => "convert(jobvt)")

    # TO DO: use a regex
    pararameters = replace(pararameters, ">alpha," => ">(alpha),")
    pararameters = replace(pararameters, ">beta," => ">(beta),")
    pararameters = replace(pararameters, ">ab," => ">(ab),")
    pararameters = replace(pararameters, ">a," => ">(a),")
    pararameters = replace(pararameters, ">b," => ">(b),")
    pararameters = replace(pararameters, ">c," => ">(c),")
    pararameters = replace(pararameters, ">x," => ">(x),")
    pararameters = replace(pararameters, ">y," => ">(y),")
    pararameters = replace(pararameters, ">u," => ">(u),")
    pararameters = replace(pararameters, ">vt," => ">(vt),")
    pararameters = replace(pararameters, ">taup," => ">(taup),")
    pararameters = replace(pararameters, ">tauq," => ">(tauq),")
    pararameters = replace(pararameters, ">tau," => ">(tau),")
    pararameters = replace(pararameters, ">scratchpad," => ">(scratchpad),")

    pararameters = replace(pararameters, ">a)" => ">(a))")
    pararameters = replace(pararameters, ">s)" => ">(s))")
    pararameters = replace(pararameters, ">result)" => ">(result))")

    variant = ""
    if library == "blas"
      variant = "column_major::"
    end

    write(oneapi_cpp, "extern \"C\" $header {\n")
    if template
      type = version_types[version]
      !occursin("scratchpad_size", name) && write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name<$type>($pararameters);\n")
      occursin("scratchpad_size", name)  && write(oneapi_cpp, "   int64_t scratchpad_size = oneapi::mkl::$library::$variant$name<$type>($pararameters);\n")
    else
      write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name($pararameters);\n")
    end
    if occursin("scratchpad_size", name)
      write(oneapi_cpp, "   return scratchpad_size;\n")
    else
      write(oneapi_cpp, "   __FORCE_MKL_FLUSH__(status);\n")
      write(oneapi_cpp, "   return 0;\n")
    end
    write(oneapi_cpp, "}")
    write(oneapi_cpp, "\n\n")
  end
  close(oneapi_cpp)
end

generate_headers(lapack, "onemkl_lapack.h")
generate_cpp("lapack", lapack, "onemkl_lapack.cpp")

# generate_headers(blas, "onemkl_blas.h")
# generate_cpp("blas", blas, "onemkl_blas.cpp")

# generate_headers(sparse, "onemkl_sparse.h")
# generate_cpp("sparse", sparse, "onemkl_sparse.cpp")
