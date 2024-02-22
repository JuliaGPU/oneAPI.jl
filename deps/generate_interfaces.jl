using oneAPI_Support_Headers_jll

blas = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include", "oneapi", "mkl", "blas", "buffer_decls.hpp")
lapack = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include", "oneapi", "mkl", "lapack.hpp")
sparse = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include", "oneapi", "mkl", "spblas.hpp")

dict_version = Dict{Int, Char}(1 => 'S', 2 => 'D', 3 => 'C', 4 => 'Z')

version_types = Dict{Char, String}('S' => "float",
                                   'D' => "double",
                                   'C' => "std::complex<float>",
                                   'Z' => "std::complex<double>",
                                   'I' => "int32_t",
                                   'L' => "int64_t")

version_types_header = Dict{Char, String}('S' => "float",
                                          'D' => "double",
                                          'C' => "float _Complex",
                                          'Z' => "double _Complex",
                                          'I' => "int32_t",
                                          'L' => "int64_t")

function generate_headers(library::String, filename::String, output::String)
  routines = Dict{String,Int}()
  signatures = []
  signatures2 = []
  cpp_headers = read(filename, String)
  headers = ""

  # Remove comments
  for header in split(cpp_headers, '\n')
    mapreduce(x -> !startswith(header, x) && !occursin("\"", header), &, ["/*", "*", "//", "[[deprecated", "#undef", "#define", "ONEMKL_DECLARE_BUF_"]) && (headers *= header)
  end

  # Analyse each header
  headers = split(headers, ';')
  for (i, header) in enumerate(headers)
    # We only generate C interfaces for exported symbols
    !occursin("DLL_EXPORT", header) && !occursin("_scratchpad_size", header) && continue

    # We don't want to interface routines with the following types, parameters or names
    occursin("class", header) && continue
    occursin("sycl::half", header) && continue
    occursin("bfloat16", header) && continue
    occursin("::int8_t", header) && continue
    occursin("sycl::event", header) && continue  # USM API
    occursin("group_count", header) && occursin("group_sizes", header) && continue  # USM API
    occursin("gemm_bias", header) && continue  # BLAS routine
    occursin("heevx", header) && continue  # LAPACK routine
    occursin("hegvx", header) && continue  # LAPACK routine
    occursin("(matrix_handle_t handle", header) && continue  # SPARSE routine
    occursin("gemvdot", header) && continue  # SPARSE routine
    occursin("matmat", header) && continue  # SPARSE routine

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
    header = replace(header, "std::int32_t" => "int32_t")
    header = replace(header, "std::int64_t" => "int64_t")

    if library == "blas"
      header = replace(header, "compute_mode mode = MKL_BLAS_COMPUTE_MODE" => "")
      header = replace(header, "index_base base=index_base::zero" => "onemklIndex base")

      header = replace(header, "sycl::buffer<Ta> &" => "Ta *")
      header = replace(header, "sycl::buffer<Tb> &" => "Tb *")
      header = replace(header, "sycl::buffer<Tc> &" => "Tc *")
      header = replace(header, "sycl::buffer<Td> &" => "Td *")
      header = replace(header, "sycl::buffer<Treal> &" => "Treal *")
      header = replace(header, "sycl::buffer<Tres> &" => "Tres *")
      header = replace(header, "sycl::buffer<T> &" => "T *")

      header = replace(header, "sycl::buffer<Ta, 1> &" => "Ta *")
      header = replace(header, "sycl::buffer<Tb, 1> &" => "Tb *")
      header = replace(header, "sycl::buffer<Tc, 1> &" => "Tc *")
      header = replace(header, "sycl::buffer<Td, 1> &" => "Td *")
      header = replace(header, "sycl::buffer<Treal, 1> &" => "Treal *")
      header = replace(header, "sycl::buffer<Tres, 1> &" => "Tres *")
      header = replace(header, "sycl::buffer<T,1> &" => "T *")
      header = replace(header, "sycl::buffer<T, 1> &" => "T *")
    end

    header = replace(header, "sycl::buffer<float> &" => "float *")
    header = replace(header, "sycl::buffer<float>  &" => "float *")
    header = replace(header, "sycl::buffer<double> &" => "double *")
    header = replace(header, "sycl::buffer<std::complex<float>> &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<float>>  &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<double>> &" => "double _Complex *")
    header = replace(header, "sycl::buffer<int32_t> &" => "int32_t *")
    header = replace(header, "sycl::buffer<int64_t> &" => "int64_t *")

    header = replace(header, "sycl::buffer<float, 1> &" => "float *")
    header = replace(header, "sycl::buffer<double, 1> &" => "double *")
    header = replace(header, "sycl::buffer<std::complex<float>, 1> &" => "float _Complex *")
    header = replace(header, "sycl::buffer<std::complex<double>, 1> &" => "double _Complex *")
    header = replace(header, "sycl::buffer<int32_t, 1> &" => "int32_t *")
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
    header = replace(header, "oneapi::mkl::property" => "onemklProperty")

    # Sanitize the header
    header = replace(header, " \\" => "")
    header = replace(header, "\n" => "")
    header = replace(header, "DLL_EXPORT " => "")
    header = replace(header, "const " => "")
    for i = 1:20
      header = replace(header, "  " => " ")
    end
    header = replace(header, "( " => "(")
    header = replace(header, ", )" => ")")
    occursin("voidgemm", header) && continue  # Bug with SPARSE routine

    ind1 = findfirst(' ', header)
    ind2 = findfirst('(', header)
    name_routine = header[ind1+1:ind2-1]
    !haskey(routines, name_routine) && (routines[name_routine] = 0)
    routines[name_routine] += 1

    # They use template for BLAS routines
    list_parameters = ()
    list_types = []
    list_versions = String[]
    if library == "blas"
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(T)", cpp_headers) && (list_parameters = ("T"))
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(T, Ts)", cpp_headers) && (list_parameters = ("T", "Ts"))
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(Ta, Tb, Tc, Ts)", cpp_headers) && (list_parameters = ("Ta", "Tb", "Tc", "Ts"))
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(T, Tres)", cpp_headers) && (list_parameters = ("T", "Tres"))
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(T, Treal)", cpp_headers) && (list_parameters = ("T", "Treal"))
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(T, Tc, Ts)", cpp_headers) && (list_parameters = ("T", "Tc", "Ts"))
      occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))(T, Tc)", cpp_headers) && (list_parameters = ("T", "Tc"))
      (list_parameters == ()) && @warn("Unable to determine the parametric parameters of $(name_routine).")
      for (type, version) in [(("float",), "S"),
                              (("double",), "D"),
                              (("std::complex<float>",), "C"),
                              (("std::complex<double>",), "Z")]
        if occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))($(type[1]))", cpp_headers)
          push!(list_types, type)
          push!(list_versions, version)
        end
      end
      for (type, version) in [(("float","float"), "S"),
                              (("double","double"), "D"),
                              (("std::complex<float>","float"), "CS"),
                              (("std::complex<double>","double"), "ZD"),
                              (("std::complex<float>","std::complex<float>"), "C"),
                              (("std::complex<double>","std::complex<double>"), "Z")]
        if occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))($(type[1]), $(type[2]))", cpp_headers)
          push!(list_types, type)
          push!(list_versions, version)
        end
      end
      for (type, version) in [(("float","float","float"), "S"),
                              (("double","double","double"), "D"),
                              (("std::complex<float>","float","float"), "CS"),
                              (("std::complex<float>","float", "std::complex<float>"), "C"),
                              (("std::complex<double>","double","double"), "ZD"),
                              (("std::complex<double>","double","std::complex<double>"), "Z")]
        if occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))($(type[1]), $(type[2]), $(type[3]))", cpp_headers)
          push!(list_types, type)
          push!(list_versions, version)
        end
      end
      for (type, version) in [(("float","float","float","float"), "S"),
                              (("double","double","double","double"), "D"),
                              (("std::complex<float>","std::complex<float>","std::complex<float>","std::complex<float>"), "C"),
                              (("std::complex<double>","std::complex<double>","std::complex<double>","std::complex<double>"), "Z")]
        if occursin("ONEMKL_DECLARE_BUF_$(uppercase(name_routine))($(type[1]), $(type[2]), $(type[3]), $(type[4]))", cpp_headers)
          push!(list_types, type)
          push!(list_versions, version)
        end
      end
    end
    version = 'X'
    if library == "sparse"
      version = occursin("int32_t", header) ? 'I' : version
      version = occursin("int64_t", header) ? 'L' : version
    end
    version = occursin("double", header) ? 'D' : version
    version = occursin("float", header) ? 'S' : version
    version = occursin("float _Complex", header) ? 'C' : version
    version = occursin("double _Complex", header) ? 'Z' : version
    version = occursin("_scratchpad_size", header) ? 'W' : version

    if version == 'W'
      # The version 'W' is used for routines with suffix "_scratchpad_size"
      versions = ('S', 'D', 'C', 'Z')
      mapreduce(x -> startswith(name_routine, x), |, ["or", "sy"]) && !startswith(name_routine, "sytrf") && (versions = ('S', 'D'))
      mapreduce(x -> startswith(name_routine, x), |, ["un", "he"]) && (versions = ('C', 'Z'))
      routines[name_routine] = routines[name_routine] - 1 + length(versions)
      for blas_version in versions
        copy_header = header
        copy_header = replace(copy_header, "typename fp_type::value_type" => version_types_header[blas_version])
        copy_header = replace(copy_header, "fp_type" => version_types_header[blas_version])
        copy_header = replace(copy_header, name_routine => "onemkl$(blas_version)$(name_routine)")
        copy_header = replace(copy_header, "void onemkl" => "int onemkl")
        push!(signatures, (copy_header, name_routine, blas_version, template))
      end
    else
      if isempty(list_versions)
        if name_routine == "set_csr_data"
          occursin("int32_t", header) && (version = "I" * version)
          occursin("int64_t", header) && (version = "L" * version)
        end
        header = replace(header, name_routine => "onemkl$(version)$(name_routine)")
        header = replace(header, "void onemkl" => "int onemkl")
        if library == "sparse"
          if occursin("std::complex", header)
            (version == 'C') && (header = replace(header, "std::complex " => "float _Complex "))
            (version == 'Z') && (header = replace(header, "std::complex " => "double _Complex "))
          end
          header = replace(header, "transpose " => "onemklTranspose ")
          header = replace(header, "uplo " => "onemklUplo ")
          header = replace(header, "diag " => "onemklDiag ")
          header = replace(header, "side " => "onemklSide ")
          header = replace(header, "layout " => "onemklLayout ")
          header = replace(header, "index_base " => "onemklIndex ")
          header = replace(header, "property " => "onemklProperty ")
          header = replace(header, name_routine => "sparse_" * name_routine)
        end
        push!(signatures, (header, name_routine, version, template))
      else

        n = length(list_parameters)
        for (i, type) in enumerate(list_types)
          version = list_versions[i]
          version = (name_routine ∈ ("her", "herk", "her2k", "rotg", "nrm2", "asum", "hpr")) && (version == "CS") ? "C" : version
          version = (name_routine ∈ ("her", "herk", "her2k", "rotg", "nrm2", "asum", "hpr")) && (version == "ZD") ? "Z" : version

          copy_header = header
          for (j, parameter) in enumerate(reverse(list_parameters))
            k = n-j+1
            copy_header = replace(copy_header, parameter => type[k])
          end
          copy_header = replace(copy_header, "transpose " => "onemklTranspose ")
          copy_header = replace(copy_header, "uplo " => "onemklUplo ")
          copy_header = replace(copy_header, "diag " => "onemklDiag ")
          copy_header = replace(copy_header, "side " => "onemklSide ")
          copy_header = replace(copy_header, "std::complex<float>" => "float _Complex")
          copy_header = replace(copy_header, "std::complex<double>" => "double _Complex")
          copy_header = replace(copy_header, name_routine => "onemkl$(version)$(name_routine)")
          copy_header = replace(copy_header, "void onemkl" => "int onemkl")
          push!(signatures, (copy_header, name_routine, version, template))
        end
      end
    end
  end

  # Check the number of methods
  blacklist = String[]
  for name_routine in keys(routines)
    if (routines[name_routine] > 5) && (library != "sparse")
      @warn "The routine $(name_routine) has more than 4 methods and will not be interfaced."
      push!(blacklist, name_routine)
    end
  end

  path_oneapi_headers = joinpath(@__DIR__, output)
  oneapi_headers = open(path_oneapi_headers, "w")
  # write(oneapi_headers, header)
  for (header, name_routine, version, template) in signatures
    # Blacklist
    (name_routine in blacklist) && continue

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
  signatures = generate_headers(library, filename, output)
  path_oneapi_cpp = joinpath(@__DIR__, output)
  oneapi_cpp = open(path_oneapi_cpp, "w")
  for (header, name, version, template) in signatures
    parameters = split(header, "(")[2]
    parameters = split(parameters, ")")[1]
    parameters = replace(parameters, "syclQueue_t device_queue" => "device_queue->val")
    parameters = replace(parameters, "int32_t " => "")
    parameters = replace(parameters, "int64_t " => "")
    parameters = replace(parameters, "matrix_handle_t " => "")
    parameters = replace(parameters, "float _Complex *" => "reinterpret_cast<std::complex<float> *>")
    parameters = replace(parameters, "double _Complex *" => "reinterpret_cast<std::complex<double> *>")
    parameters = replace(parameters, "float _Complex " => "static_cast<std::complex<float> >")
    parameters = replace(parameters, "double _Complex " => "static_cast<std::complex<double> >")
    parameters = replace(parameters, ", float *" => ", ")
    parameters = replace(parameters, ", double *" => ", ")
    parameters = replace(parameters, ", float " => ", ")
    parameters = replace(parameters, ", double " => ", ")
    parameters = replace(parameters, ", *" => ", ")

    for type in ("onemklTranspose", "onemklSide", "onemklUplo", "onemklDiag", "onemklGenerate",
                 "onemklJob", "onemklJobsvd", "onemklCompz", "onemklRangev", "onemklIndex", "onemklProperty")
      parameters = replace(parameters, Regex("$type ([a-z_]+),") => SubstitutionString("convert(\\1),"))
      parameters = replace(parameters, Regex(", $type ([a-z_]+)") => SubstitutionString(", convert(\\1)"))
    end
    parameters = replace(parameters, r" >([a-z]+)" => s" >(\1)")
    parameters = replace(parameters, r" \*>([a-z]+)" => s"*>(\1)")

    variant = ""
    if library == "blas"
      variant = "column_major::"
    end

    write(oneapi_cpp, "extern \"C\" $header {\n")
    if template
      type = version_types[version]
      !occursin("scratchpad_size", name) && write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name<$type>($parameters);\n")
      occursin("scratchpad_size", name)  && write(oneapi_cpp, "   int64_t scratchpad_size = oneapi::mkl::$library::$variant$name<$type>($parameters);\n")
    else
      write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name($parameters);\n")
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

generate_headers("lapack", lapack, "onemkl_lapack.h")
generate_headers("blas", blas, "onemkl_blas.h")
# generate_headers("sparse", sparse, "onemkl_sparse.h")

io = open("src/onemkl.h", "w")
headers_prologue = read("onemkl_prologue.h", String)
write(io, headers_prologue)
headers_blas = read("onemkl_blas.h", String)
write(io, "// BLAS\n")
write(io, headers_blas)
headers_lapack = read("onemkl_lapack.h", String)
write(io, "// LAPACK\n")
write(io, headers_lapack)
# headers_sparse = read("onemkl_sparse.h", String)
# write(io, "// SPARSE\n")
# write(io, headers_sparse)
headers_epilogue = read("onemkl_epilogue.h", String)
write(io, headers_epilogue)
close(io)

generate_cpp("lapack", lapack, "onemkl_lapack.cpp")
generate_cpp("blas", blas, "onemkl_blas.cpp")
# generate_cpp("sparse", sparse, "onemkl_sparse.cpp")

io = open("src/onemkl.cpp", "w")
cpp_prologue = read("onemkl_prologue.cpp", String)
write(io, cpp_prologue)
cpp_blas = read("onemkl_blas.cpp", String)
write(io, "// BLAS\n")
write(io, cpp_blas)
cpp_lapack = read("onemkl_lapack.cpp", String)
write(io, "// LAPACK\n")
write(io, cpp_lapack)
# cpp_sparse = read("onemkl_sparse.cpp", String)
# write(io, "// SPARSE\n")
# write(io, cpp_sparse)
cpp_epilogue = read("onemkl_epilogue.cpp", String)
write(io, cpp_epilogue)
close(io)
