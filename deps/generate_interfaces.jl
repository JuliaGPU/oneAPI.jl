using oneAPI_Support_Headers_jll

include("generate_helpers.jl")

include_dir = joinpath(oneAPI_Support_Headers_jll.artifact_dir, "include")
blas = [joinpath(include_dir, "oneapi", "mkl", "blas", "buffer_decls.hpp")]
lapack = [joinpath(include_dir, "oneapi", "mkl", "lapack", "lapack.hpp"),
          joinpath(include_dir, "oneapi", "mkl", "lapack", "scratchpad.hpp")]
sparse = [joinpath(include_dir, "oneapi", "mkl", "spblas", "sparse_structures.hpp"),
          joinpath(include_dir, "oneapi", "mkl", "spblas", "sparse_auxiliary.hpp"),
          joinpath(include_dir, "oneapi", "mkl", "spblas", "sparse_operations.hpp")]

dict_version = Dict{Int, Char}(1 => 'S', 2 => 'D', 3 => 'C', 4 => 'Z')

version_types = Dict{Char, String}('S' => "float",
                                   'D' => "double",
                                   'C' => "std::complex<float>",
                                   'Z' => "std::complex<double>")

version_types_header = Dict{Char, String}('S' => "float",
                                          'D' => "double",
                                          'C' => "float _Complex",
                                          'Z' => "double _Complex")

comments = ["namespace", "#", "}", "/*", "*", "//", "[[", "ONEMKL_DECLARE_", "ONEMKL_INLINE_DECLARE"]

void_output = ["init_matrix_handle", "init_matmat_descr", "release_matmat_descr", "set_matmat_data",
               "get_matmat_data", "init_omatadd_descr", "init_omatconvert_descr"]

function generate_headers(library::String, filename::Vector{String}, output::String; pattern::String="")
  routines = Dict{String,Int}()
  signatures = []
  signatures2 = []
  cpp_headers = ""
  for file in filename
    cpp_headers = cpp_headers * read(file, String)
  end
  cpp_headers = replace(cpp_headers, "std::int32_t" => "int32_t")
  cpp_headers = replace(cpp_headers, "std::int64_t" => "int64_t")
  cpp_headers = replace(cpp_headers, "; \\" => ";")
  cpp_headers = replace(cpp_headers, ")\n\n" => ");\n\n")
  cpp_headers = replace(cpp_headers, "\\\n" => "\n")
  cpp_headers = replace(cpp_headers, "sycl::event\n" => "sycl::event ")
  headers = ""

  # Remove comments
  for header in split(cpp_headers, '\n')
    mapreduce(x -> !startswith(strip(header), x) && !occursin("\"", header), &, comments) && (headers *= header)
  end

  # Analyse each header
  headers = split(headers, ';')
  for (i, header) in enumerate(headers)
    # We only generate C interfaces for exported symbols
    !occursin("DLL_EXPORT", header) && !occursin("_scratchpad_size", header) && continue

    # We don't want to interface routines with the following types, parameters or names
    occursin("class", header) && continue
    occursin("span", header) && continue
    occursin("bfloat16", header) && continue
    occursin("::int8_t", header) && continue
    (library == "lapack") && occursin("void", header) && continue # We only want USM routines
    (library == "sparse") && occursin("trsv", header) && !occursin("optimize_trsv", header) && !occursin("alpha", header) && continue  # SPARSE routine
    occursin("(matrix_handle_t SpMat", header) && continue  # SPARSE routine
    occursin("set_csr_data(matrix_handle_t", header) && continue  # SPARSE routine
    occursin("release_matrix_handle(matrix_handle_t", header) && continue  # SPARSE routine
    occursin("get_matmat_data", header) && continue  # SPARSE routine
    occursin("matmat(", header) && continue  # SPARSE routine
    bool = occursin("release", header) || occursin("init", header)
    (library == "sparse") && occursin("omatconvert", header) && !bool && continue  # SPARSE routine
    (library == "sparse") && occursin("omatadd", header) && !bool && continue  # SPARSE routine
    occursin("gemm_bias", header) && continue  # BLAS routine
    occursin("getri_batch", header) && occursin("ldainv", header) && continue  # LAPACK routine

    # Check if the routine is a template
    template = occursin("template", header)
    if template
      header = replace(header, "template <typename fp, oneapi::mkl::lapack::internal::is_floating_point<fp> = nullptr>         " => "")
      header = replace(header, "template <typename fp, oneapi::mkl::lapack::internal::is_real_floating_point<fp> = nullptr>    " => "")
      header = replace(header, "template <typename fp, oneapi::mkl::lapack::internal::is_complex_floating_point<fp> = nullptr> " => "")

      header = replace(header, "template <typename data_t, oneapi::mkl::lapack::internal::is_floating_point<data_t> = nullptr>" => "")
      header = replace(header, "template <typename data_t, oneapi::mkl::lapack::internal::is_real_floating_point<data_t> = nullptr>" => "")
      header = replace(header, "template <typename data_t, oneapi::mkl::lapack::internal::is_complex_floating_point<data_t> = nullptr>" => "")
      header = replace(header, "template <typename fp_type, internal::is_floating_point<fp_type> = nullptr>" => "")
      header = replace(header, "template <typename fp_type, internal::is_real_floating_point<fp_type> = nullptr>" => "")
      header = replace(header, "template <typename fp_type, internal::is_complex_floating_point<fp_type> = nullptr>" => "")
    end

    type_routine = ""
    if occursin("_scratchpad_size", header)
      type_routine = "scratchpad_size"
    elseif occursin("sycl::event", header)
      header = replace(header, "const std::vector<sycl::event> &events = {}" => "")
      header = replace(header, "const std::vector<sycl::event> &events = {}" => "")
      header = replace(header, "const std::vector<sycl::event> &event_list = {}" => "")
      header = replace(header, "std::vector<sycl::event> &dependencies = {}" => "")
      header = replace(header, "std::vector<sycl::event> &dependencies" => "")  # typo in "onemkl_sparse.cpp"
      type_routine = "usm"
    else
      type_routine = "buffer"
    end

    # Add a space for the returned argument
    header = replace(header, "sycl::event" => "sycl::event ")
    header = replace(header, "void" => "void ")

    # Replace the types
    header = replace(header, "sycl::queue &queue" => "syclQueue_t device_queue")
    header = replace(header, "sycl::queue& queue" => "syclQueue_t device_queue")

    if library ∈ ("blas", "sparse")
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
      header = replace(header, "sycl::buffer<Ti, 1> &" => "Ti *")
      header = replace(header, "sycl::buffer<Tf, 1> &" => "Tf *")
      header = replace(header, "sycl::buffer<Treal, 1> &" => "Treal *")
      header = replace(header, "sycl::buffer<Tres, 1> &" => "Tres *")
      header = replace(header, "sycl::buffer<T,1> &" => "T *")
      header = replace(header, "sycl::buffer<T, 1> &" => "T *")
      header = replace(header, "sycl::buffer<FpType, 1> &" => "FpType *")
      header = replace(header, "sycl::buffer<IntType, 1> &" => "IntType *")
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
    header = replace(header, "sycl::buffer<std::uint8_t, 1> *" => "uint8_t *")
    header = replace(header, "sycl::buffer<int32_t, 1> &" => "int32_t *")
    header = replace(header, "sycl::buffer<int64_t, 1> &" => "int64_t *")
    header = replace(header, "sycl::buffer<int64_t, 1> *" => "int64_t *")

    header = replace(header, "std::complex<float>  *" => "float _Complex *")
    header = replace(header, "std::complex<float> *" => "float _Complex *")
    header = replace(header, "std::complex<double> *" => "double _Complex *")

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
    header = replace(header, "sparse::matmat_descr_t" => "matmat_descr_t")

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
    header = replace(header, ",)" => ")")
    header = replace(header, " void" => "void")
    header = replace(header, " sycl::event" => "sycl::event")
    header = replace(header, "* const* " => "**")
    header = replace(header, "int64_t**" => "int64_t **")

    ind1 = findfirst(' ', header)
    ind2 = findfirst('(', header)
    name_routine = header[ind1+1:ind2-1]
    !haskey(routines, name_routine * type_routine) && (routines[name_routine * type_routine] = 0)
    (name_routine == "gesvd_scratchpad_size") && (routines[name_routine * type_routine] > 1) && continue
    routines[name_routine * type_routine] += 1

    # They use template for BLAS and SPARSE routines
    list_parameters, list_types, list_versions, list_suffix = analyzer_template(library, cpp_headers, name_routine)
    !isempty(list_parameters) && (type_routine == "buffer") && (library == "sparse") && continue  # Only wrap the USM version of sparse routines

    version = 'X'
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
      routines[name_routine * type_routine] = routines[name_routine * type_routine] - 1 + length(versions)
      for blas_version in versions
        copy_header = header
        copy_header = replace(copy_header, "typename fp_type::value_type" => version_types_header[blas_version])
        copy_header = replace(copy_header, "fp_type" => version_types_header[blas_version])
        copy_header = replace(copy_header, "fp" => version_types_header[blas_version])
        copy_header = replace(copy_header, name_routine => "onemkl$(blas_version)$(name_routine)")
        if name_routine ∈ ("heevx_scratchpad_size", "hegvx_scratchpad_size")
          copy_header = replace(copy_header, "typename float _Complex::value_type" => "float")
          copy_header = replace(copy_header, "typename double _Complex::value_type" => "double")
        end
        if occursin("batch", name_routine) && !occursin("*", header)
          copy_header = replace(copy_header, "_batch" => "_batch_strided")
        end
        push!(signatures, (copy_header, name_routine, blas_version, type_routine, template))
      end
    else
      if isempty(list_versions)
        # The routine "optimize_trsm" has two versions.
        suffix = ""
        (name_routine == "optimize_trsm") && occursin("columns", header) && (suffix = "_advanced")
        (name_routine == "optimize_gemm") && occursin("columns", header) && (suffix = "_advanced")
        name_routine ∈ ("set_csr_data", "set_coo_data") && occursin("int64_t", header) && (suffix = "_64")
        occursin("batch", name_routine) && !occursin("**", header) && (suffix = "_strided")

        header = replace(header, "$(name_routine)(" => "onemkl$(version)$(name_routine)$(suffix)(")
        header = replace(header, "void onemkl" => "int onemkl")
        header = replace(header, "sycl::event onemkl" => "int onemkl")
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
          header = replace(header, "sparse::matrix_view_descr " => "onemklMatrixView ")
          header = replace(header, "matrix_view_descr " => "onemklMatrixView ")
          header = replace(header, "sparse::matmat_request " => "onemklMatmatRequest ")
          header = replace(header, "omatconvert_alg " => "onemklOmatconvertAlg ")
          header = replace(header, "omatadd_alg " => "onemklOmataddAlg ")
          header = replace(header, name_routine => "sparse_" * name_routine)
        end
        push!(signatures, (header, name_routine, version, type_routine, template))
      else
        n = length(list_parameters)
        for (i, type) in enumerate(list_types)
          version = list_versions[i]
          suffix = list_suffix[i]
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
          copy_header = replace(copy_header, "layout " => "onemklLayout ")
          copy_header = replace(copy_header, "index_base " => "onemklIndex ")
          copy_header = replace(copy_header, "std::complex<float>" => "float _Complex")
          copy_header = replace(copy_header, "std::complex<double>" => "double _Complex")
          copy_header = replace(copy_header, "sycl::half" => "short")
          copy_header = replace(copy_header, name_routine => "onemkl$(version)$(name_routine)$(suffix)")
          copy_header = replace(copy_header, "sycl::event onemkl" => "int onemkl")
          copy_header = replace(copy_header, "void onemkl" => "int onemkl")
          if library == "sparse"
            copy_header = replace(copy_header, name_routine => "sparse_" * name_routine)
          end
          if occursin("batch", name_routine) && !occursin("**", header)
            copy_header = replace(copy_header, "_batch" => "_batch_strided")
          end
          if library == "blas"
            # Out-of-place variants of trsm and trmm
            if occursin("trsm", header) && occursin("ldc", header)
              copy_header = replace(copy_header, "trsm" => "trsm_variant")
            end
            if occursin("trmm", header) && occursin("ldc", header)
              copy_header = replace(copy_header, "trmm" => "trmm_variant")
            end
            copy_header = replace(copy_header, "compute_mode mode," => "")
            copy_header = replace(copy_header, ", compute_mode mode)" => ")")
            copy_header = replace(copy_header, "value_or_pointer<float _Complex>" => "float _Complex")
            copy_header = replace(copy_header, "value_or_pointer<double _Complex>" => "double _Complex")
            copy_header = replace(copy_header, "value_or_pointer<short>" => "short")
            copy_header = replace(copy_header, "value_or_pointer<float>" => "float")
            copy_header = replace(copy_header, "value_or_pointer<double>" => "double")
          end
          push!(signatures, (copy_header, name_routine, version, type_routine, template))
        end
      end
    end
  end

  # Check the number of methods
  blacklist = String[]
  for name_routine in keys(routines)
    if (routines[name_routine] > 4)
      if occursin("set_csr_data", name_routine) || occursin("set_coo_data", name_routine) || occursin("_batch", name_routine)
        if (routines[name_routine] > 8)
          @warn "The routine $(name_routine) has $(routines[name_routine]) and will not be interfaced."
          push!(blacklist, name_routine)
        end
      else
        @warn "The routine $(name_routine) has $(routines[name_routine]) and will not be interfaced."
        push!(blacklist, name_routine)
      end
    end
  end

  path_oneapi_headers = joinpath(@__DIR__, output)
  oneapi_headers = open(path_oneapi_headers, "w")

  for (header, name_routine, version, type_routine, template) in signatures
    # Blacklist
    (name_routine in blacklist) && continue

    # Pass scalars (e.g. alpha/beta inputs) as references instead of values
    for type in ("short", "float", "double", "float _Complex", "double _Complex")
      header = replace(header, Regex("$type ([A-Za-z0-9]+(?![^,]*[_*]))[^,]*,") => SubstitutionString("$type $pattern\\1,"))
      header = replace(header, Regex(", $type ([A-Za-z0-9)]+(?![^,]*[_*]))[^,]*") => SubstitutionString(", $type $pattern\\1"))
    end

    push!(signatures2, (header, name_routine, version, type_routine, template))

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

function generate_cpp(library::String, filename::Vector{String}, output::String; pattern::String="")
  signatures = generate_headers(library, filename, output; pattern)
  path_oneapi_cpp = joinpath(@__DIR__, output)
  oneapi_cpp = open(path_oneapi_cpp, "w")
  for (header, name, version, type_routine, template) in signatures
    parameters = split(header, "(")[2]
    parameters = split(parameters, ")")[1]
    parameters = replace(parameters, "syclQueue_t device_queue" => "device_queue->val")
    parameters = replace(parameters, "int32_t* " => "")
    parameters = replace(parameters, "int32_t " => "")
    parameters = replace(parameters, "int64_t* " => "")
    parameters = replace(parameters, "int64_t " => "")
    parameters = replace(parameters, "matrix_handle_t *" => "(oneapi::mkl::sparse::matrix_handle_t*) ")
    parameters = replace(parameters, "matrix_handle_t " => "(oneapi::mkl::sparse::matrix_handle_t) ")
    parameters = replace(parameters, "matmat_descr_t *" => "(oneapi::mkl::sparse::matmat_descr_t*) ")
    parameters = replace(parameters, "matmat_descr_t " => "(oneapi::mkl::sparse::matmat_descr_t) ")
    parameters = replace(parameters, "omatadd_descr_t *" => "(oneapi::mkl::sparse::omatadd_descr_t*) ")
    parameters = replace(parameters, "omatadd_descr_t " => "(oneapi::mkl::sparse::omatadd_descr_t) ")
    parameters = replace(parameters, "omatconvert_descr_t *" => "(oneapi::mkl::sparse::omatconvert_descr_t*) ")
    parameters = replace(parameters, "omatconvert_descr_t " => "(oneapi::mkl::sparse::omatconvert_descr_t) ")
    parameters = replace(parameters, "short **" => "reinterpret_cast<sycl::half **>")
    parameters = replace(parameters, "float _Complex **" => "reinterpret_cast<std::complex<float> **>")
    parameters = replace(parameters, "double _Complex **" => "reinterpret_cast<std::complex<double> **>")
    parameters = replace(parameters, "short *" => "reinterpret_cast<sycl::half *>")
    parameters = replace(parameters, "float _Complex *" => "reinterpret_cast<std::complex<float> *>")
    parameters = replace(parameters, "double _Complex *" => "reinterpret_cast<std::complex<double> *>")
    parameters = replace(parameters, "short " => "sycl::bit_cast<sycl::half>")
    parameters = replace(parameters, "float _Complex " => "static_cast<std::complex<float> >")
    parameters = replace(parameters, "double _Complex " => "static_cast<std::complex<double> >")
    parameters = replace(parameters, ", float *" => ", ")
    parameters = replace(parameters, ", double *" => ", ")
    parameters = replace(parameters, ", float " => ", ")
    parameters = replace(parameters, ", double " => ", ")
    parameters = replace(parameters, ", **" => ", ")
    parameters = replace(parameters, ", *" => ", ")
    parameters = replace(parameters, "onemklTranspose *trans," => "convert(trans, group_count),")
    parameters = replace(parameters, "onemklTranspose* trans," => "convert(trans, group_count),")
    parameters = replace(parameters, "onemklUplo *uplo," => "convert(uplo, group_count),")
    parameters = replace(parameters, "onemklUplo* uplo," => "convert(uplo, group_count),")
    parameters = replace(parameters, "onemklDiag *diag," => "convert(diag, group_count),")
    parameters = replace(parameters, "onemklDiag* diag," => "convert(diag, group_count),")
    parameters = replace(parameters, "onemklSide *side," => "convert(side, group_count),")
    parameters = replace(parameters, "onemklSide* side," => "convert(side, group_count),")

    for type in ("onemklTranspose", "onemklSide", "onemklUplo", "onemklDiag", "onemklGenerate",
                 "onemklLayout", "onemklJob", "onemklJobsvd", "onemklCompz", "onemklRangev",
                 "onemklIndex", "onemklProperty", "onemklMatrixView", "onemklMatmatRequest",
                 "onemklOmatconvertAlg", "onemklOmataddAlg")
      parameters = replace(parameters, Regex("$type ([A-Za-z0-9_]+),") => SubstitutionString("convert(\\1),"))
      parameters = replace(parameters, Regex(", $type ([A-Za-z0-9_]+)") => SubstitutionString(", convert(\\1)"))
    end

    # Pass scalars (e.g. alpha/beta inputs) as references instead of values
    header = replace(header, "§" => "*")
    parameters = replace(parameters, ", §" => ", *")
    parameters = replace(parameters, ", sycl::bit_cast<sycl::half>§" => ", *reinterpret_cast<sycl::half *>")
    parameters = replace(parameters, ", static_cast<std::complex<float> >§" => ", *reinterpret_cast<std::complex<float> *>")
    parameters = replace(parameters, ", static_cast<std::complex<double> >§" => ", *reinterpret_cast<std::complex<double> *>")

    parameters = replace(parameters, r"half>([A-Za-z0-9_]+)" => s"half>(\1)")
    parameters = replace(parameters, r" >([A-Za-z0-9_]+)" => s" >(\1)")
    parameters = replace(parameters, r" \*>([A-Za-z0-9_]+)" => s"*>(\1)")
    parameters = replace(parameters, r" \*\*>([A-Za-z0-9_]+)" => s"**>(\1)")

    variant = ""
    if library == "blas"
      variant = "column_major::"
    end

    write(oneapi_cpp, "extern \"C\" $header {\n")
    if template
      type = version_types[version]
      !occursin("scratchpad_size", name) && write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name<$type>($parameters, {});\n   device_queue->val.wait_and_throw();\n")
      occursin("scratchpad_size", name)  && write(oneapi_cpp, "   int64_t scratchpad_size = oneapi::mkl::$library::$variant$name<$type>($parameters);\n   device_queue->val.wait_and_throw();\n")
      # !occursin("scratchpad_size", name) && write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name<$type>($parameters, {});\n")
      # occursin("scratchpad_size", name)  && write(oneapi_cpp, "   int64_t scratchpad_size = oneapi::mkl::$library::$variant$name<$type>($parameters);\n")
    else
      if !(name ∈ void_output)
        write(oneapi_cpp, "   auto status = oneapi::mkl::$library::$variant$name($parameters, {});\n")
        occursin("device_queue", parameters) && write(oneapi_cpp, "   device_queue->val.wait_and_throw();\n")
      else
        write(oneapi_cpp, "   oneapi::mkl::$library::$variant$name($parameters);\n")
        occursin("device_queue", parameters) && write(oneapi_cpp, "   device_queue->val.wait_and_throw();\n")
      end
    end
    if occursin("scratchpad_size", name)
      write(oneapi_cpp, "   return scratchpad_size;\n")
    else
      write(oneapi_cpp, "   return 0;\n")
    end
    write(oneapi_cpp, "}")
    write(oneapi_cpp, "\n\n")
  end
  close(oneapi_cpp)
end

# Generate "src/onemkl.h"
generate_headers("blas", blas, "onemkl_blas.h", pattern="*")
generate_headers("lapack", lapack, "onemkl_lapack.h", pattern="*")
generate_headers("sparse", sparse, "onemkl_sparse.h", pattern="*")

io = open("src/onemkl.h", "w")
headers_prologue = read("onemkl_prologue.h", String)
write(io, headers_prologue)
headers_blas = read("onemkl_blas.h", String)
write(io, "// BLAS\n")
write(io, headers_blas)
headers_lapack = read("onemkl_lapack.h", String)
write(io, "// LAPACK\n")
write(io, headers_lapack)
headers_sparse = read("onemkl_sparse.h", String)
write(io, "// SPARSE\n")
write(io, headers_sparse)
headers_epilogue = read("onemkl_epilogue.h", String)
write(io, headers_epilogue)
close(io)

# Add the version of oneMKL in src/onemkl.h
headers_onemkl = read("src/onemkl.h", String)
version_onemkl = pkgversion(oneAPI_Support_Headers_jll)
headers_onemkl = replace(headers_onemkl, "void onemkl_version" => "const int64_t ONEMKL_VERSION_MAJOR = $(version_onemkl.major);\nconst int64_t ONEMKL_VERSION_MINOR = $(version_onemkl.minor);\nconst int64_t ONEMKL_VERSION_PATCH = $(version_onemkl.patch);\nvoid onemkl_version")
write("src/onemkl.h", headers_onemkl)

# Generate "src/onemkl.cpp"
generate_cpp("blas", blas, "onemkl_blas.cpp", pattern="§")
generate_cpp("lapack", lapack, "onemkl_lapack.cpp", pattern="§")
generate_cpp("sparse", sparse, "onemkl_sparse.cpp", pattern="§")

io = open("src/onemkl.cpp", "w")
cpp_prologue = read("onemkl_prologue.cpp", String)
write(io, cpp_prologue)
cpp_blas = read("onemkl_blas.cpp", String)
write(io, "// BLAS\n")
write(io, cpp_blas)
cpp_lapack = read("onemkl_lapack.cpp", String)
write(io, "// LAPACK\n")
write(io, cpp_lapack)
cpp_sparse = read("onemkl_sparse.cpp", String)
write(io, "// SPARSE\n")
write(io, cpp_sparse)
cpp_epilogue = read("onemkl_epilogue.cpp", String)
write(io, cpp_epilogue)
close(io)
