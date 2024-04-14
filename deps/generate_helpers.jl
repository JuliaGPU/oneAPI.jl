non_parametric_routines = ["init_matrix_handle", "release_matrix_handle", "set_matrix_property",
"init_matmat_descr", "release_matmat_descr", "set_matmat_data", "get_matmat_data", "matmat",
"omatcopy", "sort_matrix", "optimize_gemv", "optimize_trmv", "optimize_trsv", "optimize_trsm"]

function analyzer_template(library::String, cpp_headers::String, name_routine::String)
  list_parameters = Vector{String}[]
  list_types = Vector{String}[]
  list_versions = String[]
  list_suffix = String[]

  if (library == "blas") || (library == "sparse" && !(name_routine ∈ non_parametric_routines))
    prefix = (library == "sparse") ? "SPARSE_" : "BUF_"

    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(T)", cpp_headers) && (list_parameters = ["T"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(FpType)", cpp_headers) && (list_parameters = ["FpType"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(Tf, Ti)", cpp_headers) && (list_parameters = ["Tf", "Ti"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(T, Ts)", cpp_headers) && (list_parameters = ["T", "Ts"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(IntType, FpType)", cpp_headers) && (list_parameters = ["IntType", "FpType"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(Ta, Tb, Tc, Ts)", cpp_headers) && (list_parameters = ["Ta", "Tb", "Tc", "Ts"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(T, Tres)", cpp_headers) && (list_parameters = ["T", "Tres"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(T, Treal)", cpp_headers) && (list_parameters = ["T", "Treal"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(T, Tc, Ts)", cpp_headers) && (list_parameters = ["T", "Tc", "Ts"])
    occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))(T, Tc)", cpp_headers) && (list_parameters = ["T", "Tc"])
    
    (list_parameters == []) && @warn("Unable to determine the parametric parameters of $(name_routine).")
    
    for (type, version, suffix) in [(["sycl::half"], "H", ""),
                                    (["float"], "S", ""),
                                    (["double"], "D", ""),
                                    (["std::complex<float>"], "C", ""),
                                    (["std::complex<double>"], "Z", "")]
      if occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))($(type[1]))", cpp_headers)
        push!(list_types, type)
        push!(list_versions, version)
        push!(list_suffix, suffix)
      end
    end
    
    for (type, version, suffix) in [(["int32_t","float"], "S", ""),
                                    (["int64_t","float"], "S", "_64"),
                                    (["int32_t","double"], "D", ""),
                                    (["int64_t","double"], "D", "_64"),
                                    (["int32_t","std::complex<float>"], "C", ""),
                                    (["int64_t","std::complex<float>"], "C", "_64"),
                                    (["int32_t","std::complex<double>"], "Z", ""),
                                    (["int64_t","std::complex<double>"], "Z", "_64"),
                                    (["float","int32_t"], "S", ""),
                                    (["float","int64_t"], "S", "_64"),
                                    (["double","int32_t"], "D", ""),
                                    (["double","int64_t"], "D", "_64"),
                                    (["std::complex<float>","int32_t"], "C", ""),
                                    (["std::complex<float>","int64_t"], "C", "_64"),
                                    (["std::complex<double>","int32_t"], "Z", ""),
                                    (["std::complex<double>","int64_t"], "Z", "_64"),
                                    (["sycl::half","sycl::half"], "H", ""),
                                    (["float","float"], "S", ""),
                                    (["double","double"], "D", ""),
                                    (["std::complex<float>","float"], "CS", ""),
                                    (["std::complex<double>","double"], "ZD", ""),
                                    (["std::complex<float>","std::complex<float>"], "C", ""),
                                    (["std::complex<double>","std::complex<double>"], "Z", "")]
      if occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))($(type[1]), $(type[2]))", cpp_headers)
        push!(list_types, type)
        push!(list_versions, version)
        push!(list_suffix, suffix)
      end
    end
    
    for (type, version, suffix) in [(["sycl::half","sycl::half","sycl::half"], "H", ""),
                                    (["float","float","float"], "S", ""),
                                    (["double","double","double"], "D", ""),
                                    (["std::complex<float>","float","float"], "CS", ""),
                                    (["std::complex<float>","float", "std::complex<float>"], "C", ""),
                                    (["std::complex<double>","double","double"], "ZD", ""),
                                    (["std::complex<double>","double","std::complex<double>"], "Z", "")]
      if occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))($(type[1]), $(type[2]), $(type[3]))", cpp_headers)
        push!(list_types, type)
        push!(list_versions, version)
        push!(list_suffix, suffix)
      end
    end
    
    for (type, version, suffix) in [(["sycl::half","sycl::half","sycl::half","sycl::half"], "H", ""),
                                    (["float","float","float","float"], "S", ""),
                                    (["double","double","double","double"], "D", ""),
                                    (["std::complex<float>","std::complex<float>","std::complex<float>","std::complex<float>"], "C", ""),
                                    (["std::complex<double>","std::complex<double>","std::complex<double>","std::complex<double>"], "Z", "")]
      if occursin("ONEMKL_DECLARE_$(prefix)$(uppercase(name_routine))($(type[1]), $(type[2]), $(type[3]), $(type[4]))", cpp_headers)
        push!(list_types, type)
        push!(list_versions, version)
        push!(list_suffix, suffix)
      end
    end
  end

  return list_parameters, list_types, list_versions, list_suffix
end
