#include <jlcxx/jlcxx.hpp>

#include <oneapi/mkl.hpp>


// auxiliary

// C++ wrapper type for Level 0 pointers.
// The layout of this struct should be identical to plain ZePtr bitstypes
template <typename T> struct ZePtr {
    typedef T value_type;
    ZePtr(T *ptr) : m_ptr(ptr) {}

    // implicit conversion to a regular pointer
    operator T *() const { return m_ptr; }

    T *m_ptr;
};


// gemm

// https://spec.oneapi.io/versions/1.0-rev-1/elements/oneMKL/source/domains/blas/gemm.html

void onemklHgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 half alpha, ZePtr<half> A, int64_t lda, ZePtr<half> B,
                 int64_t ldb, half beta, ZePtr<half> C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void onemklSgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 float alpha, ZePtr<float> A, int64_t lda, ZePtr<float> B,
                 int64_t ldb, float beta, ZePtr<float> C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void onemklDgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 double alpha, ZePtr<double> A, int64_t lda, ZePtr<double> B,
                 int64_t ldb, double beta, ZePtr<double> C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void onemklCgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 std::complex<float> alpha, ZePtr<std::complex<float>> A,
                 int64_t lda, ZePtr<std::complex<float>> B, int64_t ldb,
                 std::complex<float> beta, ZePtr<std::complex<float>> C,
                 int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

void onemklZgemm(sycl::queue device_queue, oneapi::mkl::transpose transA,
                 oneapi::mkl::transpose transB, int64_t m, int64_t n, int64_t k,
                 std::complex<double> alpha, ZePtr<std::complex<double>> A,
                 int64_t lda, ZePtr<std::complex<double>> B, int64_t ldb,
                 std::complex<double> beta, ZePtr<std::complex<double>> C,
                 int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue, transA, transB, m, n, k,
                                          alpha, A, lda, B, ldb, beta, C, ldc);
}

JLCXX_MODULE define_module_mkl(jlcxx::Module &mod) {
    using namespace jlcxx;

    mod.map_type<half>("Float16");

    // pointer type
    // TODO: use const-correct instantiations of this template
    //       (JuliaInterop/CxxWrap.jl#303)
    // TODO: map_type directly to ZePtr instead of defining ZeCxxPtr
    mod.add_type<Parametric<TypeVar<1>>>("ZeCxxPtr")
        .apply<ZePtr<half>, ZePtr<float>, ZePtr<double>,
               ZePtr<std::complex<float>>, ZePtr<std::complex<double>>>(
            [](auto wrapped) {
                typedef typename decltype(wrapped)::type WrappedT;
                wrapped.template constructor<typename WrappedT::value_type *>();
            });

    mod.add_bits<oneapi::mkl::transpose>("onemklTranspose", julia_type("CppEnum"));
    mod.set_const("ONEMKL_TRANSPOSE_NONTRANS", oneapi::mkl::transpose::nontrans);
    mod.set_const("ONEMKL_TRANSPOSE_TRANS", oneapi::mkl::transpose::trans);
    mod.set_const("ONEMLK_TRANSPOSE_CONJTRANS", oneapi::mkl::transpose::conjtrans);

    // gemm
    mod.method("onemklHgemm", onemklHgemm);
    mod.method("onemklSgemm", onemklSgemm);
    mod.method("onemklDgemm", onemklDgemm);
    mod.method("onemklCgemm", onemklCgemm);
    mod.method("onemklZgemm", onemklZgemm);
}
