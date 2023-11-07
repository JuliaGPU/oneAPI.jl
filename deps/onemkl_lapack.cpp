extern "C" void onemklCgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float *d, float *e, float _Complex *tauq, float _Complex *taup, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, d, e, reinterpret_cast<std::complex<float> *>tauq, reinterpret_cast<std::complex<float> *>taup, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *d, double *e, double *tauq, double *taup, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgebrd(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *d, float *e, float *tauq, float *taup, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgebrd(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double *d, double *e, double _Complex *tauq, double _Complex *taup, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gebrd(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, d, e, reinterpret_cast<std::complex<double> *>tauq, reinterpret_cast<std::complex<double> *>taup, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgerqf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgerqf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gerqf(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgeqrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrf(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrf(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetri(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetri(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetri(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetri(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri(device_queue->val, n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(b), ldb, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t *ipiv, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t *ipiv, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrs(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(b), ldb, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, double *a, int64_t lda, double *s, double *u, int64_t ldu, double *vt, int64_t ldvt, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, float *a, int64_t lda, float *s, float *u, int64_t ldu, float *vt, int64_t ldvt, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, float _Complex *a, int64_t lda, float *s, float _Complex *u, int64_t ldu, float _Complex *vt, int64_t ldvt, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, reinterpret_cast<std::complex<float> *>(a), lda, s, reinterpret_cast<std::complex<float> *>u, ldu, reinterpret_cast<std::complex<float> *>vt, ldvt, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgesvd(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, double _Complex *a, int64_t lda, double *s, double _Complex *u, int64_t ldu, double _Complex *vt, int64_t ldvt, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::gesvd(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, reinterpret_cast<std::complex<double> *>(a), lda, s, reinterpret_cast<std::complex<double> *>u, ldu, reinterpret_cast<std::complex<double> *>vt, ldvt, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heevd(device_queue->val, onemklJob jobz, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, w, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZheevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::heevd(device_queue->val, onemklJob jobz, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, w, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float *w, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hegvd(device_queue->val, itype, onemklJob jobz, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(b), ldb, w, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhegvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double *w, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hegvd(device_queue->val, itype, onemklJob jobz, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(b), ldb, w, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float *d, float *e, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, d, e, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhetrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double *d, double *e, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrd(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, d, e, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklChetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZhetrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::hetrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgbr(device_queue->val, onemklGenerate vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgbr(device_queue->val, onemklGenerate vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr(device_queue->val, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgtr(device_queue->val, convert(uplo), n, a, lda, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormtr(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDormtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormtr(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormrq(device_queue->val, onemklSide side, convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDormrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormrq(device_queue->val, onemklSide side, convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, double *tau, double *c, int64_t ldc, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormqr(device_queue->val, onemklSide side, convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSormqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, float *tau, float *c, int64_t ldc, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ormqr(device_queue->val, onemklSide side, convert(trans), m, n, k, a, lda, tau, c, ldc, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, a, lda, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotri(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potri(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(b), ldb, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrs(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(b), ldb, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevd(device_queue->val, onemklJob jobz, convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsyevd(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::syevd(device_queue->val, onemklJob jobz, convert(uplo), n, a, lda, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *b, int64_t ldb, double *w, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvd(device_queue->val, itype, onemklJob jobz, convert(uplo), n, a, lda, b, ldb, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsygvd(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *b, int64_t ldb, float *w, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sygvd(device_queue->val, itype, onemklJob jobz, convert(uplo), n, a, lda, b, ldb, w, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, double *d, double *e, double *tau, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, a, lda, d, e, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsytrd(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, float *d, float *e, float *tau, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrd(device_queue->val, convert(uplo), n, a, lda, d, e, tau, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t *ipiv, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, int64_t *ipiv, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, a, lda, ipiv, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t *ipiv, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, ipiv, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZsytrf(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t *ipiv, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::sytrf(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, ipiv, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, float _Complex *b, int64_t ldb, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(b), ldb, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, double *a, int64_t lda, double *b, int64_t ldb, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklStrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, float *a, int64_t lda, float *b, int64_t ldb, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZtrtrs(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, double _Complex *b, int64_t ldb, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::trtrs(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(b), ldb, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungbr(device_queue->val, onemklGenerate vec, m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungbr(syclQueue_t device_queue, onemklGenerate vec, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungbr(device_queue->val, onemklGenerate vec, m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungqr(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr(device_queue->val, m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungtr(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungtr(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmrq(device_queue->val, onemklSide side, convert(trans), m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>c, ldc, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZunmrq(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmrq(device_queue->val, onemklSide side, convert(trans), m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>c, ldc, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmqr(device_queue->val, onemklSide side, convert(trans), m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>c, ldc, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZunmqr(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmqr(device_queue->val, onemklSide side, convert(trans), m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>c, ldc, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, float _Complex *a, int64_t lda, float _Complex *tau, float _Complex *c, int64_t ldc, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmtr(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, reinterpret_cast<std::complex<float> *>(a), lda, reinterpret_cast<std::complex<float> *>(tau), reinterpret_cast<std::complex<float> *>c, ldc, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZunmtr(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, double _Complex *a, int64_t lda, double _Complex *tau, double _Complex *c, int64_t ldc, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::unmtr(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, reinterpret_cast<std::complex<double> *>(a), lda, reinterpret_cast<std::complex<double> *>(tau), reinterpret_cast<std::complex<double> *>c, ldc, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, reinterpret_cast<std::complex<float> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgeqrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::geqrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, reinterpret_cast<std::complex<double> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetri_batch(syclQueue_t device_queue, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetri_batch(syclQueue_t device_queue, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetri_batch(syclQueue_t device_queue, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetri_batch(syclQueue_t device_queue, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getri_batch(device_queue->val, n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, ipiv, stride_ipiv, reinterpret_cast<std::complex<float> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrs_batch(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrs_batch(device_queue->val, convert(trans), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, ipiv, stride_ipiv, reinterpret_cast<std::complex<double> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZgetrf_batch(syclQueue_t device_queue, int64_t m, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t *ipiv, int64_t stride_ipiv, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::getrf_batch(device_queue->val, m, n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, ipiv, stride_ipiv, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float *a, int64_t lda, int64_t stride_a, float *tau, int64_t stride_tau, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr_batch(device_queue->val, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDorgqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double *a, int64_t lda, int64_t stride_a, double *tau, int64_t stride_tau, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::orgqr_batch(device_queue->val, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float *a, int64_t lda, int64_t stride_a, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double *a, int64_t lda, int64_t stride_a, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, float _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrf_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, double _Complex *a, int64_t lda, int64_t stride_a, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrf_batch(device_queue->val, convert(uplo), n, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklSpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklDpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, reinterpret_cast<std::complex<float> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZpotrs_batch(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *b, int64_t ldb, int64_t stride_b, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::potrs_batch(device_queue->val, convert(uplo), n, nrhs, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, reinterpret_cast<std::complex<double> *>(b), ldb, stride_b, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklCungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, float _Complex *a, int64_t lda, int64_t stride_a, float _Complex *tau, int64_t stride_tau, int64_t batch_size, float _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr_batch(device_queue->val, m, n, k, reinterpret_cast<std::complex<float> *>(a), lda, stride_a, reinterpret_cast<std::complex<float> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<float> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" void onemklZungqr_batch(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, double _Complex *a, int64_t lda, int64_t stride_a, double _Complex *tau, int64_t stride_tau, int64_t batch_size, double _Complex *scratchpad, int64_t scratchpad_size) {
   auto status = oneapi::mkl::lapack::ungqr_batch(device_queue->val, m, n, k, reinterpret_cast<std::complex<double> *>(a), lda, stride_a, reinterpret_cast<std::complex<double> *>(tau), stride_tau, batch_size, reinterpret_cast<std::complex<double> *>(scratchpad), scratchpad_size);
   __FORCE_MKL_FLUSH__(status);
}

extern "C" int64_t onemklSgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<float>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<double>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgebrd_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<float>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<double>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgerqf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gerqf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<float>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgeqrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, lda, ldu, ldvt);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, lda, ldu, ldvt);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, lda, ldu, ldvt);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgesvd_scratchpad_size(syclQueue_t device_queue, onemklJobsvd jobu, onemklJobsvd jobvt, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
   int64_t scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(device_queue->val, onemklJobsvd jobu, onemklJobsvd jobvt, m, n, lda, ldu, ldvt);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgetrf_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<float>(device_queue->val, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<double>(device_queue->val, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<std::complex<float>>(device_queue->val, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgetri_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_scratchpad_size<std::complex<double>>(device_queue->val, n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<double>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgetrs_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<float>(device_queue->val, onemklJob jobz, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDheevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::heevd_scratchpad_size<double>(device_queue->val, onemklJob jobz, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklShegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<float>(device_queue->val, itype, onemklJob jobz, convert(uplo), n, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDhegvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hegvd_scratchpad_size<double>(device_queue->val, itype, onemklJob jobz, convert(uplo), n, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklShetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrd_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDhetrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrd_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklShetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDhetrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::hetrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgbr_scratchpad_size<float>(device_queue->val, onemklGenerate vect, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDorgbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgbr_scratchpad_size<double>(device_queue->val, onemklGenerate vect, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgtr_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDorgtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgtr_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<float>(device_queue->val, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDorgqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<double>(device_queue->val, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormrq_scratchpad_size<float>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDormrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormrq_scratchpad_size<double>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<float>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDormqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<double>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<float>(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDormtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ormtr_scratchpad_size<double>(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZpotrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<float>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<double>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZpotrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZpotri_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potri_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZsytrf_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrf_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<float>(device_queue->val, onemklJob jobz, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDsyevd_scratchpad_size(syclQueue_t device_queue, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<double>(device_queue->val, onemklJob jobz, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvd_scratchpad_size<float>(device_queue->val, itype, onemklJob jobz, convert(uplo), n, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDsygvd_scratchpad_size(syclQueue_t device_queue, int64_t itype, onemklJob jobz, onemklUplo uplo, int64_t n, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(device_queue->val, itype, onemklJob jobz, convert(uplo), n, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrd_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDsytrd_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::sytrd_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklStrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<float>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<double>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZtrtrs_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, onemklTranspose trans, onemklDiag diag, int64_t n, int64_t nrhs, int64_t lda, int64_t ldb) {
   int64_t scratchpad_size = oneapi::mkl::lapack::trtrs_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), convert(trans), convert(diag), n, nrhs, lda, ldb);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<float>(device_queue->val, onemklGenerate vect, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDungbr_scratchpad_size(syclQueue_t device_queue, onemklGenerate vect, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungbr_scratchpad_size<double>(device_queue->val, onemklGenerate vect, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_scratchpad_size<float>(device_queue->val, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDungqr_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_scratchpad_size<double>(device_queue->val, m, n, k, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungtr_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDungtr_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungtr_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<float>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDunmrq_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmrq_scratchpad_size<double>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<float>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDunmqr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklTranspose trans, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<double>(device_queue->val, onemklSide side, convert(trans), m, n, k, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmtr_scratchpad_size<float>(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDunmtr_scratchpad_size(syclQueue_t device_queue, onemklSide side, onemklUplo uplo, onemklTranspose trans, int64_t m, int64_t n, int64_t lda, int64_t ldc) {
   int64_t scratchpad_size = oneapi::mkl::lapack::unmtr_scratchpad_size<double>(device_queue->val, onemklSide side, convert(uplo), convert(trans), m, n, lda, ldc);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<float>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<double>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgetrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<float>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<double>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<float>>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgetri_batch_scratchpad_size(syclQueue_t device_queue, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<double>>(device_queue->val, n, lda, stride_a, stride_ipiv, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<float>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<double>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgetrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose trans, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t stride_ipiv, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(trans), n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<float>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<double>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgeqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<float>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<double>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZpotrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t lda, int64_t stride_a, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, lda, stride_a, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<float>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<double>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<std::complex<float>>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZpotrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo uplo, int64_t n, int64_t nrhs, int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<std::complex<double>>(device_queue->val, convert(uplo), n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSorgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_batch_scratchpad_size<float>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDorgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::orgqr_batch_scratchpad_size<double>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<float>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t stride_a, int64_t stride_tau, int64_t batch_size) {
   int64_t scratchpad_size = oneapi::mkl::lapack::ungqr_batch_scratchpad_size<double>(device_queue->val, m, n, k, lda, stride_a, stride_tau, batch_size);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_getrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrf_batch_scratchpad_size<float>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_getrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrf_batch_scratchpad_size<double>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgroup_getrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgroup_getrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_getri_batch_scratchpad_size(syclQueue_t device_queue, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getri_batch_scratchpad_size<float>(device_queue->val, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_getri_batch_scratchpad_size(syclQueue_t device_queue, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getri_batch_scratchpad_size<double>(device_queue->val, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgroup_getri_batch_scratchpad_size(syclQueue_t device_queue, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getri_batch_scratchpad_size<std::complex<float>>(device_queue->val, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgroup_getri_batch_scratchpad_size(syclQueue_t device_queue, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getri_batch_scratchpad_size<std::complex<double>>(device_queue->val, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_getrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose *trans, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrs_batch_scratchpad_size<float>(device_queue->val, onemklTranspose *trans, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_getrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose *trans, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrs_batch_scratchpad_size<double>(device_queue->val, onemklTranspose *trans, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgroup_getrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose *trans, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrs_batch_scratchpad_size<std::complex<float>>(device_queue->val, onemklTranspose *trans, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgroup_getrs_batch_scratchpad_size(syclQueue_t device_queue, onemklTranspose *trans, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_getrs_batch_scratchpad_size<std::complex<double>>(device_queue->val, onemklTranspose *trans, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_geqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_geqrf_batch_scratchpad_size<float>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_geqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_geqrf_batch_scratchpad_size<double>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgroup_geqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_geqrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgroup_geqrf_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_geqrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, m, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_orgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *k, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_orgqr_batch_scratchpad_size<float>(device_queue->val, m, n, k, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_orgqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *k, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_orgqr_batch_scratchpad_size<double>(device_queue->val, m, n, k, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_potrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrf_batch_scratchpad_size<float>(device_queue->val, onemklUplo *uplo, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_potrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrf_batch_scratchpad_size<double>(device_queue->val, onemklUplo *uplo, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgroup_potrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrf_batch_scratchpad_size<std::complex<float>>(device_queue->val, onemklUplo *uplo, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgroup_potrf_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrf_batch_scratchpad_size<std::complex<double>>(device_queue->val, onemklUplo *uplo, n, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_potrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrs_batch_scratchpad_size<float>(device_queue->val, onemklUplo *uplo, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_potrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrs_batch_scratchpad_size<double>(device_queue->val, onemklUplo *uplo, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklCgroup_potrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrs_batch_scratchpad_size<std::complex<float>>(device_queue->val, onemklUplo *uplo, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklZgroup_potrs_batch_scratchpad_size(syclQueue_t device_queue, onemklUplo *uplo, int64_t *n, int64_t *nrhs, int64_t *lda, int64_t *ldb, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_potrs_batch_scratchpad_size<std::complex<double>>(device_queue->val, onemklUplo *uplo, n, nrhs, lda, ldb, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklSgroup_ungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *k, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_ungqr_batch_scratchpad_size<float>(device_queue->val, m, n, k, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}

extern "C" int64_t onemklDgroup_ungqr_batch_scratchpad_size(syclQueue_t device_queue, int64_t *m, int64_t *n, int64_t *k, int64_t *lda, int64_t group_count, int64_t *group_sizes) {
   int64_t scratchpad_size = oneapi::mkl::lapack::group_ungqr_batch_scratchpad_size<double>(device_queue->val, m, n, k, lda, group_count, group_sizes);
   __FORCE_MKL_FLUSH__(scratchpad_size);
}
