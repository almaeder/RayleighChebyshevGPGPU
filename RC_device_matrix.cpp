
#include "RC_device_matrix.h"

typedef std::complex<double> CPX;

//////////////////////////////////////////////////////////
//  Required constructors
//////////////////////////////////////////////////////////
template <typename T>
RC_device_matrix<T>::RC_device_matrix()
{

}

template <typename T>
RC_device_matrix<T>::RC_device_matrix(const RC_device_matrix<T> &W)
{
    resize(W.m, W.n);
    initialize(W);
}

//////////////////////////////////////////////////////////
// Constructor not required for RayleighChebyshve
// but useful for creating test program
//////////////////////////////////////////////////////////

template <typename T>
RC_device_matrix<T>::RC_device_matrix(RC_INT m, RC_INT n)
{
    resize(m, n);
}

template <typename T>
RC_device_matrix<T>::RC_device_matrix(RC_INT m, RC_INT n, std::vector<T> data)
{
    resize(m, n);
    this->mData = data;
    this->host_to_device_copy();
}

template <typename T>
RC_device_matrix<T>::~RC_device_matrix()
{
    if (matrix_desc != NULL && mData_d != NULL){
        cusparseDestroyDnMat(matrix_desc);
        cudaErrchk(cudaFree(mData_d));
        matrix_desc = NULL;
        mData_d = NULL;
    }
    else if (matrix_desc == NULL && mData_d == NULL){
        return;
    }
    else{
        std::cout << "matrix_desc xor mData_d is Null" << std::endl;
    }

    if (_eig_values_d != NULL)
    {
        cudaErrchk(cudaFree(_eig_values_d));
        _eig_values_d = NULL;
    }
    if (_eig_residuals_d != NULL)
    {
        cudaErrchk(cudaFree(_eig_residuals_d));
        _eig_residuals_d = NULL;
    }
    if (tau_d != NULL)
    {
        cudaErrchk(cudaFree(tau_d));
        tau_d = NULL;
    }
    if (geqrf_work_d != NULL)
    {
        cudaErrchk(cudaFree(geqrf_work_d));
        geqrf_work_d = NULL;
    }
    if (gqr_work_d != NULL)
    {
        cudaErrchk(cudaFree(gqr_work_d));
        gqr_work_d = NULL;
    }
    if (info_d != NULL)
    {
        cudaErrchk(cudaFree(info_d));
        info_d = NULL;
    }


}

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////

template <typename T>
void RC_device_matrix<T>::host_to_device_copy()
{
    if (mData_d != NULL){
        cudaErrchk(cudaMemcpy(mData_d, mData.data(), this->m*this->n*sizeof(T), cudaMemcpyHostToDevice));
    }
    else{
        // raise exception
        throw std::runtime_error("Error: mData_d is NULL");
    }
}

template <typename T>
void RC_device_matrix<T>::device_to_host_copy()
{
    if (mData_d != NULL){
        cudaErrchk(cudaMemcpy(mData.data(), mData_d, this->get_size()*sizeof(T), cudaMemcpyDeviceToHost));
    }
    else{
        // raise exception
        throw std::runtime_error("Error: mData_d is NULL");
    }
}

template <typename T>
void RC_device_matrix<T>::device_to_device_copy(const RC_device_matrix<T> &W)
{
    // this is m by n
    // W is m by n
    // this = W

    if (this->m != W.m || this->n != W.n)
    {
        throw std::runtime_error("Error: matrix sizes do not match");
    }

    cudaErrchk(cudaMemcpy(mData_d,
                        W.mData_d, sizeof(T) * this->get_size(), cudaMemcpyDeviceToDevice));

}

template <typename T>
void RC_device_matrix<T>::initialize(const RC_device_matrix<T> &W)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < W.n; i++)
    {
        for (size_t j = 0; j < W.m; j++)
        {
            mData[i * W.m + j] = W.mData[i * W.m + j];
        }
    }
}

template <typename T>
void RC_device_matrix<T>::initialize(const RCvector<T> &V, RC_INT n)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < this->m; j++)
        {
            mData[i * this->m + j] = V.vData[j];
        }
    }
}

template <typename T>
void RC_device_matrix<T>::orthogonalize()
{
    if constexpr (std::is_same<T, double>::value)
    {
        cusolverErrchk(cusolverDnDgeqrf(
            cusolverDn_handle,
            this->m,
            this->n,
            this->mData_d,
            this->m,
            tau_d,
            geqrf_work_d,
            geqrf_lwork,
            info_d));

        cusolverErrchk(cusolverDnDorgqr(
            cusolverDn_handle,
            this->m,
            this->n,
            this->n,
            this->mData_d,
            this->m,
            tau_d,
            gqr_work_d,
            gqr_lwork,
            info_d));
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        cusolverErrchk(cusolverDnZgeqrf(
            cusolverDn_handle,
            this->m,
            this->n,
            (cuDoubleComplex *)this->mData_d,
            this->m,
            (cuDoubleComplex *)tau_d,
            (cuDoubleComplex *)geqrf_work_d,
            geqrf_lwork,
            info_d));
        cusolverErrchk(cusolverDnZungqr(
            cusolverDn_handle,
            this->m,
            this->n,
            this->n,
            (cuDoubleComplex *)this->mData_d,
            this->m,
            (cuDoubleComplex *)tau_d,
            (cuDoubleComplex *)gqr_work_d,
            gqr_lwork,
            info_d));
    }
}

template <typename T>
RC_device_matrix<T> &RC_device_matrix<T>::operator=(const RC_device_matrix<T> &W)
{
    if (this == &W)
    {
        return *this; // Check for self-assignment
    }

    resize(W.m, W.n);
    initialize(W);

    return *this; // Enable chaining
}

template <typename T>
void RC_device_matrix<T>::normalize()
{
    cuda_kernels::normalize(
        this->mData_d,
        this->m,
        this->n
    );
}

template <typename T>
T RC_device_matrix<T>::inner_product(const RC_INT k, const RC_INT l) const
{
    return _innerprod(this, k, l);
}

CPX _innerprod(const RC_device_matrix<CPX> *matrix, const RC_INT k, const RC_INT l)
{

    RC_INT m = matrix->get_row_size();
    RC_INT n = matrix->get_col_size();

#ifdef _OPENMP
#pragma omp declare reduction(complex_add : std::complex<double> : omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0, 0))
#endif

    std::complex<double> normSquared = std::complex<double>(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(complex_add : normSquared)
#endif
    for (size_t j = 0; j < m; j++)
    {
        normSquared += matrix->mData[k * m + j] * std::conj(matrix->mData[l * m + j]);
    }

    return normSquared;
}

double _innerprod(const RC_device_matrix<double> *matrix, const RC_INT k, const RC_INT l)
{

    RC_INT m = matrix->get_row_size();
    RC_INT n = matrix->get_col_size();

    double normSquared = double(0.0);
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : normSquared)
#endif
    for (size_t j = 0; j < m; j++)
    {
        normSquared += matrix->mData[k * m + j] * matrix->mData[l * m + j];
    }

    return normSquared;
}


template <typename T>
void RC_device_matrix<T>::resize_cols(RC_INT n)
{
    mData.resize(this->m * n);
    this->n = n;
    _create_cuda_memory();
    _create_cuda_descriptor();
    _create_cuda_handles();
    _allocate_qr_memory();
}

template <typename T>
void RC_device_matrix<T>::resize_cols(RC_INT n, T value)
{
    if (n == this->n)
    {
        for (size_t i = 0; i < mData.size(); i++)
        {
            mData[i] = value;
        }
    }
    else
    {
        mData.resize(this->m * n, value);
        this->n = n;
        _create_cuda_memory();
        _create_cuda_descriptor();
        _create_cuda_handles();
        _allocate_qr_memory();
    }
}

template <typename T>
void RC_device_matrix<T>::resize_cols(RC_INT n, RCvector<T> &V)
{
    if (n == this->n)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < this->m; j++)
            {
                mData[i * this->m + j] = V[j];
            }
        }
    }
    else
    {
        mData.resize(this->m * n);
        this->n = n;
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < this->m; j++)
            {
                mData[i * this->m + j] = V[j];
            }
        }
        _create_cuda_memory();
        _create_cuda_descriptor();
        _create_cuda_handles();
        _allocate_qr_memory();
    }
}

template <typename T>
void RC_device_matrix<T>::resize(RC_INT m, RC_INT n)
{
    mData.resize(m * n);
    this->n = n;
    this->m = m;
    _create_cuda_memory();
    _create_cuda_descriptor();
    _create_cuda_handles();
    _allocate_qr_memory();
}

template <typename T>
void RC_device_matrix<T>::matmult(const RC_device_matrix<T> &A, const RC_device_matrix<T> &B, T alpha, T beta)
{
    // this is m by n
    // A is m by k
    // B is k by n
    // this = alpha * A * B + beta * this
    if (this->m != A.m || this->n != B.n || A.n != B.m)
    {
        throw std::runtime_error("Error: matrix sizes do not match");
    }

    if constexpr (std::is_same<T, double>::value)
    {
        cublasErrchk(
            cublasDgemm(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->m, this->n, A.n,
                &alpha,
                A.mData_d, this->m,
                B.mData_d, A.n,
                &beta,
                this->mData_d, this->m));
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        cublasErrchk(
            cublasZgemm(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->m, this->n, A.n,
                (cuDoubleComplex *)&alpha,
                (cuDoubleComplex *)A.mData_d, this->m,
                (cuDoubleComplex *)B.mData_d, A.n,
                (cuDoubleComplex *)&beta,
                (cuDoubleComplex *)this->mData_d, this->m));
    }
    else
    {
        throw std::runtime_error("Error: innerprod_complex not defined for this type");
    }
}

template <typename T>
void RC_device_matrix<T>::matmult(const RC_device_matrix<T> &A, const RC_device_matrix<T> &B, T alpha, T beta, std::string conj_A, std::string conj_B)
{
    // this is m by n
    // A is m by k
    // B is k by n
    // this = alpha * A * B + beta * this

    if (conj_A != "C" | conj_B != "N"){
        throw std::runtime_error("Error: invalid conjugation type for A or B, not implemented");
    }

    RC_INT k = A.n;
    if (conj_A == "C" | conj_A == "T")
    {
        k = A.m;
    }

    else
    {
        throw std::runtime_error("Error: invalid conjugation type for A");
    }

    if (conj_A == "N")
    {
        if (this->m != A.m)
        {
            throw std::runtime_error("Error: matrix sizes do not match");
        }
    }

    else if (conj_A == "C")
    {
        if (this->m != A.n)
        {
            throw std::runtime_error("Error: matrix sizes do not match");
        }
    }

    if (conj_B == "N")
    {
        if ((this->n != B.n) | (k != B.m))
        {
            throw std::runtime_error("Error: matrix sizes do not match");
        }
    }
    else if (conj_B == "C")
    {
        if ((this->n != B.m) | (k != B.n))
        {
            throw std::runtime_error("Error: matrix sizes do not match");
        }
    }
    else
    {
        throw std::runtime_error("Error: invalid conjugation type for B");
    }

    if constexpr (std::is_same<T, double>::value)
    {
        cublasErrchk(
            cublasDgemm(
                cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                this->m, this->n, k,
                &alpha,
                A.mData_d, A.m,
                B.mData_d, B.m,
                &beta,
                this->mData_d, this->m));
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        cublasErrchk(
            cublasZgemm(
                cublas_handle,
                CUBLAS_OP_C, CUBLAS_OP_N,
                this->m, this->n, k,
                (cuDoubleComplex *)&alpha,
                (cuDoubleComplex *)A.mData_d, A.m,
                (cuDoubleComplex *)B.mData_d, B.m,
                (cuDoubleComplex *)&beta,
                (cuDoubleComplex *)this->mData_d, this->m));
    }
}

template <typename T>
void RC_device_matrix<T>::residuals(const RC_device_matrix<T> &OpA, const std::vector<double> &eig_values, std::vector<double> &eig_residuals, RC_INT residualCheckCount)
{

    if (residualCheckCount > this->n)
    {
        throw std::runtime_error("Error: residualCheckCount is greater than the number of columns in the matrix");
    }
    if (residualCheckCount != eig_residuals.size())
    {
        throw std::runtime_error("Error: residualCheckCount does not match the size of eig_residuals");
    }
    // this is m by n
    // OpA is m by n
    if (this->m != OpA.m || this->n != OpA.n)
    {
        throw std::runtime_error("Error: matrix sizes do not match");
    }

    if (_old_residualCheckCount != residualCheckCount)
    {
        if (_eig_residuals_d != NULL)
        {
            cudaErrchk(cudaFree(_eig_residuals_d));
        }
        cudaErrchk(cudaMalloc((void **)&_eig_residuals_d, sizeof(double) * residualCheckCount));

        if (_eig_values_d != NULL)
        {
            cudaErrchk(cudaFree(_eig_values_d));
        }
        cudaErrchk(cudaMalloc((void **)&_eig_values_d, sizeof(double) * residualCheckCount));

        _old_residualCheckCount = residualCheckCount;
    }

    cudaErrchk(cudaMemcpy(_eig_values_d,
                            eig_values.data(), sizeof(double) * residualCheckCount, cudaMemcpyHostToDevice));

    cuda_kernels::residuals(
        _eig_residuals_d,
        OpA.mData_d,
        mData_d,
        _eig_values_d,
        this->m,
        residualCheckCount);

    cudaErrchk(cudaMemcpy(eig_residuals.data(),
                            _eig_residuals_d, sizeof(double) * residualCheckCount, cudaMemcpyDeviceToHost));
}

template <typename T>
void RC_device_matrix<T>::substract(const RC_device_matrix<T> &A){
    if (this->m != A.m || this->n != A.n)
    {
        throw std::runtime_error("Error: matrix sizes do not match");
    }

    cuda_kernels::substract(mData_d, A.mData_d, this->get_size());

}

template <typename T>
template <typename T1>
void RC_device_matrix<T>::scale(const T1 alpha)
{
    T alphat;
    if constexpr (std::is_same<T, T1>::value) {
        alphat = alpha;
    } else if constexpr (std::is_same<T, std::complex<double>>::value && std::is_same<T1, double>::value) {
        alphat = std::complex<double>(alpha, 0.0);
    } else {
        throw std::runtime_error("Error: scale not defined for this type");
    }

   cuda_kernels::scale(mData_d, this->get_size(), alphat);
}


template <typename T>
void RC_device_matrix<T>::_create_cuda_handles(){
    cusolverDn_handle = cudaHandles::getInstance()->cusolverDn_handle;
    cublas_handle = cudaHandles::getInstance()->cublas_handle;
}

template <typename T>
void RC_device_matrix<T>::_create_cuda_memory(){
    if (mData_d != NULL){
        cudaFree(mData_d);
    }
    if(this->m > 0 && this->n > 0){
        cudaMalloc((void**)&mData_d,this->get_size()*sizeof(T));
    }
    else{
        throw std::runtime_error("Error: m or n are zero");
    }
}

template <typename T>
void RC_device_matrix<T>::_create_cuda_memory(const RC_device_matrix<T>& W){
    if (W.m != this->m || W.n != this->n){
        throw std::runtime_error("Error: matrix sizes do not match");
    }

    if (mData_d != NULL){
        cudaFree(mData_d);
    }
    if(this->m > 0 && this->n > 0){
        cudaMalloc((void**)&mData_d, this->get_size()*sizeof(T));
        cudaMemcpy(mData_d, W.mData_d, this->get_size()*sizeof(T), cudaMemcpyDeviceToDevice);
    }
    else{
        throw std::runtime_error("Error: m or n are zero");
    }
}

template <typename T>
void RC_device_matrix<T>::_create_cuda_descriptor(){
    if (matrix_desc != NULL && mData_d != NULL){
        cusparseDestroyDnMat(matrix_desc);
        if constexpr (std::is_same<T, double>::value){
            cusparseCreateDnMat(&matrix_desc, this->m, this->n, this->m, mData_d, CUDA_R_64F, CUSPARSE_ORDER_COL);
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value){
            cusparseCreateDnMat(&matrix_desc, this->m, this->n, this->m, mData_d, CUDA_C_64F, CUSPARSE_ORDER_COL);
        }
    }
    else if (mData_d != NULL){
        if constexpr (std::is_same<T, double>::value){
            cusparseCreateDnMat(&matrix_desc, this->m, this->n, this->m, mData_d, CUDA_R_64F, CUSPARSE_ORDER_COL);
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value){
            cusparseCreateDnMat(&matrix_desc, this->m, this->n, this->m, mData_d, CUDA_C_64F, CUSPARSE_ORDER_COL);
        }
    }
    else if (matrix_desc == NULL && mData_d == NULL){
        return;
    }
    else{
        // raise exception
        throw std::runtime_error("Error: mData_d is NULL, but not matrix_desc");
    }
}

template <typename T>
void RC_device_matrix<T>::_allocate_qr_memory(){
    if (tau_d == NULL)
    {
        cudaErrchk(cudaMalloc((void **)&tau_d, sizeof(T) * this->n));
    }
    else if (this->n > tau_n)
    {
        cudaErrchk(cudaFree(tau_d));
        cudaErrchk(cudaMalloc((void **)&tau_d, sizeof(T) * this->n));
        tau_n = this->n;
    }

    if (this->m > geqrf_m || this->n > geqrf_n)
    {
        cudaErrchk(cudaFree(geqrf_work_d));
        cudaErrchk(cudaFree(gqr_work_d));

        geqrf_m = this->m;
        geqrf_n = this->n;
    }

    if (geqrf_work_d == NULL || (this->m > geqrf_m || this->n > geqrf_n))
    {

        if constexpr (std::is_same<T, double>::value)
        {
            cusolverErrchk(cusolverDnDgeqrf_bufferSize(
                cusolverDn_handle,
                this->m,
                this->n,
                this->mData_d,
                this->m,
                &geqrf_lwork));

            cusolverErrchk(cusolverDnDorgqr_bufferSize(
                cusolverDn_handle,
                this->m,
                this->n,
                this->n,
                this->mData_d,
                this->m,
                tau_d,
                &gqr_lwork));

            cudaErrchk(cudaMalloc((void **)&geqrf_work_d, sizeof(T) * geqrf_lwork));
            cudaErrchk(cudaMalloc((void **)&gqr_work_d, sizeof(T) * gqr_lwork));
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            cusolverErrchk(cusolverDnZgeqrf_bufferSize(
                cusolverDn_handle,
                this->m,
                this->n,
                (cuDoubleComplex *)this->mData_d,
                this->m,
                &geqrf_lwork));

            cusolverErrchk(
                cusolverDnZungqr_bufferSize(
                    cusolverDn_handle,
                    this->m,
                    this->n,
                    this->n,
                    (cuDoubleComplex *)this->mData_d,
                    this->m,
                    (cuDoubleComplex *)tau_d,
                    &gqr_lwork));
            cudaErrchk(cudaMalloc((void **)&geqrf_work_d, sizeof(T) * geqrf_lwork));
            cudaErrchk(cudaMalloc((void **)&gqr_work_d, sizeof(T) * gqr_lwork));
        }
    }

    if (info_d == NULL)
    {
        cudaErrchk(cudaMalloc((void **)&info_d, sizeof(int)));
    }

}


template void RC_device_matrix<double>::scale<double>(const double);
template void RC_device_matrix<CPX>::scale<double>(const double);
template void RC_device_matrix<CPX>::scale<CPX>(const CPX);

template class RC_device_matrix<double>;
template class RC_device_matrix<CPX>;


template <typename T>
RC_device_randomize<T>::RC_device_randomize()
{
    seed = 3141592;
    randomGenerator.seed(seed);

    // Initialize the distribution to be uniform in the interval [-1,1]
    std::uniform_real_distribution<double>::param_type distParams(-1.0, 1.0);
    distribution.param(distParams);
}

template <typename T>
void RC_device_randomize<T>::randomize(RCvector<T> &V)
{
    for (size_t i = 0; i < V.get_size(); i++)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            V.vData[i] = distribution(randomGenerator);
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            std::complex<double> random_complex(distribution(randomGenerator), distribution(randomGenerator));
            V.vData[i] = random_complex;
        }
        else
        {
            throw std::runtime_error("Error: randomize not defined for this type");
        }
    }
}

template <typename T>
void RC_device_randomize<T>::randomize(RC_device_matrix<T> &M)
{
    for (size_t i = 0; i < M.get_size(); i++)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            M.mData[i] = distribution(randomGenerator);
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            std::complex<double> random_complex(distribution(randomGenerator), distribution(randomGenerator));
            M.mData[i] = random_complex;
        }
        else
        {
            throw std::runtime_error("Error: randomize not defined for this type");
        }
    }
}

template class RC_device_randomize<double>;
template class RC_device_randomize<CPX>;
