
#include "RC_host_matrix.h"

typedef std::complex<double> CPX;

//////////////////////////////////////////////////////////
//  Required constructors
//////////////////////////////////////////////////////////
template<typename T>
RC_host_matrix<T>::RC_host_matrix()
{
}

template<typename T>
RC_host_matrix<T>::RC_host_matrix(const RC_host_matrix<T>& W)
{
    resize(W.m, W.n);
    initialize(W);
}

//////////////////////////////////////////////////////////
// Constructor not required for RayleighChebyshve
// but useful for creating test program
//////////////////////////////////////////////////////////

template<typename T>
RC_host_matrix<T>::RC_host_matrix(RC_INT m, RC_INT n)
{
    resize(m,n);
}

template<typename T>
RC_host_matrix<T>::RC_host_matrix(RC_INT m, RC_INT n, std::vector<T> data) {
    this->mData = data;
    this->n = n;
    this->m = m;
}

template<typename T>
RC_host_matrix<T>::~RC_host_matrix(){
}

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////

template<typename T>
void RC_host_matrix<T>::host_to_device_copy(){
}

template<typename T>
void RC_host_matrix<T>::device_to_host_copy(){
}

template<typename T>
void RC_host_matrix<T>::initialize(const RC_host_matrix<T>& W)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for(size_t i = 0; i < W.n; i++)
    {
        for(size_t j = 0; j < W.m; j++)
        {
            mData[i*W.m + j] = W.mData[i*W.m + j];
        }
    }
}

template<typename T>
void RC_host_matrix<T>::initialize(const RCvector<T>& V, RC_INT n)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < this->m; j++)
        {
            mData[i*this->m + j] = V.vData[j];
        }
    }

}

template<typename T>
void RC_host_matrix<T>::orthogonalize(){
}

template<typename T>
RC_host_matrix<T>& RC_host_matrix<T>::operator=(const RC_host_matrix<T>& W) {
    if (this == &W) {
        return *this;  // Check for self-assignment
    }

    resize(W.m, W.n);
    initialize(W);

    return *this;  // Enable chaining
}


template<typename T>
void RC_host_matrix<T>::normalize()
{	
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int k = 0; k < this->n; k++)
    {
        T normSquared = T(0.0);
        for(size_t j = 0; j < this->m; j++)
        {
            if constexpr (std::is_same<T, std::complex<double>>::value)
                normSquared += mData[k*this->m + j]*std::conj(mData[k*this->m + j]);
            else if constexpr (std::is_same<T, double>::value)
                normSquared += mData[k*this->m + j]*mData[k*this->m + j];
            else
                throw std::runtime_error("Error: innerprod_complex not defined for this type");
        }

        normSquared = std::sqrt(std::abs(normSquared));

        for(size_t j = 0; j < this->m; j++)
        {
            mData[k*this->m + j] /= normSquared;
        }
    }
}

template <typename T>
template <typename T1>
T1 RC_host_matrix<T>::innerprod(const RC_INT k, const RC_INT l) const
{
    if constexpr (std::is_same<T1, std::complex<double>>::value)
    {
        return innerprod_complex(this,  k,l);
    }
    else if constexpr (std::is_same<T1, double>::value)
    {
        return innerprod_real(this,  k,l);
    }
    else
    {
        throw std::runtime_error("Error: innerprod_complex not defined for this type");
    }
}

template <typename T>
std::complex<double> innerprod_complex(RC_host_matrix<T> matrix, const RC_INT k, const RC_INT l)
{

    RC_INT m = matrix.get_row_size();
    RC_INT n = matrix.get_col_size();

#ifdef _OPENMP
    #pragma omp declare reduction \
    (complex_add:std::complex<double>: \
    omp_out += omp_in) \
    initializer(omp_priv = std::complex<double>(0, 0))
#endif

    std::complex<double> normSquared = std::complex<double>(0.0);
#ifdef _OPENMP
    #pragma omp parallel for reduction(complex_add:normSquared)
#endif
    for(size_t j = 0; j < m; j++)
    {
        normSquared += matrix.mData[k*m + j]*std::conj(matrix.mData[l*m + j]);
    }

    return normSquared;

}

template <typename T>
double innerprod_real(RC_host_matrix<T> matrix, const RC_INT k, const RC_INT l)
{

    RC_INT m = matrix.get_row_size();
    RC_INT n = matrix.get_col_size();

    double normSquared = double(0.0);
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:normSquared)
#endif
    for(size_t j = 0; j < m; j++)
    {
        normSquared += matrix.mData[k*m + j]*matrix.mData[l*m + j];
    }

    return normSquared;

}


template<typename T>
void RC_host_matrix<T>::resize_rows(RC_INT n)
{
    mData.resize(this->m*n);
    this->n = n;
}

template<typename T>
void RC_host_matrix<T>::resize_rows(RC_INT n, T value)
{
    if(n == this->n)
    {
        for(size_t i = 0; i < mData.size(); i++)
        {
            mData[i] = value;
        }
    }
    else{
        mData.resize(this->m*n, value);
        this->n = n;
    }
}

template<typename T>
void RC_host_matrix<T>::resize_rows(RC_INT n, RCvector<T>& V)
{
    if(n == this->n)
    {
        #pragma omp parallel for collapse(2)
        for(size_t i = 0; i < n; i++)
        {
            for(size_t j = 0; j < this->m; j++)
            {
                mData[i*this->m + j] = V[j];
            }
        }
    }
    else{
        mData.resize(this->m*n);
        this->n = n;
        #pragma omp parallel for collapse(2)
        for(size_t i = 0; i < n; i++)
        {
            for(size_t j = 0; j < this->m; j++)
            {
                mData[i*this->m + j] = V[j];
            }
        }
    }
}

template<typename T>
void RC_host_matrix<T>::resize(RC_INT m, RC_INT n)
{
    mData.resize(m*n);
    this->n = n;
    this->m = m;
}

template class RC_host_matrix<double>;
template class RC_host_matrix<CPX>;