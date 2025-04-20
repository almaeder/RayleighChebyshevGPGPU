
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
    this->mData = data;
    this->n = n;
    this->m = m;
}

template <typename T>
RC_device_matrix<T>::~RC_device_matrix()
{
}

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////

template <typename T>
void RC_device_matrix<T>::host_to_device_copy()
{
}

template <typename T>
void RC_device_matrix<T>::device_to_host_copy()
{
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

    for (size_t i = 0; i < this->m; i++)
    {
        for (size_t j = 0; j < this->n; j++)
        {
            operator()(i, j) = W(i, j);
        }
    }
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
    // orthogonalize the columns of the matrix
    // wiith modified gram schmidt

    for (long k = 1; k <= this->n; k++)
    {

        auto rkk = std::sqrt(std::abs(this->inner_product(k - 1, k - 1)));

        _scale(k - 1, 1.0 / rkk);
        for (long j = k + 1; j <= this->n; j++)
        {
            auto rkj = this->inner_product(j - 1, k - 1);

            _scale_add(j - 1, k - 1, -rkj);
        }
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < this->n; k++)
    {
        T normSquared = T(0.0);
        for (size_t j = 0; j < this->m; j++)
        {
            if constexpr (std::is_same<T, std::complex<double>>::value)
                normSquared += mData[k * this->m + j] * std::conj(mData[k * this->m + j]);
            else if constexpr (std::is_same<T, double>::value)
                normSquared += mData[k * this->m + j] * mData[k * this->m + j];
            else
                throw std::runtime_error("Error: innerprod_complex not defined for this type");
        }

        normSquared = std::sqrt(std::abs(normSquared));

        for (size_t j = 0; j < this->m; j++)
        {
            mData[k * this->m + j] /= normSquared;
        }
    }
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
void RC_device_matrix<T>::_scale(const RC_INT k, const T alpha)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < this->m; j++)
    {
        mData[k * this->m + j] *= alpha;
    }
}

template <typename T>
void RC_device_matrix<T>::_scale_add(const RC_INT k, const RC_INT l, const T alpha)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < this->m; j++)
    {
        mData[k * this->m + j] += alpha * mData[l * this->m + j];
    }
}

template <typename T>
void RC_device_matrix<T>::resize_rows(RC_INT n)
{
    mData.resize(this->m * n);
    this->n = n;
}

template <typename T>
void RC_device_matrix<T>::resize_rows(RC_INT n, T value)
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
    }
}

template <typename T>
void RC_device_matrix<T>::resize_rows(RC_INT n, RCvector<T> &V)
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
    }
}

template <typename T>
void RC_device_matrix<T>::resize(RC_INT m, RC_INT n)
{
    mData.resize(m * n);
    this->n = n;
    this->m = m;
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

    for (size_t i = 0; i < this->m; i++)
    {
        for (size_t j = 0; j < this->n; j++)
        {
            operator()(i, j) = beta * operator()(i, j);
            for (size_t k = 0; k < A.n; k++)
            {
                operator()(i, j) += alpha * A(i, k) * B(k, j);
            }
        }
    }
}

template <typename T>
void RC_device_matrix<T>::matmult(const RC_device_matrix<T> &A, const RC_device_matrix<T> &B, T alpha, T beta, std::string conj_A, std::string conj_B)
{
    // this is m by n
    // A is m by k
    // B is k by n
    // this = alpha * A * B + beta * this

    RC_INT kdim = A.n;
    if (conj_A == "C" | conj_A == "T")
    {
        kdim = A.m;
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
        if ((this->n != B.n) | (kdim != B.m))
        {
            throw std::runtime_error("Error: matrix sizes do not match");
        }
    }
    else if (conj_B == "C")
    {
        if ((this->n != B.m) | (kdim != B.n))
        {
            throw std::runtime_error("Error: matrix sizes do not match");
        }
    }
    else
    {
        throw std::runtime_error("Error: invalid conjugation type for B");
    }

    for (size_t i = 0; i < this->m; i++)
    {
        for (size_t j = 0; j < this->n; j++)
        {
            operator()(i, j) = beta * operator()(i, j);
            for (size_t k = 0; k < kdim; k++)
            {
                if (conj_A == "N" & conj_B == "N")
                {
                    operator()(i, j) += alpha * A(i, k) * B(k, j);
                }
                else if (conj_A == "C" & conj_B == "N")
                {
                    if constexpr (std::is_same<T, std::complex<double>>::value)
                    {
                        operator()(i, j) += alpha * std::conj(A(k, i)) * B(k, j);
                    }
                    // else
                    //     operator()(i, j) += alpha * A(k,i) * B(k,j);
                    continue;
                }
                else if (conj_A == "N" & conj_B == "C")
                {
                    if constexpr (std::is_same<T, std::complex<double>>::value)
                        operator()(i, j) += alpha * A(i, k) * std::conj(B(j, k));
                    else
                        operator()(i, j) += alpha * A(i, k) * B(j, k);
                }
                else if (conj_A == "C" & conj_B == "C")
                {
                    if constexpr (std::is_same<T, std::complex<double>>::value)
                        operator()(i, j) += alpha * std::conj(A(k, i)) * std::conj(B(j, k));
                    else
                        operator()(i, j) += alpha * A(k, i) * B(j, k);
                }
            }
        }
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

    for (int i = 0; i < residualCheckCount; i++)
    {
        T normSquared = T(0.0);
        for (size_t j = 0; j < this->m; j++)
        {
            if constexpr (std::is_same<T, std::complex<double>>::value)
                normSquared += (OpA(i, j) - eig_values[i] * operator()(i, j)) * std::conj(OpA(i, j) - eig_values[i] * operator()(i, j));
            else if constexpr (std::is_same<T, double>::value)
                normSquared += (OpA(i, j) - eig_values[i] * operator()(i, j)) * (OpA(i, j) - eig_values[i] * operator()(i, j));
            else
                throw std::runtime_error("Error: innerprod_complex not defined for this type");
        }

        eig_residuals[i] = std::sqrt(std::abs(normSquared));
    }
}

template <typename T>
void RC_device_matrix<T>::substract(const RC_device_matrix<T> &A){
    if (this->m != A.m || this->n != A.n)
    {
        throw std::runtime_error("Error: matrix sizes do not match");
    }

    for (size_t i = 0; i < this->m; i++)
    {
        for (size_t j = 0; j < this->n; j++)
        {
            operator()(i, j) -= A(i, j);
        }
    }

}

template <typename T>
void RC_device_matrix<T>::scale(const T alpha)
{
    for (size_t i = 0; i < this->m; i++)
    {
        for (size_t j = 0; j < this->n; j++)
        {
            operator()(i, j) *= alpha;
        }
    }

}


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
