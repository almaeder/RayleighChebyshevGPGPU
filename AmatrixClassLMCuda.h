
#pragma once
#include <vector>
#include <cmath>
#include "AvectorClassLMCuda.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#ifndef AMATRIXCLASSLMCuda_
#define AMATRIXCLASSLMCuda_

template <typename T>
class AmatrixClassLMCuda
{
	public:

//////////////////////////////////////////////////////////
//  Required constructors
//////////////////////////////////////////////////////////

	AmatrixClassLMCuda()
	{
	}

    AmatrixClassLMCuda(const AmatrixClassLMCuda<T>& W)
    {
		resize(W.m, W.n);
        initialize(W);
    }

//////////////////////////////////////////////////////////
// Constructor not required for RayleighChebyshve
// but useful for creating test program
//////////////////////////////////////////////////////////

    AmatrixClassLMCuda(long m, long n)
	{
		resize(m,n);
	}

	~AmatrixClassLMCuda(){
		if (matrix_desc != NULL && mData_d != NULL){
			cusparseDestroyDnMat(matrix_desc);
			cudaFree(mData_d);
		}
		else if (matrix_desc == NULL && mData_d == NULL){
			return;
		}
		else{
			std::cout << "Error: mData_d is NULL" << std::endl;
		}
	}

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////

	void create_cuda_memory(){
		if (mData_d != NULL){
			cudaFree(mData_d);
		}
		if(m > 0 && n > 0){
			cudaMalloc((void**)&mData_d, m*n*sizeof(T));
		}
	}

	void create_cuda_memory(const AmatrixClassLMCuda<T>& W){
		if (W.m != m || W.n != n){
			throw std::runtime_error("Error: Dimensions of W do not match dimensions of this matrix");
		}

		if (mData_d != NULL){
			cudaFree(mData_d);
		}
		if(m > 0 && n > 0){
			cudaMalloc((void**)&mData_d, m*n*sizeof(T));
			cudaMemcpy(mData_d, W.mData_d, m*n*sizeof(T), cudaMemcpyDeviceToDevice);
		}
	}

	void create_cuda_descriptor(){
		if (matrix_desc != NULL && mData_d != NULL){
			cusparseDestroyDnMat(matrix_desc);
			if constexpr (std::is_same<T, double>::value){
				cusparseCreateDnMat(&matrix_desc, m, n, m, mData_d, CUDA_R_64F, CUSPARSE_ORDER_COL);
			}
			else if constexpr (std::is_same<T, std::complex<double>>::value){
				cusparseCreateDnMat(&matrix_desc, m, n, m, mData_d, CUDA_C_64F, CUSPARSE_ORDER_COL);
			}
		}
		else if (mData_d != NULL){
			if constexpr (std::is_same<T, double>::value){
				cusparseCreateDnMat(&matrix_desc, m, n, m, mData_d, CUDA_R_64F, CUSPARSE_ORDER_COL);
			}
			else if constexpr (std::is_same<T, std::complex<double>>::value){
				cusparseCreateDnMat(&matrix_desc, m, n, m, mData_d, CUDA_C_64F, CUSPARSE_ORDER_COL);
			}
		}
		else if (matrix_desc == NULL && mData_d == NULL){
			return;
		}
		else{
			// raise exception
			throw std::runtime_error("Error: mData_d is NULL");
		}
	}

	void host_to_device(){
		if (mData_d != NULL){
			cudaMemcpy(mData_d, mData.data(), m*n*sizeof(T), cudaMemcpyHostToDevice);
		}
		else{
			// raise exception
			throw std::runtime_error("Error: mData_d is NULL");
		}
	}

	void device_to_host(){
		if (mData_d != NULL){
			cudaMemcpy(mData.data(), mData_d, m*n*sizeof(T), cudaMemcpyDeviceToHost);
		}
		else{
			// raise exception
			throw std::runtime_error("Error: mData_d is NULL");
		}
	}


	void initialize(const AmatrixClassLMCuda<T>& W)
	{
#ifdef _OPENMP
		#pragma omp parallel for collapse(2)
#endif
		for(size_t i = 0; i < W.n; i++)
		{
			for(size_t j = 0; j < W.m; j++)
			{
				mData[i*m + j] = W.mData[i*W.m + j];
			}
		}
	}

	void initialize(const AvectorClassLMCuda<T>& V, long n)
	{
#ifdef _OPENMP
		#pragma omp parallel for collapse(2)
#endif
		for(size_t i = 0; i < n; i++)
		{
			for(size_t j = 0; j < m; j++)
			{
				mData[i*m + j] = V.vData[j];
			}
		}

	}

    AmatrixClassLMCuda<T>& operator=(const AmatrixClassLMCuda<T>& W) {
        if (this == &W) {
            return *this;  // Check for self-assignment
        }

		resize(W.m, W.n);
        initialize(W);

        return *this;  // Enable chaining
    }


	void normalize()
	{	
#ifdef _OPENMP
       #pragma omp parallel for
#endif
		for(int k = 0; k < n; k++)
		{
			T normSquared = T(0.0);
			for(size_t j = 0; j < m; j++)
			{
				normSquared += mData[k*m + j]*mData[k*m + j];
			}

			normSquared = std::sqrt(std::abs(normSquared));

			for(size_t j = 0; j < m; j++)
			{
				mData[k*m + j] /= normSquared;
			}
		}
	}

	template <typename T1>
	T1 innerprod(const long k, const long l) const
	{
		if constexpr (std::is_same<T1, std::complex<double>>::value)
		{
			return innerprod_complex(k,l);
		}
		else if constexpr (std::is_same<T1, double>::value)
		{
			return innerprod_real(k,l);
		}
		else
		{
			throw std::runtime_error("Error: innerprod_complex not defined for this type");
		}
	}


	std::complex<double> innerprod_complex(const long k, const long l) const
	{
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
			normSquared += mData[k*m + j]*std::conj(mData[l*m + j]);
		}

		return normSquared;

	}

	double innerprod_real(const long k, const long l) const
	{

		double normSquared = double(0.0);
#ifdef _OPENMP
		#pragma omp parallel for reduction(+:normSquared)
#endif
		for(size_t j = 0; j < m; j++)
		{
			normSquared += mData[k*m + j]*mData[l*m + j];
		}

		return normSquared;

	}



	void resize(long n)
	{
		mData.resize(m*n);
		this->n = n;
		create_cuda_memory();
		create_cuda_descriptor();
	}

	void resize(long m, long n)
	{
		mData.resize(m*n);
		this->n = n;
		this->m = m;
		create_cuda_memory();
		create_cuda_descriptor();
	}


	void resize(long n, T value)
	{
		if(n == this->n)
		{
			for(size_t i = 0; i < mData.size(); i++)
			{
				mData[i] = value;
			}
		}
		else{
			mData.resize(m*n, value);
			this->n = n;
			create_cuda_memory();
			create_cuda_descriptor();
		}
	}
	

	void resize(long n, AvectorClassLMCuda<T>& V)
	{
		if(n == this->n)
		{
			#pragma omp parallel for collapse(2)
			for(size_t i = 0; i < n; i++)
			{
				for(size_t j = 0; j < m; j++)
				{
					mData[i*m + j] = V[j];
				}
			}
		}
		else{
			mData.resize(m*n);
			this->n = n;
			#pragma omp parallel for collapse(2)
			for(size_t i = 0; i < n; i++)
			{
				for(size_t j = 0; j < m; j++)
				{
					mData[i*m + j] = V[j];
				}
			}

			create_cuda_memory();
			create_cuda_descriptor();
		}
	}


	size_t getDimension() const
	{
	return m*n;
	}

    long getRowSize() const
    {
    return m;
    }

    long getColSize() const
    {
    return n;
    }

    inline T& operator()(long i, long j)
    {
    return mData[i  + j*m];
    };

    const inline T& operator()(long i, long j) const
    {
    return mData[i + j*m];
    };

    T* getDataPointer(){return mData.data();};

    const T* getDataPointer() const {return mData.data();};

    T* getDataPointer_d(){return mData_d;};

    const T* getDataPointer_d() const {return mData_d;};


	std::vector<T> mData;
	T *mData_d = NULL;
	cusparseDnMatDescr_t matrix_desc = NULL;

	long n;
	long m;
	size_t data_type_size = sizeof(T);
};



#endif /* AmatrixClassLMCuda_ */
