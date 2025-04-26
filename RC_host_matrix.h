
#pragma once
#include "RC_Vector.h"
#include "RC_abstract_matrix.h"

extern "C" {
    #ifdef USE_MKL
    #include <mkl_cblas.h>
    #include <mkl_lapacke.h>
    #else
    #include <cblas.h>
    #include <lapacke.h>
    #endif
}

#ifndef RC_HOST_MATRIX_
#define RC_HOST_MATRIX_

template <typename T>
class RC_host_matrix : public RC_abstract_matrix<T, RC_host_matrix<T>>
{
public:
	//////////////////////////////////////////////////////////
	//  Required constructors
	//////////////////////////////////////////////////////////

	RC_host_matrix();

	RC_host_matrix(const RC_host_matrix<T> &W);

	//////////////////////////////////////////////////////////
	// Constructor not required for RayleighChebyshve
	// but useful for creating test program
	//////////////////////////////////////////////////////////

	RC_host_matrix(RC_INT m, RC_INT n);

	RC_host_matrix(RC_INT m, RC_INT n, std::vector<T> data);

	~RC_host_matrix();

	//////////////////////////////////////////////////////////////////
	//  Member functions required to use this class as a
	//  RayleighChebyshev template parameter
	//////////////////////////////////////////////////////////////////

	void host_to_device_copy();

	void device_to_host_copy();

	void device_to_device_copy(const RC_host_matrix<T>& W);

	void initialize(const RC_host_matrix<T> &W) override;

	void initialize(const RCvector<T> &V, RC_INT n);

	void orthogonalize();

	RC_host_matrix<T> &operator=(const RC_host_matrix<T> &W) override;

    inline T& operator()(RC_INT i, RC_INT j)
	{
		return mData[i  + j*this->m];
	};

    const inline T& operator()(RC_INT i, RC_INT j) const
	{
	return mData[i + j*this->m];
	};

	void normalize();

	T inner_product(const RC_INT k, const RC_INT l) const;

	void _scale(const RC_INT k, const T alpha);
	void _scale_add(const RC_INT k, const RC_INT l, const T alpha);

	void resize_cols(RC_INT n);

	void resize_cols(RC_INT n, T value);

	void resize_cols(RC_INT n, RCvector<T> &V);

	void resize(RC_INT m, RC_INT n);

	void matmult(const RC_host_matrix<T>& A, const RC_host_matrix<T>& B, T alpha, T beta);

	void matmult(const RC_host_matrix<T>& A, const RC_host_matrix<T>& B, T alpha, T beta, std::string conj_A, std::string conj_B);

	void residuals(const RC_host_matrix<T> &OpA, const std::vector<double> &eig_values, std::vector<double> &eig_residuals, RC_INT residualCheckCount);

	void substract(const RC_host_matrix<T> &A);

	void scale(const T alpha);

	void _create_cuda_memory();
	void _create_cuda_memory(const RC_host_matrix<T>& W);
	void _create_cuda_descriptor();

	std::vector<T> mData;
	std::vector<T> tau;
};

#endif /* RC_HOST_MATRIX_ */

#ifndef RC_HOST_RANDOMIZE_
#define RC_HOST_RANDOMIZE_

template <typename T>
class RC_host_randomize : public RC_abstract_randomize<T, RC_host_matrix<T>>
{
public:
	RC_host_randomize();

	void randomize(RCvector<T> &V);

	void randomize(RC_host_matrix<T> &M);

	int seed;
	std::mt19937_64 randomGenerator;
	std::uniform_real_distribution<double> distribution;
};

#endif /* RC_HOST_RANDOMIZE_ */
