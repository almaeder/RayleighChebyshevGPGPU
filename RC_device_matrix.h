
#pragma once
#include "RC_Vector.h"
#include "RC_host_matrix.h"
#include "cudaErrchk.h"
#include "cuda_kernels.h"

#include <cusolverDn.h>
#include <cublas_v2.h>

#ifndef RC_DEVICE_MATRIX_
#define RC_DEVICE_MATRIX_

template <typename T>
class RC_device_matrix : public RC_host_matrix<T>
{

public:
	//////////////////////////////////////////////////////////
	//  Required constructors
	//////////////////////////////////////////////////////////

	RC_device_matrix();

	RC_device_matrix(const RC_device_matrix<T> &W);

	//////////////////////////////////////////////////////////
	// Constructor not required for RayleighChebyshve
	// but useful for creating test program
	//////////////////////////////////////////////////////////

	RC_device_matrix(RC_INT m, RC_INT n);

	RC_device_matrix(RC_INT m, RC_INT n, std::vector<T> data);

	~RC_device_matrix();

	//////////////////////////////////////////////////////////////////
	//  Member functions required to use this class as a
	//  RayleighChebyshev template parameter
	//////////////////////////////////////////////////////////////////

	void host_to_device_copy();

	void device_to_host_copy();

	void device_to_device_copy(const RC_device_matrix<T>& W);

	void initialize(const RC_device_matrix<T> &W);

	void initialize(const RCvector<T> &V, RC_INT n);

	void orthogonalize();

	RC_device_matrix<T> &operator=(const RC_device_matrix<T> &W);

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

	void matmult(const RC_device_matrix<T>& A, const RC_device_matrix<T>& B, T alpha, T beta);

	void matmult(const RC_device_matrix<T>& A, const RC_device_matrix<T>& B, T alpha, T beta, std::string conj_A, std::string conj_B);

	void residuals(const RC_device_matrix<T> &OpA, const std::vector<double> &eig_values, std::vector<double> &eig_residuals, RC_INT residualCheckCount);

	void substract(const RC_device_matrix<T> &A);

	template <typename T1>
	void scale(const T1 alpha);

	void _create_cuda_handles();
	void _create_cuda_memory();
	void _create_cuda_memory(const RC_device_matrix<T>& W);
	void _create_cuda_descriptor();
	void _allocate_qr_memory();

	std::vector<T> mData;
	T *mData_d = NULL;
	cusparseDnMatDescr_t matrix_desc = NULL;
	RC_INT _old_residualCheckCount = -1;
	double *_eig_residuals_d = NULL;
	double *_eig_values_d = NULL;

    int geqrf_lwork = -1;
    T *geqrf_work_d = NULL;
    int geqrf_m, geqrf_n;
    T *tau_d = NULL;
    int tau_n = -1;
    int gqr_lwork = -1;
    T *gqr_work_d = NULL;
    int *info_d = NULL;

	cublasHandle_t cublas_handle = NULL;
	cusolverDnHandle_t cusolverDn_handle = NULL;

};

#endif /* RC_DEVICE_MATRIX_ */

#ifndef RC_DEVICE_RANDOMIZE_
#define RC_DEVICE_RANDOMIZE_

template <typename T>
class RC_device_randomize : public RC_abstract_randomize<T, RC_device_matrix<T>>
{
public:
	RC_device_randomize();

	void randomize(RCvector<T> &V);

	void randomize(RC_device_matrix<T> &M);

	int seed;
	std::mt19937_64 randomGenerator;
	std::uniform_real_distribution<double> distribution;
};

#endif /* RC_DEVICE_RANDOMIZE_ */
