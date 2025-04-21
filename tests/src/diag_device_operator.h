#pragma once
#include "../../RC_device_matrix.h"
#include <iostream>
#include <fstream>
#include <string>
// include cuda runtime and cuda sparse
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <chrono>
#include "../../cudaErrchk.h"
#include <mkl_spblas.h>

using CPX = std::complex<double>;


#ifndef DIAG_DEVICE_OPERATOR
#define DIAG_DEVICE_OPERATOR

template <typename T, typename Matrix>
class diag_device_operator
{
    public:

    // matrix size
	int size;
    // number of non-zero elements
	int nnz;
    // size of the subspace
	int subspace_size;

    // cuda handles
	cusparseHandle_t handle = NULL;

    // host data and mkl descriptor
	int* indptr = NULL;
	int* indices = NULL;
	T* data = NULL;
	char *matdescra = NULL;

    // device data
	int* indptr_d = NULL;
	int* indices_d = NULL;
	T* data_d = NULL;
	cusparseSpMatDescr_t spMatDescr = NULL;

    // spmm io buffers
	T *in_vector_d = NULL;
	T *out_vector_d = NULL;
	T *in_vectors_d = NULL;
	T *out_vectors_d = NULL;
	T *in_vectors_h = NULL;
	T *out_vectors_h = NULL;
	cusparseDnVecDescr_t in_vector_desc = NULL;
	cusparseDnVecDescr_t out_vector_desc = NULL;
	cusparseDnMatDescr_t in_vectors_desc = NULL;
	cusparseDnMatDescr_t out_vectors_desc = NULL;

    // spmm buffers
	size_t bufferSize_spmv;
	size_t bufferSize_spmm;
	void *buffer_spmv_d;
	void *buffer_spmm_d;

	diag_device_operator();

	diag_device_operator(RC_INT n, RC_INT subspace_size);

	~diag_device_operator();

	void apply(RCvector<T>& V);

	void apply(Matrix& M);

    void apply(Matrix& Min, Matrix& Mout, T alpha, T beta);

};

#endif /* DIAG_DEVICE_OPERATOR */
