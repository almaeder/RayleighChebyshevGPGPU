#include "diag_device_operator.h"

template <typename T, typename Matrix>
diag_device_operator<T, Matrix>::diag_device_operator(){
}


// TODO change behavior that the inputs are preallocated to lazy evaluation
// makes it easier to use and subspace size is not needed as input
template <typename T, typename Matrix>
diag_device_operator<T, Matrix>::diag_device_operator(RC_INT n, RC_INT subspace_size) {

    this->subspace_size = subspace_size;
    size = n;
    nnz = n;

    indptr = new int[size + 1];
    indices = new int[nnz];
    data = new T[nnz];

    // copy the data
    for (int i = 0; i < size + 1; i++) {
        // Base one, TODO change
        indptr[i] = i + 1;
    }
    for (int i = 0; i < nnz; i++) {
        // Base one, TODO change
        indices[i] = i + 1;
        if constexpr (std::is_same<T, CPX>::value) {
            data[i] = CPX(i + 1, 0);
        } else if constexpr (std::is_same<T, double>::value) {
            data[i] = static_cast<T>(i + 1);
        } 
        else
            throw std::runtime_error("Unsupported type");
    }

    #ifdef USE_MKL
    matdescra = new char[6];
    matdescra[0] = 'G';
    matdescra[1] = 'L';
    matdescra[2] = 'N';
    matdescra[3] = 'F';
	#else
	#endif


    cudaMalloc(&in_vector_d, size * sizeof(T));
    cudaMalloc(&out_vector_d, size * sizeof(T));
    cudaMalloc(&in_vectors_d, subspace_size*size * sizeof(T));
    cudaMalloc(&out_vectors_d, subspace_size*size * sizeof(T));

    cudaMallocHost(&in_vectors_h, subspace_size*size * sizeof(T));
    cudaMallocHost(&out_vectors_h, subspace_size*size * sizeof(T));

    cudaMalloc(&indptr_d, (size + 1) * sizeof(int));
    cudaMalloc(&indices_d, nnz * sizeof(int));
    cudaMalloc(&data_d, nnz * sizeof(T));

    cudaMemcpy(indptr_d, indptr, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, data, nnz * sizeof(T), cudaMemcpyHostToDevice);

    cusparseCreateCsr(
        &spMatDescr,
        size,
        size,
        nnz,
        indptr_d,
        indices_d,
        data_d,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ONE,
        CUDA_C_64F
    );

    cusparseCreate(&handle);

    cusparseCreateDnVec(&in_vector_desc, size, in_vector_d, CUDA_C_64F);
    cusparseCreateDnVec(&out_vector_desc, size, out_vector_d, CUDA_C_64F);
    cusparseCreateDnMat(&in_vectors_desc, size, subspace_size, size, in_vectors_d, CUDA_C_64F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&out_vectors_desc, size, subspace_size, size, out_vectors_d, CUDA_C_64F, CUSPARSE_ORDER_COL);

    T alpha = CPX(1.0, 0.0);
    T beta = CPX(0.0, 0.0);
    cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        spMatDescr,
        in_vector_desc,
        &beta,
        out_vector_desc,
        CUDA_C_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize_spmv
    );

    cudaMalloc(&buffer_spmv_d, bufferSize_spmv);

    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        spMatDescr,
        in_vectors_desc,
        &beta,
        out_vectors_desc,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize_spmm
    );

    cudaMalloc(&buffer_spmm_d, bufferSize_spmm);

}

template <typename T, typename Matrix>
diag_device_operator<T, Matrix>::~diag_device_operator()
{
    if (indptr != NULL) {
        delete[] indptr;
    }
    if (indices != NULL) {
        delete[] indices;
    }
    if (data != NULL) {
        delete[] data;
    }
    #ifdef USE_MKL
    if (matdescra != NULL) {
        delete[] matdescra;
    }
    #else
    #endif

    if (in_vector_d != NULL) {
        cudaFree(in_vector_d);
    }
    if (out_vector_d != NULL) {
        cudaFree(out_vector_d);
    }
    if (in_vectors_d != NULL) {
        cudaFree(in_vectors_d);
    }
    if (out_vectors_d != NULL) {
        cudaFree(out_vectors_d);
    }
    if (in_vectors_h != NULL) {
        cudaFreeHost(in_vectors_h);
    }
    if (out_vectors_h != NULL) {
        cudaFreeHost(out_vectors_h);
    }
    if (indptr_d != NULL) {
        cudaFree(indptr_d);
    }
    if (indices_d != NULL) {
        cudaFree(indices_d);
    }
    if (data_d != NULL) {
        cudaFree(data_d);
    }
    if (spMatDescr != NULL) {
        cusparseDestroySpMat(spMatDescr);
    }
    if (handle != NULL) {
        cusparseDestroy(handle);
    }
    if (in_vector_desc != NULL) {
        cusparseDestroyDnVec(in_vector_desc);
    }
    if (out_vector_desc != NULL) {
        cusparseDestroyDnVec(out_vector_desc);
    }
    if (in_vectors_desc != NULL) {
        cusparseDestroyDnMat(in_vectors_desc);
    }
    if (out_vectors_desc != NULL) {
        cusparseDestroyDnMat(out_vectors_desc);
    }
    if (buffer_spmv_d != NULL) {
        cudaFree(buffer_spmv_d);
    }
    if (buffer_spmm_d != NULL) {
        cudaFree(buffer_spmm_d);
    }


}

template <typename T, typename Matrix>
void diag_device_operator<T, Matrix>::apply(RCvector<T>& V)
{

    char transa = 'N';

    if constexpr (std::is_same<T, double>::value) {

        double alpha = 1.0;
        double beta = 0.0;

        #ifdef USE_MKL
    	mkl_dcsrmv(
    		&transa,
    		&size,
    		&size,
    		&alpha,
    		matdescra,
    		data,
    		indices,
    		indptr,
    		indptr + 1,
    		V.vData.data(),
    		&beta,
    		out_vectors_h);
        #else
        #endif
    }
    else if constexpr (std::is_same<T, CPX>::value) {

        T alpha = CPX(1.0, 0.0);
        T beta = CPX(0.0, 0.0);
    
        #ifdef USE_MKL
    	mkl_zcsrmv(
    		&transa,
    		&size,
    		&size,
    		(MKL_Complex16*)&alpha,
    		matdescra,
    		(MKL_Complex16*)data,
    		indices,
    		indptr,
    		indptr + 1,
    		(MKL_Complex16*)V.vData.data(),
    		(MKL_Complex16*)&beta,
    		(MKL_Complex16*)out_vectors_h);
        #else
        #endif
    }


    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        V.vData[i] = out_vectors_h[i];
    }

}

template <typename T, typename Matrix>
void diag_device_operator<T, Matrix>::apply(Matrix& M)
{
    cudaErrchk(cudaMemcpy(in_vectors_d, M.mData.data(), M.get_size() * sizeof(T), cudaMemcpyHostToDevice));

    T alpha = CPX(1.0, 0.0);
    T beta = CPX(0.0, 0.0);

    cusparseErrchk(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        spMatDescr,
        in_vectors_desc,
        &beta,
        out_vectors_desc,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        buffer_spmm_d
    ));

    cudaErrchk(cudaMemcpy(M.mData.data(), out_vectors_d, M.get_size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename Matrix>
void diag_device_operator<T, Matrix>::apply(Matrix& Min, Matrix& Mout, T alpha, T beta)
{
    cusparseErrchk(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        spMatDescr,
        Min.matrix_desc,
        &beta,
        Mout.matrix_desc,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        buffer_spmm_d
    ));
}

template class diag_device_operator<CPX, RC_device_matrix<CPX>>;
