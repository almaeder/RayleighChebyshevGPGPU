
#pragma once
#include "cudaErrchk.h"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <mutex>


class cudaHandles {

    public:
        // Member variables
        cublasHandle_t cublas_handle = NULL;
        cusolverDnHandle_t cusolverDn_handle = NULL;
        cusparseHandle_t cusparse_handle = NULL;

    private:
    
        // Static pointer to the cudaHandles instance
        static cudaHandles* instancePtr;
    
        // Mutex to ensure thread safety
        static std::mutex mtx;
    
        // Private Constructor
        cudaHandles() {
            // Initialize the handles
            cublasErrchk(cublasCreate(&cublas_handle));
            cusolverErrchk(cusolverDnCreate(&cusolverDn_handle));
            cusparseErrchk(cusparseCreate(&cusparse_handle));

        }
    
        ~cudaHandles() {
            // Initialize the handles
            cublasErrchk(cublasDestroy(cublas_handle));
            cusolverErrchk(cusolverDnDestroy(cusolverDn_handle));
            cusparseErrchk(cusparseDestroy(cusparse_handle));

        }


    public:
        // Deleting the copy constructor to prevent copies
        cudaHandles(const cudaHandles& obj) = delete;
    
        // Static method to get the cudaHandles instance
        static cudaHandles* getInstance() {
            if (instancePtr == nullptr) {
                std::lock_guard<std::mutex> lock(mtx);
                if (instancePtr == nullptr) {
                    instancePtr = new cudaHandles();
                }
            }
            return instancePtr;
        }

    };
