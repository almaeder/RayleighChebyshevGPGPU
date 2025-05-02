#GTEST = -L/home/almaeder/Documents/QTSolver-dev/modules/gtest/build/lib64
#INCGTEST = -I/home/almaeder/Documents/QTSolver-dev/modules/gtest/build/include
#
#INCLUDES := ${INCGTEST}  -I${MKLROOT}/include
#
#LIBS := ${GTEST} -L${MKLROOT}/lib
#
#NVCC := $(CUDA_HOME)/bin/nvcc
#CC := g++-11.2.0-af -DUSE_MKL
#
#INCLUDES += -I$(CUDA_HOME)/include
#
#CUDA_INCLUDES := -I$(CUDA_HOME)/include
#CUDA_LIBS := -L${CUDA_HOME}/lib64 -lcudart -lcusparse -lcublas -lcusolver
#
#LIBS += $(CUDA_LIBS)
#
#CXXFLAGS := -fopenmp -std=c++17 -O2 -funroll-loops 
#CXXFLAGS += -DUSE_MKL -DMKL_Complex16="std::complex<double>"
#CXXFLAGS += -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
#CUDA_FLAGS := -std=c++17 --gpu-architecture=sm_60 -O2



BLAS_HOME = /usr/local/nvidia_hpc_sdk/Linux_aarch64/25.3/compilers
SPBLAS_HOME = /user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-24.11-x4bklma5ij2g75dijbcrn673fmqzm3wv/Linux_aarch64/2024/compilers
GTEST_HOME = /home/almaeder/googletest/build

CUDA_HOME = /usr/local/nvidia_hpc_sdk/Linux_aarch64/25.3/cuda
CUDA_MATH_HOME = /usr/local/nvidia_hpc_sdk/Linux_aarch64/25.3/math_libs


INCLUDES :=  -I${GTEST_HOME}/include -I${MKLROOT}/include
LIBS := -L${GTEST_HOME}/lib -L${MKLROOT}/lib
INCLUDES += -I${BLAS_HOME}/include/lp64 -I${BLAS_HOME}/include
LIBS += -L${BLAS_HOME}/lib


NVCC := nvcc
CC := nvc++

CUDA_INCLUDES := -I$(CUDA_HOME)/include -I$(CUDA_MATH_HOME)/include
CUDA_LIBS := -L${CUDA_HOME}/lib64  -L$(CUDA_MATH_HOME)/lib64

CUDA_LIBS += -lcudart -lcusparse -lcublas -lcusolver -llapack -lblas

INCLUDES += $(CUDA_INCLUDES)
LIBS += $(CUDA_LIBS)

CXXFLAGS := -fopenmp -std=c++17 -O2 
CUDA_FLAGS := -std=c++17 --gpu-architecture=sm_60 -O2

# CXXFLAGS += -DUSE_MKL