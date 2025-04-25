

CUDA_HOME = /usr/local/nvidia_hpc_sdk/Linux_aarch64/25.3/cuda
CUDA_MATH_HOME = /usr/local/nvidia_hpc_sdk/Linux_aarch64/25.3/math_libs

BLAS_HOME = /usr/local/nvidia_hpc_sdk/Linux_aarch64/25.3/compilers


GTEST = -L/home/almaeder/googletest/build/lib64
INCGTEST = -I/home/almaeder/googletest/build/include

INCLAPACKINTERFACE = -I/home/almaeder/Documents/QTSolver-dev/modules/LapackInterfaceGPGPU

INCLUDES := ${INCGTEST} ${INCLAPACKINTERFACE} -I${MKLROOT}/include -I${BLAS_HOME}/include/ilp64 -I${BLAS_HOME}/include

LIBS := ${GTEST} -L${MKLROOT}/lib -L${BLAS_HOME}/lib

NVCC := nvcc
CC := gcc

INCLUDES += -I$(CUDA_HOME)/include -I$(CUDA_MATH_HOME)/include

CUDA_INCLUDES := -I$(CUDA_HOME)/include -I$(CUDA_MATH_HOME)/include
CUDA_LIBS := -L${CUDA_HOME}/lib64 -L$(CUDA_MATH_HOME)/lib64 -lcudart -lcusparse -lcublas -lcusolver

LIBS += $(CUDA_LIBS)

CXXFLAGS := -fopenmp -std=c++17 -O2
CXXFLAGS += -DUSE_MKL -DMKL_Complex16="std::complex<double>"
CXXFLAGS += -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
CUDA_FLAGS := -std=c++17 --gpu-architecture=sm_60 -O2
