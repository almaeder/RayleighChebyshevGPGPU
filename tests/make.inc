GTEST = -L/home/almaeder/Documents/QTSolver-dev/modules/gtest/build/lib64
INCGTEST = -I/home/almaeder/Documents/QTSolver-dev/modules/gtest/build/include

INCLAPACKINTERFACE = -I/home/almaeder/Documents/QTSolver-dev/modules/LapackInterfaceGPGPU

INCLUDES := ${INCGTEST} ${INCLAPACKINTERFACE} -I${MKLROOT}/include

LIBS := ${GTEST} -L${MKLROOT}/lib

NVCC := $(CUDA_HOME)/bin/nvcc
CC := g++-11.2.0-af

INCLUDES += -I$(CUDA_HOME)/include

CUDA_INCLUDES := -I$(CUDA_HOME)/include
CUDA_LIBS := -L${CUDA_HOME}/lib64 -lcudart -lcusparse -lcublas -lcusolver

LIBS += $(CUDA_LIBS)

CXXFLAGS := -fopenmp -std=c++17 -O3 -funroll-loops 
CXXFLAGS += -DUSE_MKL -DMKL_Complex16="std::complex<double>"
CUDA_FLAGS := -std=c++17 --gpu-architecture=sm_60 -O3
