#! /bin/tcsh

source /usr/pack/intel_compiler-2015.0.090-af/Linux-x86_64/bin/compilervars.csh intel64
source /usr/pack/intel_compiler-2015.0.090-af/Linux-x86_64/mkl/bin/mklvars.csh intel64

setenv ARPACK_DIR /usr/scratch/mont-fort4/almaeder/libraries/arpack-ng-v3.9.1/build

setenv CUDA_HOME /usr/local/cuda-12.4/
setenv GTEST_ROOT /home/almaeder/Documents/QTSolver-dev/modules/gtest/build
setenv LD_LIBRARY_PATH $ARPACK_DIR/lib:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH $CUDA_HOME/lib64:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH $GTEST_ROOT/lib64:$LD_LIBRARY_PATH


./build/test