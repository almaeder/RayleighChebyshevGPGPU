//
//              LanczosCpolyOperatorLMCuda.h
//
//     LanczosCpolyOperator for Large Matrix problems
//
// This version of LanczosCpolyOperator is a multi-threaded implementation
// that uses only one copy of the operator whose eigensystem is being computed,
// and hence facilitates the construction of the eigensystem of very high dimensional
// linear operators. In order to exploit multi-threading of the application of the
// polynomial in the operator as implemented in this class, the operator
// class's apply(std::vector<Vtype>& V) implementation must internally utilize
// multi-threading, i.e. the apply(...) is not nested within a parallellized loop.
//
// Instances of this class are scaled Lanczos "C" polynomial operators
// The apply(...) member function of this class applies the Lanczos C
// in the associated operator to a vector.
//
// This is a templated class with respect to both vector and operator
// types.
//
// The apply routine is not multi-threaded, as this is typically
// done externally e.g. the application of the operator
// to a block of vectors is carried out by multi-threading the
// loop over each vector with each thread being associated with
// a separate instance of a LanczosCpolyOperatorLMCuda.
//
// Chris Anderson 2022
//
// Updated : Refactored code by removing duplicate functionality
// now provided by the internal instance of LanczosCpoly, and
// updated documentation.
/*
   The minimal functionality required of the classes
   that are used in this template are

   Vtype
   ---------
   A vector class with the following member functions:

   Vtype()                            (null constructor)
   Vtype(const Vtype&)                (copy constructor)

   initialize()                       (null initializer)
   initialize(const Vtype&)           (copy initializer)

   operator =                         (duplicate assignemnt)
   operator +=                        (incremental addition)
   operator -=                        (incremental subtraction)
   operator *=(double alpha)          (scalar multiplication)

   if VBLAS_ is defined, then the Vtype class must also possess member functions

   void   scal(double alpha)                                (scalar multiplication)
   void   axpy(double alpha,const Vtype& x)                 (this = this + alpah*x)
   void   axpby(double alpha,const Vtype& x, double beta)   (this = alpha*x + beta*this)

   If OpenMP is defined, then the vector class should NOT SET any class or static
   variables of the vector class arguments to copy, dot, or axpy. Also,
   no class variables or static variables should be set by nrm2().


   ############################################################################

   Otype
   ----------

   An operator class with the following member function:

   void apply(std::vector<Vtype>& Varray)

   which applies the operator to all vectors in the argument Varray and returns the result in Varray.

   To take advantage of multi-threading this apply operator should multi-thread
   internally, i.e. use a multi-threaded loop to apply the operator to each of the
   individual vectors.

   ############################################################################
*/
/*
#############################################################################
#
# Copyright 2009-2023 Chris Anderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# For a copy of the GNU General Public License see
# <http://www.gnu.org/licenses/>.
#
#############################################################################
*/
#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include "LanczosCpoly.h"
#include "cuda_kernels.h"
#include "cudaErrchk.h"

#ifndef LANCZOS_C_POLY_OPERATOR_LM_
#define LANCZOS_C_POLY_OPERATOR_LM_

template <class Amatrix, class Otype>
class LanczosCpolyOperatorLMCuda
{
public:

LanczosCpolyOperatorLMCuda()
{
      lanczosCpoly.initialize();
      Op                  = 0;
};

LanczosCpolyOperatorLMCuda(long polyDegree, long repetitionFactor,
double  lambdaMax,  double shift, Otype& Op)
{
    lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);

    this->Op                  = &Op;
}


// The following two member functions are used together to
// associate an operator with a LanczosCpolyOperator class
// instance separately from the Lanczos C polynomial
// parameters.
//
void initialize(Otype& Op)
{
      lanczosCpoly.initialize();
      this->Op = &Op;
};

void setLanczosCpolyParameters(long polyDegree, long repetitionFactor,
double  lambdaMax, double shift)
{
	if(this->Op == nullptr)
	{
		std::string errMsg = "\n     LanczosCpolyOperatorLMCuda : setting Lanczos C poly parameters before \n";
	                errMsg += "\n    associating an operator with the instance. \n";
		throw std::runtime_error(errMsg);
	}

	lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);
}

void initialize(long polyDegree, long repetitionFactor, 
double  lambdaMax, double shift, Otype& Op)
{
    lanczosCpoly.initialize(polyDegree, repetitionFactor, lambdaMax, shift);

    this->Op                  = &Op;
}

~LanczosCpolyOperatorLMCuda(void)
{};

void setShift(double shift)
{
    lanczosCpoly.setShift(shift);
}

void setPolyDegree(long polyDegree)
{
	lanczosCpoly.setPolyDegree(polyDegree);
}

void setRepetitionFactor(long repetitionFactor)
{
    lanczosCpoly.setRepetitionFactor(repetitionFactor);
}


//
// If 
// lambdaMax      = maximal  eigenvalue of A
// sigma          = applied shift with the constraint that sigma + lambda >= 0 for all lambda
// A'             = (A + sigma)/[(lambdaMax + sigma)/UpperXstar]
//
// The apply(Amatrix& v) operator of this class applies the operator
// Pm(A') to the input vector v. 
//


void apply(Amatrix& mArray)
{
    long repCount;
    long k;

    double UpperXStar       = lanczosCpoly.UpperXStar;
    double shift            = lanczosCpoly.shift;
    double lambdaMax        = lanczosCpoly.lambdaMax;
    long   polyDegree       = lanczosCpoly.polyDegree;
    long   repetitionFactor = lanczosCpoly.repetitionFactor;


    double starFactor = UpperXStar; // 1.0 - XStar;
    double rhoB       = lambdaMax/starFactor + shift/starFactor;
    double gamma1     = -2.0/(rhoB - 2.0*shift);
    double gamma2     =  2.0 - (4.0*shift)/rhoB;

    long vSize = (long)mArray.n;
    vn.resize(mArray.m, mArray.n);
    vnm1.resize(mArray.m, mArray.n);
    vnm2.resize(mArray.m, mArray.n);

    cudaErrchk(cudaMemcpy(vn.mData_d, mArray.mData_d, vnm2.getDimension()*mArray.data_type_size, cudaMemcpyDeviceToDevice));
    cudaErrchk(cudaMemcpy(vnm1.mData_d, mArray.mData_d, vnm2.getDimension()*mArray.data_type_size, cudaMemcpyDeviceToDevice));
    cudaErrchk(cudaMemcpy(vnm2.mData_d, mArray.mData_d, vnm2.getDimension()*mArray.data_type_size, cudaMemcpyDeviceToDevice));

    vnArrayPtr   = &vn;
    vnm1ArrayPtr = &vnm1;
    vnm2ArrayPtr = &vnm2;


    // An = gamma2 ( gamma1 X @ An-1 + An-1) - An-2
    for(repCount = 1; repCount <= repetitionFactor; repCount++)
    {
        // 
        // initialization of recurrance
        //
        if(repCount != 1)
        {
            cudaErrchk(cudaMemcpy((*vnm2ArrayPtr).mData_d, (*vnm1ArrayPtr).mData_d, vnm2.getDimension()*mArray.data_type_size, cudaMemcpyDeviceToDevice));
        }

        Op->apply(*vnm2ArrayPtr, *vnm1ArrayPtr, gamma2*gamma1, gamma2);

        // 
        // general recurrance
        //
        for(k = 2; k <= polyDegree; k++)
        {
            cudaErrchk(cudaMemcpy((*vnArrayPtr).mData_d, (*vnm1ArrayPtr).mData_d, vnm2.getDimension()*mArray.data_type_size, cudaMemcpyDeviceToDevice));

            Op->apply(*vnm1ArrayPtr, *vnArrayPtr, gamma2*gamma1, gamma2);

            cuda_kernels::substract((*vnArrayPtr).mData_d, (*vnm2ArrayPtr).mData_d, vnm2.getDimension());

            // 
            // swap pointers to implicitly shift the 
            // indices of the iteration vectors
            //
            vTmpArrayPtr = vnm2ArrayPtr;
            vnm2ArrayPtr = vnm1ArrayPtr;
            vnm1ArrayPtr = vnArrayPtr;
            vnArrayPtr   = vTmpArrayPtr;
        }

        cuda_kernels::scale((*vnm1ArrayPtr).mData_d, vnm2.getDimension(), 1.0/double(polyDegree+1));

     }

    cudaErrchk(cudaMemcpy(mArray.mData_d, (*vnm1ArrayPtr).mData_d, vnm2.getDimension()*mArray.data_type_size, cudaMemcpyDeviceToDevice));

}

    LanczosCpoly lanczosCpoly;

    long   polyDegree;
    long   repetitionFactor;
    double lambdaMax;
    double shift;
    double XStar;
    double UpperXStar;

    Otype* Op;

   Amatrix vn;
   Amatrix vnm1;
   Amatrix vnm2;


   Amatrix* vnArrayPtr;
   Amatrix* vnm1ArrayPtr;
   Amatrix* vnm2ArrayPtr;
   Amatrix* vTmpArrayPtr;
};

#endif

 
