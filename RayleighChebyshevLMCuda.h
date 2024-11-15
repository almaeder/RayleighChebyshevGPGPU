/**
                         RayleighChebyshevLMCuda.h


   A templated class with member functions for computing eigenpairs
   corresponding to the lowest eigenvalues of a linear operator.

   This routine is a version of RayleighChebyshev designed for
   linear operator classes whose instances require a large memory
   allocation. To take advantage of multi-threading using this version
   multiple instances of the linear operator are not created; just a
   single instance. It is assumed that, if available, the apply(std::vector<Vtype>& Varray)
   member function takes advantage of multi-threading.

   The routine is designed for both real symmetric and
   complex Hermitian operators.

   The eigenvalues are returned in a std::vector<double> instance
   while the eigenvectors are internally allocated and returned in
   a std::vector<Vtype> class instance.

   To use with complex Hermitian operators the template parameter
   specification

   typename Dtype  =  std::complex<double>.

   must be used. For complex Hermitian operators it is assumed that
   the inner product of the vector type Vtype is the
   complex inner product.

   OpenMP multi-thread usage is enabled by defining _OPENMP

   Note: _OPENMP is automatically defined if -fopenmp is specified
   as part of the compilation command.

   The minimal functionality required of the classes
   that are used in this template are

   Vtype
   ---------
   A std::vector class with the following member functions:

   Vtype()                            (null constructor)
   Vtype(const Vtype&)                (copy constructor)

   initialize()                       (null initializer)
   initialize(const Vtype&)           (copy initializer)

   operator =                         (duplicate assignemnt)
   operator +=                        (incremental addition)
   operator -=                        (incremental subtraction)
   operator *=(double alpha)          (scalar multiplication)

   double dot(const Vtype&)           (dot product)
   RAY_INT getDimension() const          (returns dimension of the Vtype subspace)

   if VBLAS_ is defined, then the Vtype class must also possess member functions

   double nrm2()                                            (2-norm based on std::vector dot product)
   void   scal(double alpha)                                (scalar multiplication)
   void   axpy(double alpha,const Vtype& x)                 (this = this + alpha*x)
   void   axpby(double alpha,const Vtype& x, double beta)   (this = alpha*x + beta*this)

   If OpenMP is defined, then the std::vector class should NOT SET any class or static
   variables of the std::vector class arguments to copy, dot, or axpy. Also,
   no class variables or static variables should be set by nrm2().

   ############################################################################

   Otype
   ----------

   An operator class with the following member function:

   void apply(std::vector<Vtype>& Varray)

   which applies the operator to each element of Varray and returns the results in Varray.

   If OpenMP is used then Otype must have a copy constructor of the form

   Otype(const Otype& O);

   ############################################################################

   VRandomizeOpType
   ----------

   An opearator class with the following member function:

   void randomize(Vtype& V)

   which initializes the elements of the Vtype std::vector V to have random values.

   ############################################################################


    ###########################################################################

    !!!! Important restriction on the std::vector classes and operator classes
    used by the RayleighChebyshevLMCuda template.

    ###########################################################################

    When specifying a std::vector class to be used with a RayleighChebyshevLMCuda instance, it is critical
    that the copy constructor handle null instances correctly (e.g. instances that were created
    with the null constructor).

    Specifically, one cannot assume that the input argument to the copy constructor is a
    non-null instance and one has to guard against copying over data that doesn't exist in
    the copy constructor code.

    This coding restriction arises because of the use of stl::vectors of the specified
    std::vector class; when intializing it creates a null instance and apparently calls the
    copy constructor to create the duplicates required of the array, rather than
    calling the null constructor multiple times.

    The symptoms of not doing this are segmentation faults that occur when one
    tries to copy data associated with a null instance.


    When using multi-threaded execution it is also necessary that the operator classes
    specified in the template also have copy constructors that handle null
    input instances correctly. This restriction arises because of the use of
    stl::vectors to create operator instances for each thread.

---

   External dependencies: Default use of LAPACK from SCC::LapackInterface component

   LAPACK is necessary for complex Hermitian operators, for real symmetric
   operators one can remove dependency on LAPACK and the SCC::LapackInterface component
   by specifying the compiler define RC_WITHOUT_LAPACK_.
   Reference:

   Christopher R. Anderson, "A Rayleigh-Chebyshev procedure for finding
   the smallest eigenvalues and associated eigenvectors of large sparse
   Hermitian matrices" Journal of Computational Physics,
   Volume 229 Issue 19, September, 2010.


   Author Chris Anderson July 12, 2005
   Version : May 24, 2023
*/
/*
#############################################################################
#
# Copyright 2005-2023 Chris Anderson
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
#include <iostream>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>

#ifndef RC_WITHOUT_LAPACK_
#include "LapackInterface/SCC_LapackMatrix.h"
#include "LapackInterface/SCC_LapackMatrixRoutines.h"

#include "LapackInterface/SCC_LapackMatrixCmplx16.h"
#include "LapackInterface/SCC_LapackMatrixRoutinesCmplx16.h"
#endif

#include "RCarray2d.h"
#include "RC_Types.h"

#include "LanczosCpoly.h"               // Chebyshev polynomial based filter polynomial
#include "LanczosCpolyOperatorLMCuda.h" // Chebyshev polynomial based filter polynomial operator
#include "LanczosMaxMinFinder.h"

#include "JacobiDiagonalizer.h"

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "cudaErrchk.h"
#include <cublas_v2.h>
#include <complex>
#include <type_traits>

#ifndef RAYLEIGH_CHEBYSHEV_LM_
#define RAYLEIGH_CHEBYSHEV_LM_

#define DEFAULT_MAX_MIN_TOL 1.0e-06
#define JACOBI_TOL 1.0e-12
#define DEFAULT_MAX_INNER_LOOP_COUNT 10000
#define DEFAULT_POLY_DEGREE_MAX 200
#define DEFAULT_FILTER_REPETITION_COUNT 1

#ifndef RC_WITHOUT_LAPACK_
#define DEFAULT_USE_JACOBI_FLAG false
#else
#define DEFAULT_USE_JACOBI_FLAG true
#endif

#define RAYLEIGH_CHEBYSHEV_SMALL_TOL_ 1.0e-11

#ifdef TIMING_
#include "ClockIt.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

template <class Atype, class Vtype, class Otype, class VRandomizeOpType, typename Dtype>
class RayleighChebyshevLMCuda
{
public:
    RayleighChebyshevLMCuda()
    {
        initialize();
    }

    ~RayleighChebyshevLMCuda()
    {
        if (geqrf_work_d != NULL)
        {
            cudaFree(geqrf_work_d);
        }
        if (gqr_work_d != NULL)
        {
            cudaFree(gqr_work_d);
        }
        if (tau_d != NULL)
        {
            cudaFree(tau_d);
        }
        if (info_d != NULL)
        {
            cudaFree(info_d);
        }

        if (cusolverDn_handle != NULL)
        {
            cusolverErrchk(cusolverDnDestroy(cusolverDn_handle));
        }
        if (cublas_handle != NULL)
        {
            cublasErrchk(cublasDestroy(cublas_handle));
        }

        if (VtAVeigValue_d != NULL)
        {
            cudaFree(VtAVeigValue_d);
        }
        if (eigVresiduals_d != NULL)
        {
            cudaFree(eigVresiduals_d);
        }
        if (VtAVeigValue_h != NULL)
        {
            delete[] VtAVeigValue_h;
        }
        if (eigVresiduals_h != NULL)
        {
            delete[] eigVresiduals_h;
        }
        res_residualCheckCount = -1;
        gqr_lwork = -1;
        geqrf_lwork = -1;
    }

    void initialize()
    {
        OpPtr = nullptr;
        verboseFlag = false;
        eigDiagnosticsFlag = false;
        verboseSubspaceFlag = false;
        jacobiMethod.tol = JACOBI_TOL;
        useJacobiFlag = DEFAULT_USE_JACOBI_FLAG;
        minIntervalPolyDegreeMax = DEFAULT_POLY_DEGREE_MAX;
        filterRepetitionCount = DEFAULT_FILTER_REPETITION_COUNT;
        minEigValueEst = 0.0;
        maxEigValueEst = 0.0;
        guardValue = 0.0;
        intervalStopConditionFlag = false;
        hardIntervalStopFlag = false;
        stopCondition = RC_Types::StopCondition::COMBINATION;
        maxMinTol = DEFAULT_MAX_MIN_TOL;

        nonRandomStartFlag = false;
        fixedIterationCount = false;
        maxInnerLoopCount = DEFAULT_MAX_INNER_LOOP_COUNT;

        eigVecResiduals.clear();

        finalData.clear();
        countData.clear();

        resultsStreamPtr = nullptr;

#ifdef TIMING_
        timeValue.clear();
#endif

#ifdef _OPENMP
        MtVarray.clear();
#endif
    }

    // This routine determines the factor used for estimating
    // relative errors.
    //
    //  || Rel Error || = || Err ||/|| val || when ||val|| > (default small tolerance)/tol
    //
    //  || Rel Error || = || Err ||/ ((default small tolerance)/tol)  otherwise
    //
    //

    double getRelErrorFactor(double val, double tol)
    {
        double relErrFactor = 1.0;

        if (std::abs(val) * tol > RAYLEIGH_CHEBYSHEV_SMALL_TOL_)
        {
            relErrFactor = std::abs(val);
        }
        else
        {
            relErrFactor = RAYLEIGH_CHEBYSHEV_SMALL_TOL_ / tol;
        }
        return relErrFactor;
    }

    void setStopCondition(std::string stopConditionStr)
    {
        std::transform(stopConditionStr.begin(), stopConditionStr.end(), stopConditionStr.begin(), [](unsigned char c)
                       { return static_cast<char>(std::toupper(c)); });

        if (stopConditionStr == "COMBINATION")
        {
            setStopCondition(RC_Types::StopCondition::COMBINATION);
        }
        else if (stopConditionStr == "EIGENVALUE_ONLY")
        {
            setStopCondition(RC_Types::StopCondition::EIGENVALUE_ONLY);
        }
        else if (stopConditionStr == "RESIDUAL_ONLY")
        {
            setStopCondition(RC_Types::StopCondition::RESIDUAL_ONLY);
        }
        else if (stopConditionStr == "DEFAULT")
        {
            setStopCondition(RC_Types::StopCondition::COMBINATION);
        }
        else
        {
            std::string errMsg = "\nRayleighChebyshevLM Error : Stopping condition type specified not";
            errMsg += "\none of DEFAULT, COMBINATION, EIGENVALUE_ONLY,or RESIDUAL_ONLY.";
            errMsg += "\nOffending specification : " + stopConditionStr + "\n";
            throw std::runtime_error(errMsg);
        }
    }

    void setStopCondition(RC_Types::StopCondition stopCondition)
    {
        switch (stopCondition)
        {
        case RC_Types::StopCondition::COMBINATION:
        {
            this->stopCondition = RC_Types::StopCondition::COMBINATION;
        }
        break;
        case RC_Types::StopCondition::EIGENVALUE_ONLY:
        {
            this->stopCondition = RC_Types::StopCondition::EIGENVALUE_ONLY;
        }
        break;

        case RC_Types::StopCondition::RESIDUAL_ONLY:
        {
            this->stopCondition = RC_Types::StopCondition::RESIDUAL_ONLY;
        }
        break;

        case RC_Types::StopCondition::DEFAULT:
        {
            this->stopCondition = RC_Types::StopCondition::COMBINATION;
        }
        break;

        default:
        {
            this->stopCondition = RC_Types::StopCondition::COMBINATION;
        }
        }
    }

    void setMaxMinTolerance(double val = DEFAULT_MAX_MIN_TOL)
    {
        maxMinTol = val;
    }

    void setMaxInnerLoopCount(RAY_INT val)
    {
        maxInnerLoopCount = val;
    }

    void resetMaxInnerLoopCount()
    {
        maxInnerLoopCount = DEFAULT_MAX_INNER_LOOP_COUNT;
    }

    void setFixedIterationCount(bool val)
    {
        fixedIterationCount = val;
    }

    void clearFixedIteratonCount()
    {
        fixedIterationCount = false;
    }

    void setNonRandomStartFlag(bool val = true)
    {
        nonRandomStartFlag = val;
    }

    void clearNonRandomStartFlag()
    {
        nonRandomStartFlag = false;
    }

    void setVerboseFlag(bool val = true)
    {
        verboseFlag = val;
    }

    void clearVerboseFlag()
    {
        verboseFlag = false;
    }

    void setVerboseSubspaceFlag(bool val = true)
    {
        verboseSubspaceFlag = val;
    }

    void clearVerboseSubspaceFlag()
    {
        verboseSubspaceFlag = false;
    }

    void setEigDiagnosticsFlag(bool val = true)
    {
        eigDiagnosticsFlag = val;
    }

    void clearEigDiagnosticsFlag()
    {
        eigDiagnosticsFlag = 0;
    }

    double getMinEigValueEst()
    {
        return minEigValueEst;
    }

    double getMaxEigValueEst()
    {
        return maxEigValueEst;
    }

    void setMinIntervalPolyDegreeMax(RAY_INT polyDegMax)
    {
        minIntervalPolyDegreeMax = polyDegMax;
    }

    void setFilterRepetitionCount(RAY_INT repetitionCount)
    {
        filterRepetitionCount = repetitionCount;
    }

    double getGuardEigenvalue()
    {
        return guardValue;
    }

    void setIntervalStopCondition(bool val = true)
    {
        intervalStopConditionFlag = val;
    }

    void clearIntervalStopCondition()
    {
        intervalStopConditionFlag = false;
    }

    void setHardIntervalStop(bool val = true)
    {
        hardIntervalStopFlag = val;
    }

    void clearHardIntervalStop()
    {
        hardIntervalStopFlag = false;
    }

    void setMaxSpectralBoundsIter(RAY_INT iterCount)
    {
        lanczosMaxMinFinder.setIterationMax(iterCount);
    }

    RAY_INT getSpectralBoundsIterCount()
    {
        return lanczosMaxMinFinder.getIterationCount();
    }

    void setUseJacobi(bool val)
    {
        useJacobiFlag = val;
    }

    void clearUseJacobi()
    {
        useJacobiFlag = false;
    }

    std::vector<double> getEigVectorResiduals() const
    {
        return eigVecResiduals;
    }

    std::map<std::string, double> getFinalData() const
    {
        return finalData;
    }

    std::map<std::string, RAY_INT> getCountData() const
    {
        return countData;
    }

    void setResultsStream(std::ostream &S)
    {
        resultsStreamPtr = &S;
    }

    void clearResultsStream()
    {
        resultsStreamPtr = nullptr;
    }

#ifdef TIMING_
    std::map<std::string, double> getTimingData() const
    {
        return timeValue;
    }
#endif

    //
    //  Member functions called to return eigensystem of operator projected onto
    //  working subspace.
    //
    void computeVtVeigensystem(Atype &VtAV, std::vector<double> &VtAVeigValue,
                               Atype &VtAVeigVector)
    {
        VtAV.device_to_host();

        RAY_INT rowSize = VtAV.getRowSize();

        if (useJacobiFlag)
        {
            // TODO: make work with complex numbers
            // jacobiMethod.setSortIncreasing(true);
            // jacobiMethod.setIOdataRowStorage(false);
            // jacobiMethod.getEigenSystem(VtAV.getDataPointer(), rowSize, &VtAVeigValue[0], VtAVeigVector.getDataPointer());
        }
        else
#ifndef RC_WITHOUT_LAPACK_
        {
            /////////////////////////////////////////////////////////////////////////////
            //     Calculation using LAPACK
            ////////////////////////////////////////////////////////////////////////////
            RAY_INT colSize = VtAV.getColSize();

            using MatrixType = typename std::conditional<std::is_same<Dtype, double>::value,
                                                         SCC::LapackMatrix,
                                                         SCC::LapackMatrixCmplx16>::type;

            MatrixType VtAVmatrix;
            MatrixType VtAVeigVectorMatrix;

            VtAVmatrix.initialize(rowSize, colSize);
            VtAVeigVectorMatrix.initialize(rowSize, colSize);

            for (RAY_INT i = 0; i < rowSize; i++)
            {
                for (RAY_INT j = 0; j < colSize; j++)
                {
                    VtAVmatrix(i, j) = VtAV(i, j);
                    VtAVeigVectorMatrix(i, j) = VtAVeigVector(i, j);
                }
            }

            VtAVeigValue.resize(colSize, 0.0);
            if constexpr (std::is_same<Dtype, double>::value)
            {
                dsyev.computeEigensystem(VtAVmatrix, VtAVeigValue, VtAVeigVectorMatrix);
            }
            else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
            {
                zhpevx.createEigensystem(VtAVmatrix, VtAVeigValue, VtAVeigVectorMatrix);
            }
            else
            {
                static_assert(std::is_same<Dtype, double>::value || std::is_same<Dtype, std::complex<double>>::value, "Unsupported Dtype");
            }

            for (RAY_INT i = 0; i < rowSize; i++)
            {
                for (RAY_INT j = 0; j < colSize; j++)
                {
                    VtAVeigVectorMatrix(i, j) = VtAVeigVector(i, j) = VtAVeigVectorMatrix(i, j);
                }
            }
        }
#else
        {
            std::string errMsg = "\nXXXX RayleighChebyshev Error XXXX";
            errMsg += "\nUse of Lapack solvers not supported without SCC::LapackInterface components\n";
            throw std::runtime_error(errMsg);
        }
#endif

        VtAVeigVector.host_to_device();
    }

    void getMinEigAndMaxEig(double iterationTol, Vtype &vStart, Otype &oP,
                            VRandomizeOpType &randOp, double &minEigValue,
                            double &maxEigValue)
    {
        //
        // Create temporaries based on vStart
        //
        Vtype w(vStart);
        Vtype wTmp(vStart);

        if (verboseFlag)
        {
            lanczosMaxMinFinder.setVerboseFlag();
        }

        // Specify accurate estimates of largest and smallest

        lanczosMaxMinFinder.setMinMaxEigStopCondition();

        lanczosMaxMinFinder.getMinMaxEigenvalues(iterationTol, vStart, w, wTmp, oP,
                                                 randOp, minEigValue, maxEigValue);

        char charBuf[256];
        if (verboseFlag)
        {
            snprintf(charBuf, 256, "\nEstMinimum_Eigenvalue : %15.10g  \nEstMaximum_Eigenvalue : %15.10g \n", minEigValue, maxEigValue);
            std::cout << charBuf << std::endl;
            if (resultsStreamPtr)
            {
                *resultsStreamPtr << charBuf << std::endl;
            }
        }

        minEigValueEst = minEigValue;
        maxEigValueEst = maxEigValue;
    }

    //
    // This routine obtains the spectral estimates required for the core RayleighChebyshev
    // routine, in particular, an accurate estimate of the largest eigenvalue and an
    // estimate of the smallest eigenvalue.
    //
    void getInitialRCspectralEstimates(double iterationTol, Vtype &vStart, Otype &oP,
                                       VRandomizeOpType &randOp, double &minEigValue, double &maxEigValue)
    {
        //
        // Create temporaries based on vStart
        //
        Vtype w(vStart);
        Vtype wTmp(vStart);

        if (verboseFlag)
        {
            lanczosMaxMinFinder.setVerboseFlag();
        }

        // Specify accurate estimates of largest. Here we use the
        // fact that the core procedure only requires a good upper
        // bound and a reasonably good lower bound.
        //
        // If this turns out to be problematic, then one can get
        // accurate estimates of both the lower and the upper
        // using getMinEigAndMaxEig(...) and then
        // invoking a version of the eigen system routine that
        //

        lanczosMaxMinFinder.setMaxEigStopCondition();

        lanczosMaxMinFinder.getMinMaxEigenvalues(iterationTol, vStart, w, wTmp, oP,
                                                 randOp, minEigValue, maxEigValue);

        if (verboseFlag)
        {
            printf("Minimum_Eigenvalue : %10.5g  \nMaximum_Eigenvalue : %10.5g \n", minEigValue,
                   maxEigValue);
        }

        minEigValueEst = minEigValue;
        maxEigValueEst = maxEigValue;
    }

    //
    //  Computes the lowest eigCount eigenvalues and eigenvectors.
    //
    //  Input:
    //
    //  eigCount    : The desired number of eigenvalues
    //  minEigValue : The minimum eigenvalue
    //  maxEigValue : The maximum eigenvalue
    //
    // e.g. minEigValue <= lambda <= maxEigValue for all eigenvalues lambda.
    //
    // subspaceTol  : Stopping tolerance.
    //
    // An eigenvalue lambda is considered converged when
    //
    // | lambda^(n) - lambda^(n-1) | < subspaceTol*(1 + |lambda^(n)|)
    //
    // subspaceIncrementSize and bufferSize : sizes used to determine the
    // dimension of the subspace used to determine eigenpars.
    //
    // The dimension of the subspace is subspaceIncrementSize + bufferSize,
    // and the lowest subspaceIncrementSize are evaluated for convergence.
    // The remaining subspace dimensions (of size bufferSize) are used to
    // increase the gap between the desired states and other states in
    // order to improve performance.
    //
    // vStart : A std::vector instance used as a template for for the
    //          the construction of all eigenvectors computed.
    //
    //    oP  : The linear operator whose eigenpairs are sought
    //
    // randOp : An operator that assigns random values to the elements
    //          of a std::vector. Used for initial guesses.
    //
    // Input/Output
    //
    // eigValues  : std::vector of doubles to capture the eigenvalues
    //
    // eigVectors : std::vector of vectors to capture the eigenvectors.
    //              If this std::vector is non-empty, then the non-null
    //              vectors are used as starting vectors for the
    //              subspace iteration.
    //
    //
    //
    // Restrictions on the spectral ranges estimates:
    //
    // maxEigValueBound > maxEigValue
    // minEigValueEst   > minEigValue
    //
    RAY_INT getMinEigenSystem(RAY_INT eigCount, double minEigValueEst, double maxEigValueBound,
                           double subspaceTol, RAY_INT subspaceIncrementSize, RAY_INT bufferSize,
                           Vtype &vStart, Otype &oP, VRandomizeOpType &randOp, std::vector<double> &eigValues,
                           Atype &eigVectors)
    {

        RAY_INT maxEigensystemDim = eigCount;
        double lambdaMax = maxEigValueBound;

        this->clearIntervalStopCondition();
        this->setHardIntervalStop();

        return getMinIntervalEigenSystem_Base(minEigValueEst, lambdaMax, maxEigValueBound,
                                              subspaceTol, subspaceIncrementSize, bufferSize, maxEigensystemDim,
                                              vStart, oP, randOp, eigValues, eigVectors);
    }
    //
    //  Computes the lowest eigCount eigenvalues and eigenvectors
    //
    RAY_INT getMinEigenSystem(RAY_INT eigCount, double subspaceTol, RAY_INT subspaceIncrementSize,
                           RAY_INT bufferSize, Vtype &vStart, Otype &oP, VRandomizeOpType &randOp,
                           std::vector<double> &eigValues, Atype &eigVectors)
    {

        double minFinderTol = maxMinTol;

        double minEigValue;
        double maxEigValue;

        getInitialRCspectralEstimates(minFinderTol, vStart, oP, randOp, minEigValue, maxEigValue);

        //
        // Increase maxEigValue slightly to be on the safe side if we don't have
        // the identity.
        //
        if (std::abs(maxEigValue - minEigValue) > 1.0e-12)
        {
            maxEigValue += 0.001 * std::abs(maxEigValue - minEigValue);
        }

        this->clearIntervalStopCondition();

        return getMinEigenSystem(eigCount, minEigValue, maxEigValue,
                                 subspaceTol, subspaceIncrementSize, bufferSize,
                                 vStart, oP, randOp, eigValues, eigVectors);
    }

    //
    // return value >= 0 returns the number of eigenpairs found
    // return value <  0 returns error code
    //

    RAY_INT getMinIntervalEigenSystem(double lambdaMax, double subspaceTol,
                                   RAY_INT subspaceIncrementSize, RAY_INT bufferSize, RAY_INT maxEigensystemDim,
                                   Vtype &vStart, Otype &oP, VRandomizeOpType &randOp, std::vector<double> &eigValues,
                                   Atype &eigVectors)
    {
        double minFinderTol = maxMinTol;

        double minEigValue;
        double maxEigValue;

        // Get accurate estimates of both the largest and smallest eigenvalues

        getMinEigAndMaxEig(minFinderTol, vStart, oP, randOp, minEigValue, maxEigValue);

        // Quick return if the upper bound is smaller than the smallest eigenvalue

        if (lambdaMax < minEigValue)
        {
            return 0;
            eigValues.clear();
        }

        //
        // Increase maxEigValue slightly to be on the safe side if we don't have
        // the identity.
        //
        if (std::abs(maxEigValue - minEigValue) > 1.0e-12)
        {
            maxEigValue += 0.001 * std::abs(maxEigValue - minEigValue);
        }

        this->setIntervalStopCondition();

        return getMinIntervalEigenSystem_Base(minEigValue, lambdaMax, maxEigValue,
                                              subspaceTol, subspaceIncrementSize, bufferSize, maxEigensystemDim,
                                              vStart, oP, randOp, eigValues, eigVectors);
    }

    //
    // Restrictions on the spectral ranges estimates:
    //
    // maxEigValueBound > maxEigValue
    // minEigValueEst   > minEigValue
    //
    // While it is technically ok to have lambdaMax < minEigValueEst, in such
    // cases it's probably better to use the version that estimates the
    // the minimal eigenvalue for you.
    //

    RAY_INT getMinIntervalEigenSystem(double minEigValueEst, double lambdaMax, double maxEigValueBound,
                                   double subspaceTol, RAY_INT subspaceIncrementSize, RAY_INT bufferSize, RAY_INT maxEigensystemDim,
                                   Vtype &vStart, Otype &oP, VRandomizeOpType &randOp, std::vector<double> &eigValues,
                                   Atype &eigVectors)
    {
        this->setIntervalStopCondition();

        return getMinIntervalEigenSystem_Base(minEigValueEst, lambdaMax, maxEigValueBound,
                                              subspaceTol, subspaceIncrementSize, bufferSize, maxEigensystemDim,
                                              vStart, oP, randOp, eigValues, eigVectors);
    }

    //
    //  Base routine
    //
    //  If the nonRandomStart flag is set, the code will use all available or up to
    //  subspaceSize = subspaceIncrementSize + bufferSize vectors that are
    //  specified in the eigVectors input argument.
    //
protected:
    RAY_INT getMinIntervalEigenSystem_Base(double minEigValue, double lambdaMax, double maxEigValue,
                                        double subspaceTol, RAY_INT subspaceIncrementSize, RAY_INT bufferSize, RAY_INT maxEigensystemDim,
                                        Vtype &vStart, Otype &oP, VRandomizeOpType &randOp, std::vector<double> &eigValues,
                                        Atype &eigVectors)
    {
        res_residualCheckCount = -1;
        gqr_lwork = -1;
        geqrf_lwork = -1;
        if (cusolverDn_handle == NULL)
        {
            cusolverErrchk(cusolverDnCreate(&cusolverDn_handle));
        }
        if (cublas_handle == NULL)
        {
            cublasErrchk(cublasCreate(&cublas_handle));
        }

        OpPtr = &oP; // Pointer to input operator for use by supporting member functions

#ifdef _OPENMP
        int threadCount = omp_get_max_threads();
#endif

        this->setupCountData();
        this->setupFinalData();

        /////////////////////////////////////////////////////////////////////
        // Compile with _TIMING defined for capturing timing data
        // otherwise timeing routines are noOps
        /////////////////////////////////////////////////////////////////////

        this->setupTimeData();
        this->startGlobalTimer();

        std::vector<double> residualHistory;

        // Insure that subspaceTol isn't too small

        if (subspaceTol < RAYLEIGH_CHEBYSHEV_SMALL_TOL_)
        {
            subspaceTol = RAYLEIGH_CHEBYSHEV_SMALL_TOL_;
        }
        double relErrFactor;
        //
        // Delete any old eigenvalues and eigenvectors if not random start, otherwise
        // use input eigenvectors for as much of the initial subspace as possible

        eigValues.clear();
        eigVecResiduals.clear();

        if (not nonRandomStartFlag)
        {
            eigVectors.resize((RAY_INT)(vStart.getDimension()), (RAY_INT)0);
        }

        RAY_INT returnFlag = 0;

        double lambdaStar;
        RAY_INT subspaceSize;
        RAY_INT foundSize;
        RAY_INT foundCount;

        bool completedBasisFlag = false;

        std::vector<double> oldEigs;
        std::vector<double> eigDiffs;
        std::vector<double> oldEigDiffs;
        std::vector<double> subspaceResiduals;

        double eigDiff;

        lambdaStar = maxEigValue;
        subspaceSize = subspaceIncrementSize + bufferSize;
        foundSize = 0;

        //
        // Reset sizes if subspaceSize is larger
        // than dimension of system

        RAY_INT vectorDimension = vStart.getDimension();

        if (subspaceSize > vectorDimension)
        {
            if (subspaceIncrementSize < vectorDimension)
            {
                bufferSize = vectorDimension - subspaceIncrementSize;
                subspaceSize = vectorDimension;
            }
            else
            {
                subspaceSize = vectorDimension;
                subspaceIncrementSize = vectorDimension;
                bufferSize = 0;
            }

            maxEigensystemDim = vectorDimension;
        }

        oldEigs.resize(subspaceSize, 0.0);
        eigDiffs.resize(subspaceSize, 1.0);
        oldEigDiffs.resize(subspaceSize, 1.0);

        //
        // mArray contains the current subspace, and is of size
        // equal to the sum of number of desired states and
        // the number of  buffer vectors
        //
        //
        // eigVectors is the array of eigenvectors that have
        // been found
        //

        mArray.resize(vStart.getDimension(), subspaceSize);
        mArrayTmp.resize(vStart.getDimension(), subspaceSize);

        VtAVeigValue.resize(subspaceSize, 0.0);

        VtAV.resize(subspaceSize, subspaceSize);
        VtAVeigVector.resize(subspaceSize, subspaceSize);
        tau.resize(subspaceSize, 0.0);

        RAY_INT starDegree = 0;
        RAY_INT starDegreeSave = 0;
        double starBoundSave = 0.0;
        double shift = 0.0;
        double starBound = 0.0;
        double maxEigDiff = 0.0;
        double maxResidual = 0.0;
        double maxGap = 0.0;
        double stopCheckValue = 0.0;
        double eigDiffRatio = 0.0;

        RAY_INT residualCheckCount = 0;
        RAY_INT innerLoopCount = 0;

        double vtvEig;
        double vtvEigCheck;

        RAY_INT indexA_start;
        RAY_INT indexA_end;
        RAY_INT indexB_start;
        RAY_INT indexB_end;

        //
        // Initialize subspace vectors using random vectors, or input
        // starting vectors if the latter is specified.
        //

        if (not nonRandomStartFlag)
        {
            randOp.randomize(mArray);
            mArrayTmp.initialize(vStart, subspaceSize);
        }
        else
        {
            if (subspaceSize > (RAY_INT)eigVectors.getColSize())
            {

                randOp.randomize(mArray);
                for (RAY_INT i = 0; i < (RAY_INT)eigVectors.getRowSize(); i++)
                {
                    for (RAY_INT j = 0; j < (RAY_INT)eigVectors.getColSize(); j++)
                    {
                        mArray(i, j) = eigVectors(i, j);
                    }
                }

                mArrayTmp.initialize(vStart, subspaceSize);
            }
            else
            {
                mArray.initialize(eigVectors);
                mArrayTmp.initialize(vStart, subspaceSize);
            }
        }

        // Initialize temporaries

        vTemp.initialize(vStart);

#ifdef _OPENMP
        MtVarray.clear();
        MtVarray.resize(threadCount);

        for (RAY_INT k = 0; k < threadCount; k++)
        {
            MtVarray[k].initialize(vStart);
        }
#endif

        // Quick return if subspaceSize >= vector dimension

        if (vectorDimension == subspaceSize)
        {
            RAY_INT maxOrthoCheck = 10;
            RAY_INT orthoCheckCount = 1;
            mArray.host_to_device();
            orthogonalize(mArray);
            mArray.device_to_host();

            // Due to instability of modified Gram-Schmidt for creating an
            // orthonormal basis for a high dimensional vector space, multiple
            // orthogonalization passes may be needed.

            while ((OrthogonalityCheck(mArray, false) > 1.0e-12) && (orthoCheckCount <= maxOrthoCheck))
            {
                orthoCheckCount += 1;
                mArray.host_to_device();
                orthogonalize(mArray);
                mArray.device_to_host();
            }

            if (orthoCheckCount > maxOrthoCheck)
            {
                std::string errMsg = "\nXXXX RayleighChebyshevLMCuda Error XXXX";
                errMsg += "\nUnable to create basis for complete vector space.\n";
                errMsg += "\nReduce size of buffer and/or subspaceIncrement \n";
                throw std::runtime_error(errMsg);
            }
            mArray.host_to_device();
            formVtAV(mArray);

            computeVtVeigensystem(VtAV, VtAVeigValue, VtAVeigVector);

            createEigenVectorsAndResiduals(VtAVeigVector, mArray, subspaceSize, eigVecResiduals);

            mArray.host_to_device();
            eigVectors = mArray;
            eigValues = VtAVeigValue;
            return subspaceSize;
        }

        // Initialize filter polynomial operators

        cOp.initialize(oP);

        //
        //  ################## Main Loop ######################
        //
        if (fixedIterationCount)
        {
            maxInnerLoopCount = fixedIterationCount;
        }

        int exitFlag = 0;

        RAY_INT applyCount = 0;
        RAY_INT applyCountCumulative = 0;

        char charBuf[256];   // To enable use of C-style formatted output
        std::string oString; // To enable use of C-style formatted output
                             //
                             ////////////////////////////////////////////////////////////
                             //                Main loop
                             ////////////////////////////////////////////////////////////
                             //
        if (verboseFlag)
        {
            snprintf(charBuf, 256, "\nRayleighChebyshevLM eigensystem computation \n");
            std::cout << charBuf << std::endl;
            if (resultsStreamPtr)
            {
                *resultsStreamPtr << charBuf << std::endl;
            }
        }

        while (exitFlag == 0)
        {
            //
            //  Initialize old eigenvalue array using buffer values.
            //  This step is done for cases when the routine
            //  is called to continue an existing computation.
            //
            for (RAY_INT k = bufferSize; k < subspaceSize; k++)
            {
                oldEigs[k] = oldEigs[bufferSize];
            }
            //
            //  Randomize buffer vectors after first increment.
            //
            if (applyCountCumulative > 0)
            {
                randOp.randomize(mArray);
            }

            startTimer();

            // Orthogonalize working subspace (mArray)
            // Repeat orthogonalization initially to compensate for possible
            // inaccuracies using modified-Gram Schmidt. During iteration,
            // subspace orthogonality is preserved due to use of eigenvector
            // basis of projected operator.
            //


            mArray.host_to_device();
            // std::cout << mArray(0, 0) << " " << mArray(0, 1) << std::endl;
            // std::cout << mArray(1, 0) << " " << mArray(1, 1) << std::endl;
            // std::cout << std::endl;
            orthogonalize(mArray);
            orthogonalize(mArray);

            incrementTime("ortho");
            incrementCount("ortho", 2);

            lambdaStar = maxEigValue;
            eigDiffRatio = 1.0;
            innerLoopCount = 0;
            starDegreeSave = 0;
            starBoundSave = 0.0;

            double eMin;
            double eMax;

            stopCheckValue = subspaceTol + 1.0e10;

            applyCount = 0;
            residualHistory.clear();

            maxGap = 0.0;
            maxResidual = 0.0;
            maxEigDiff = 0.0;

            RAY_INT oscillationCount = 0;

            while ((stopCheckValue > subspaceTol) && (innerLoopCount < maxInnerLoopCount))
            {
                //
                //  Compute filter polynomial parameters
                //
                shift = -minEigValue;
                starDegreeSave = starDegree;
                starBoundSave = starBound;
                starDegree = 0;
                //
                //  Conditions to check for the identity matrix, and for multiplicity > subspace dimension,
                //  which can lead to an erroneous increase in polynomial filtering degree.
                //
                relErrFactor = getRelErrorFactor(minEigValue, subspaceTol);
                eigDiff = std::abs(minEigValue - lambdaStar) / relErrFactor;
                eMin = eigDiff;

                relErrFactor = getRelErrorFactor(maxEigValue, subspaceTol);
                eMax = std::abs(maxEigValue - lambdaStar) / relErrFactor;

                if // Identity matrix
                    ((eMin < subspaceTol) && (eMax < subspaceTol))
                {
                    starDegree = 1;
                    starBound = maxEigValue;
                }

                ////////ZZZZZZZZZZZZZZZZZZZZZZZ
                // For multiplicity > subspace dimension
                else if ((not(maxEigDiff > subspaceTol)) && (innerLoopCount > 3) && (eigDiffRatio < .2)) // .2 is slightly less than the secondary
                {                                                                                        // maximum of the Lanczos C polynmoial
                    starDegree = starDegreeSave;
                    starBound = starBoundSave;
                }

                //
                /////////////////////////////////////////////////////////////////////////////
                //         Determining filter polynomial parameters
                /////////////////////////////////////////////////////////////////////////////
                //
                //  Find the polynomial that captures eigenvalues between current minEigValue
                //  and lambdaStar. If the required polynomial is greater than minIntervalPolyDegreeMax,
                //  set starDegree = minIntervalPolyDegreeMax. This is a safe modification, as
                //  it always safe to use a polynomial of lower degree than required.
                //
                if (starDegree == 0)
                {
                    cPoly.getStarDegreeAndSpectralRadius(shift, maxEigValue,
                                                         lambdaStar, minIntervalPolyDegreeMax, starDegree, starBound);
                }

                //
                // 	Only allow for increasing degrees. A decrease in degree
                // 	arises when orthogonalization with respect to previously
                // 	found eigenvectors isn't sufficient to insure
                // 	their negligible contribution to the working subspace
                // 	after the polynomial in the operator is applied. The
                // 	fix here is to revert to the previous polynomial.
                //  Often when the eigensystems associated with
                //  working subspaces that overlap, the eigenvalues
                //  created won't be monotonically increasing. When this
                //  occurs the problems are corrected by doing a final
                //  projection of the subspace and an additional eigensystem
                //  computation.
                //
                //  The monotonically increasing degree is only enforced
                //  for a given subspace. When the subspace changes because
                //  because of a shift, the degree and bound are set to 1
                //  and the maximal eigenvalue bound respectively.
                //
                //
                if (starDegree < starDegreeSave)
                {
                    starDegree = starDegreeSave;
                    starBound = starBoundSave;
                }

                /////////////////////////////////////////////////////////////////////////////
                //      Applying filter polynomial to working subspace (mArray)
                /////////////////////////////////////////////////////////////////////////////

                cOp.setLanczosCpolyParameters(starDegree, filterRepetitionCount, starBound, shift);

                startTimer();

                // mArray.host_to_device();
                // std::cout << mArray(0, 0) << " " << mArray(0, 1) << std::endl;
                // std::cout << mArray(1, 0) << " " << mArray(1, 1) << std::endl;
                // std::cout << std::endl;
                if (not completedBasisFlag)
                {
                    cOp.apply(mArray);
                }

                applyCount += 1;
                this->incrementTime("OpApply");
                this->incrementCount("OpApply", starDegree * subspaceSize);

                /////////////////////////////////////////////////////////////////////////////
                //                Orthogonalizing working subspace (mArray)
                /////////////////////////////////////////////////////////////////////////////

                startTimer();

                indexA_start = 0;
                indexA_end = subspaceSize - 1;
                indexB_start = 0;
                indexB_end = foundSize - 1;

                //  Orthogonalize working subspace (mArray) to subspace of found eigenvectors (eigVectors)
                //  It is important to do this before orthgonalizing the new vectors with respect to each other.

                //  Orthogonalize the subspace vectors using Modified Gram-Schmidt
                orthogonalize(mArray);

                incrementTime("ortho");
                incrementCount("ortho");
                //
                // #############################################################################
                // 			Forming projection of operator onto working subspace (VtAV)
                // #############################################################################
                //
                startTimer();
    
                // mArray.device_to_host();
                // std::cout << mArray(0, 0) << " " << mArray(0, 1) << std::endl;
                // std::cout << mArray(1, 0) << " " << mArray(1, 1) << std::endl;

                formVtAV(mArray);

                incrementCount("OpApply", subspaceSize);

                /////////////////////////////////////////////////////////////////////////////
                //         Compute eigenvalues of  Vt*A*V
                /////////////////////////////////////////////////////////////////////////////

                computeVtVeigensystem(VtAV, VtAVeigValue, VtAVeigVector);

                // std::cout << VtAVeigValue[0] << " " << VtAVeigValue[1] << std::endl;

                /////////////////////////////////////////////////////////////////////////////
                // Compute new approximations to eigenvectors and evaluate selected residuals
                /////////////////////////////////////////////////////////////////////////////

                // Only check residuals of subspace eigenvectors for
                // the eigenvectors one is determining in this subspace,
                // e.g. do not check residuals of buffer vectors.

                if (foundSize + subspaceSize < vectorDimension)
                {
                    residualCheckCount = subspaceIncrementSize;
                }
                else
                {
                    residualCheckCount = subspaceSize;
                }

                subspaceResiduals.clear();


                // std::cout << "residualCheckCount " << residualCheckCount << std::endl;
                // std::cout << "VtAVeigValue.size() " << VtAVeigValue.size() << std::endl;
                // std::cout << "VtAVeigValue " << VtAVeigValue[0] << " " << VtAVeigValue[1] << std::endl;

                createEigenVectorsAndResiduals(VtAVeigVector, mArray, residualCheckCount, subspaceResiduals);

                // mArray.device_to_host();
                // std::cout << mArray(0, 0) << " " << mArray(0, 1) << std::endl;
                // std::cout << mArray(1, 0) << " " << mArray(1, 1) << std::endl;
                // std::cout << std::endl;
                // std::cout << subspaceResiduals[0] << std::endl;
                // exit(0);

                incrementCount("OpApply", residualCheckCount);

                maxResidual = 0.0;
                for (size_t k = 0; k < subspaceResiduals.size(); k++)
                {
                    relErrFactor = getRelErrorFactor(VtAVeigValue[k], subspaceTol);
                    maxResidual = std::max(maxResidual, subspaceResiduals[k] / relErrFactor);
                }

                residualHistory.push_back(maxResidual);

                incrementTime("eigenvalue");
                incrementCount("eigenvalue");
                /////////////////////////////////////////////////////////////////////////////
                //
                /////////////////////////////////////////////////////////////////////////////

                if (verboseSubspaceFlag)
                {
                    oString.clear();
                    snprintf(charBuf, 256, "XXXX Subspace Eigs XXXX \n");
                    oString = charBuf;
                    for (RAY_INT i = 0; i < subspaceSize; i++)
                    {
                        snprintf(charBuf, 256, "%3ld : %+10.5e \n", i, VtAVeigValue[i]);
                        oString += charBuf;
                    }
                    snprintf(charBuf, 256, "\n");
                    snprintf(charBuf, 256, "Shift      : %10.5e MaxEigValue : %10.5e \n ", shift, maxEigValue);
                    oString += charBuf;
                    snprintf(charBuf, 256, "LambdaStar : %10.5e StarBound   : %10.5e StarDegree : %3ld \n", lambdaStar, starBound, starDegree);
                    oString += charBuf;
                    snprintf(charBuf, 256, "XXXXXXXXXXXXXXXXXXXXXXX \n");
                    oString += charBuf;
                    std::cout << oString << std::endl;
                    if (resultsStreamPtr)
                    {
                        *resultsStreamPtr << oString << std::endl;
                    }
                }

                //
                //  Determining the subspace size to check for eigenvalue convergence.
                //  Ignore the eigenvalues associated with the buffer vectors, except
                //  buffer vector with smallest eigenvalue when determining eigenvalues
                //  over an interval.
                //

                RAY_INT eigSubspaceCheckSize;

                if (intervalStopConditionFlag)
                {
                    eigSubspaceCheckSize = subspaceIncrementSize + 1;
                }
                else
                {
                    eigSubspaceCheckSize = subspaceIncrementSize;
                }

                for (RAY_INT i = 0; i < eigSubspaceCheckSize; i++)
                {
                    oldEigDiffs[i] = eigDiffs[i];
                }

                maxEigDiff = 0.0;

                for (RAY_INT i = 0; i < eigSubspaceCheckSize; i++)
                {
                    eigDiff = std::abs(VtAVeigValue[i] - oldEigs[i]);
                    relErrFactor = getRelErrorFactor(oldEigs[i], subspaceTol);
                    eigDiff = eigDiff / relErrFactor;
                    eigDiffs[i] = eigDiff;
                    maxEigDiff = (eigDiff > maxEigDiff) ? eigDiff : maxEigDiff;
                }

                for (RAY_INT i = 0; i < subspaceSize; i++)
                {
                    oldEigs[i] = VtAVeigValue[i];
                }

                //
                // Compute an average estimated convergence rate based upon components
                // for which the convergence tolerance has not been achieved.
                //

                RAY_INT diffCount = 0;
                eigDiffRatio = 0.0;
                if (maxEigDiff > subspaceTol)
                {
                    for (RAY_INT i = 0; i < eigSubspaceCheckSize; i++)
                    {
                        if (std::abs(oldEigDiffs[i]) > subspaceTol / 10.0)
                        {
                            eigDiffRatio += std::abs(eigDiffs[i] / oldEigDiffs[i]);
                            diffCount++;
                        }
                    }
                    if (diffCount > 0)
                    {
                        eigDiffRatio /= (double)diffCount;
                    }
                    else
                    {
                        eigDiffRatio = 1.0;
                    }
                }

                double spectralRange = std::abs((lambdaMax - minEigValue));

                maxGap = 0.0;
                for (RAY_INT i = 1; i < eigSubspaceCheckSize; i++)
                {
                    maxGap = std::max(maxGap, std::abs(VtAVeigValue[i] - VtAVeigValue[i - 1]) / spectralRange);
                }

                if (verboseFlag)
                {
                    snprintf(charBuf, 256, "%-5ld : Degree %-3ld  Residual Max: %-10.5g  Eig Diff Max: %-10.5g  Eig Conv Factor: %-10.5g Max Gap %-10.5g",
                             innerLoopCount, starDegree, maxResidual, maxEigDiff, eigDiffRatio, maxGap);
                    std::cout << charBuf << std::endl;
                    if (resultsStreamPtr)
                    {
                        *resultsStreamPtr << charBuf << std::endl;
                    }
                }

                // Create value to determine when iteration should terminate

                if (stopCondition == RC_Types::StopCondition::RESIDUAL_ONLY)
                {
                    stopCheckValue = maxResidual;
                }
                else if (stopCondition == RC_Types::StopCondition::EIGENVALUE_ONLY)
                {
                    stopCheckValue = maxEigDiff;
                }
                else // Stop based up convergence of eigenvalues and residuals  < sqrt(subspaceTol)
                {
                    stopCheckValue = maxEigDiff;

                    if (maxResidual > std::sqrt(subspaceTol))
                    {
                        stopCheckValue = maxResidual;
                    }
                }

                //
                // Force termination if we've filled out the subspace
                //

                if (subspaceIncrementSize == 0)
                {
                    stopCheckValue = 0.0;
                }

                // When using residual stop tolearnce force termination
                // if residual is oscillating and has value < sqrt(subspaceTol)
                //
                RAY_INT rIndex;
                double residual2ndDiffA;
                double residual2ndDiffB;

                if ((stopCondition == RC_Types::StopCondition::RESIDUAL_ONLY) && (maxResidual < std::sqrt(subspaceTol)))
                {
                    rIndex = residualHistory.size() - 1;

                    if (rIndex > 3)
                    {
                        residual2ndDiffA = (residualHistory[rIndex - 3] - 2.0 * residualHistory[rIndex - 2] + residualHistory[rIndex - 1]) / (std::abs(maxResidual));
                        residual2ndDiffB = (residualHistory[rIndex - 2] - 2.0 * residualHistory[rIndex - 1] + residualHistory[rIndex]) / (std::abs(maxResidual));
                        if (residual2ndDiffA * residual2ndDiffB < 0.0)
                        {
                            oscillationCount += 1;
                            if (oscillationCount > 5)
                            {
                                stopCheckValue = 0.0;
                                if (verboseFlag)
                                {
                                    oString.clear();
                                    snprintf(charBuf, 256, "Warning : Oscillatory residuals observed when max residual less than square root of subspace tolerance.\n");
                                    oString = charBuf;
                                    snprintf(charBuf, 256, "          RayleighChebyshevLMCuda subspace iteration stopped before residual termination criterion met.\n");
                                    oString += charBuf;
                                    snprintf(charBuf, 256, "          Subspace tolerance specified :  %10.5e \n", subspaceTol);
                                    oString += charBuf;
                                    snprintf(charBuf, 256, "          Resididual obtained          :  %10.5e \n", maxResidual);
                                    oString += charBuf;
                                    snprintf(charBuf, 256, "Typical remediation involves either increasing subspace tolerance or buffer size.\n");
                                    oString += charBuf;
                                    std::cout << oString << std::endl;
                                    if (resultsStreamPtr)
                                    {
                                        *resultsStreamPtr << oString << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }

                //
                // Update cPoly parameters based upon the eigensystem computation.
                //
                //
                // lambdaStar  : is reset to the largest eigenvalue currently computed
                // minEigValue : is reset when the subspace computation yields a
                // minimum eigenvalue smaller than minEigValue.
                //

                lambdaStar = VtAVeigValue[subspaceSize - 1];
                minEigValue = (minEigValue < VtAVeigValue[0]) ? minEigValue : VtAVeigValue[0];

                innerLoopCount++;
            }

            // Capture inner loop parameters

            finalData["maxResidual"] = std::max(maxResidual, finalData["maxResidual"]);
            finalData["maxEigValueDifference"] = std::max(maxEigDiff, finalData["maxEigValueDifference"]);
            finalData["maxRelEigValueGap"] = std::max(maxGap, finalData["maxRelEigValueGap"]);

            applyCountCumulative += applyCount;

            if (verboseFlag == 1)
            {
                if ((not fixedIterationCount) && (innerLoopCount >= maxInnerLoopCount))
                {
                    oString.clear();
                    snprintf(charBuf, 256, " Warning             : Maximal number of iterations taken before tolerance reached \n");
                    oString = charBuf;
                    snprintf(charBuf, 256, " Iterations taken    : %ld \n", innerLoopCount);
                    oString += charBuf;
                    snprintf(charBuf, 256, " Eig Diff Max        : %-10.5g \n", maxEigDiff);
                    oString += charBuf;
                    snprintf(charBuf, 256, " Residual Max        : %-10.5g \n", maxResidual);
                    oString += charBuf;
                    snprintf(charBuf, 256, " Requested Tolerance : %-10.5g \n", subspaceTol);
                    oString += charBuf;
                    std::cout << oString << std::endl;
                    if (resultsStreamPtr)
                    {
                        *resultsStreamPtr << oString << std::endl;
                    }
                }
            }

            foundCount = 0;
            //
            //  Capture the found eigenpairs
            //
            RAY_INT checkIndexCount;

            if (foundSize + subspaceSize < vectorDimension)
            {
                checkIndexCount = subspaceIncrementSize;
            }
            else
            {
                checkIndexCount = subspaceSize;
            }

            // Check for eigenvalues less than maximal eigenvalue

            for (RAY_INT i = 0; i < checkIndexCount; i++)
            {
                vtvEig = VtAVeigValue[i];
                relErrFactor = getRelErrorFactor(lambdaMax, subspaceTol);

                vtvEigCheck = (vtvEig - lambdaMax) / relErrFactor; // Signed value is important here

                if (vtvEigCheck < subspaceTol)
                    foundCount++;
            }

            if ((foundCount + foundSize) >= maxEigensystemDim)
            {
                foundCount = maxEigensystemDim - foundSize;
                exitFlag = 1;
            }

            if ((foundCount + foundSize) >= vectorDimension)
            {
                foundCount = vectorDimension - foundSize;
                exitFlag = 1;
            }

            // Capture found eigenvalues and eigenvectors

            if (foundCount > 0)
            {
                mArray.device_to_host();
                eigVectors.resize(foundSize + foundCount, vStart);
                eigValues.resize(foundSize + foundCount, 0.0);

#pragma omp parallel for
                for (RAY_INT i = 0; i < foundCount; i++)
                {

                    for (RAY_INT j = 0; j < mArray.getRowSize(); j++)
                    {
                        eigVectors(j, foundSize + i) = mArray(j, i);
                    }
                }

                for (RAY_INT i = 0; i < foundCount; i++)
                {
                    eigValues[foundSize + i] = VtAVeigValue[i];
                    eigVecResiduals.push_back(subspaceResiduals[i]);
                }

                foundSize += foundCount;
                if (verboseFlag)
                {
                    snprintf(charBuf, 256, "\nFound Count: %3ld Largest Eig Found: %-20.15g Lambda Bound: %-20.15g \n", foundSize, eigValues[foundSize - 1], lambdaMax);
                    std::cout << charBuf << std::endl;
                    if (resultsStreamPtr)
                    {
                        *resultsStreamPtr << charBuf << std::endl;
                    }
                }
            }

            //
            // Shuffle all computed vectors to head of mArray
            //
            if (not exitFlag)
            {
                for (RAY_INT k = 0; k + foundCount < subspaceSize; k++)
                {
                    for (RAY_INT j = 0; j < mArray.getRowSize(); j++)
                    {
                        mArrayTmp(j, k) = mArray(j, k + foundCount);
                    }
                }
            }
            //
            //  See if "guard" eigenvalue is greater than lambdaMax.
            //
            if (bufferSize > 0)
            {
                vtvEig = VtAVeigValue[subspaceSize - bufferSize];
            }
            else
            {
                vtvEig = lambdaMax;
            }

            guardValue = vtvEig;
            relErrFactor = getRelErrorFactor(lambdaMax, subspaceTol);
            vtvEigCheck = (vtvEig - lambdaMax) / relErrFactor;

            //
            //  Using a hard interval stop when computing a fixed number of eigenpairs.
            //
            //  We assume that states are degenerate if |lambda[i] - lambda[i+1]|/relErrFactor < 10.0*subspaceTol
            //  and hence require a relative gap of size 10.0*subspaceTol between lambdaMax and the guard vector
            //  to insure that all vectors in the subspace associated with an eigenvalue with multiplicity > 1
            //  are captured.

            if (hardIntervalStopFlag)
            {
                if (guardValue > lambdaMax)
                {
                    exitFlag = 1;
                }
            }
            else
            {
                if (vtvEigCheck >= 10.0 * subspaceTol)
                {
                    exitFlag = 1;
                }
            }

            //
            // Shifting minEigenValue
            //

            minEigValue = eigValues[foundSize - 1]; // New minEigValue = largest of found eigenvalues

            // Reset star degree and bound and addjust

            starDegree = 1;
            starBound = maxEigValue;
            //
            //  Check for exceeding vector space dimension
            //  this step always reduces the subspace size, so the
            //  the resize of the mArray does not alter the initial
            //  elements of the mArray.
            //
            if (not exitFlag)
            {
                if ((foundSize + subspaceSize) >= vectorDimension)
                {
                    // The computational subspace fills out the dimension of the
                    // vector space so the last iteration will just be
                    // projection onto a collection of random vectors that
                    // are orthogonal to the current collection of eigenvectors.
                    //
                    if (foundSize + subspaceIncrementSize >= vectorDimension)
                    {
                        bufferSize = 0;
                        subspaceIncrementSize = vectorDimension - foundSize;
                        subspaceSize = subspaceIncrementSize;
                        completedBasisFlag = true;
                        mArray.resize(subspaceSize);

                        randOp.randomize(mArray);
                    }
                    else
                    {
                        bufferSize = vectorDimension - (foundSize + subspaceIncrementSize);
                        subspaceSize = subspaceIncrementSize + bufferSize;
                        mArray.resize(subspaceSize);
                    }

                    mArrayTmp.resize(subspaceSize);

                    VtAV.resize(subspaceSize, subspaceSize);
                    VtAVeigVector.initialize(subspaceSize, subspaceSize);
                    VtAVeigValue.resize(subspaceSize, 0.0);
                }
            }

        } // end of main loop

        //
        // Validate that the eigenvectors computed are associated with a monotonically increasing
        // sequence of eigenvalues. If not, this problem is corrected by recomputing the eigenvalues
        // and eigenvectors of the projected operator.
        //
        // This task is typically only required when when the desired eigenvalues are associated
        // with multiple clusters where
        // (a) the eigenvalues in the clusters are nearly degenerate
        // (b) the subspaceIncrement size is insufficient to contain a complete collection of
        //     clustered states, e.g. when the approximating subspace "splits" a cluster.
        //
        // This task will never be required if the subspaceIncrementSize > number of eigenvalues
        // that are found.
        //
        //
        bool nonMonotoneFlag = false;

        for (RAY_INT i = 0; i < foundSize - 1; i++)
        {
            if (eigValues[i] > eigValues[i + 1])
            {
                nonMonotoneFlag = true;
            }
        }

        if (nonMonotoneFlag)
        {
            eigVectors.host_to_device();
            foundSize = eigVectors.getColSize();
            mArrayTmp.resize(foundSize, vStart);

            VtAV.resize(foundSize, foundSize);
            VtAVeigVector.initialize(foundSize, foundSize);
            VtAVeigValue.resize(foundSize, 0.0);

            startTimer();

            /////////////////////////////////////////////////////////////////////////////
            //     Form projection of operation on subspace of found eigenvectors
            /////////////////////////////////////////////////////////////////////////////
            formVtAV(eigVectors);

            incrementCount("OpApply", foundSize);

            /////////////////////////////////////////////////////////////////////////////
            //             Compute eigenvalues of  Vt*A*V
            /////////////////////////////////////////////////////////////////////////////

            computeVtVeigensystem(VtAV, VtAVeigValue, VtAVeigVector);

            /////////////////////////////////////////////////////////////////////////////
            //                   Create eigenvectors
            /////////////////////////////////////////////////////////////////////////////

            eigVecResiduals.clear();
            createEigenVectorsAndResiduals(VtAVeigVector, eigVectors, foundSize, eigVecResiduals);

            incrementCount("OpApply", foundSize);
            incrementTime("eigenvalue");
            incrementCount("eigenvalue");
            eigVectors.device_to_host();

        } // End non-monotone eigensystem correction

        //
        //   In the case of computing all eigenvalues less than a specified bound,
        //   trim the resulting vectors to those that are within tolerance of
        //   being less than lambdaMax
        //
        //
        RAY_INT finalFoundCount = 0;

        if (intervalStopConditionFlag)
        {
            relErrFactor = getRelErrorFactor(lambdaMax, subspaceTol);
            for (RAY_INT i = 0; i < (RAY_INT)eigValues.size(); i++)
            {
                if ((eigValues[i] - lambdaMax) / relErrFactor < subspaceTol)
                {
                    finalFoundCount++;
                }
            }

            if (finalFoundCount < (RAY_INT)eigValues.size())
            {
                eigValues.resize(finalFoundCount);
                eigVectors.resize(eigVectors.getRowSize(), finalFoundCount);
                eigVecResiduals.resize(finalFoundCount);
                foundSize = finalFoundCount;
            }
        }

        this->incrementTotalTime();
        if (verboseFlag)
        {
            snprintf(charBuf, 256, "Total Found Count: %3ld Largest Eig Found: %-20.15g Lambda Bound: %-20.15g \n", foundSize, eigValues[foundSize - 1], lambdaMax);
            std::cout << charBuf << std::endl;
            if (resultsStreamPtr)
            {
                *resultsStreamPtr << charBuf << std::endl;
            }
        }
        times[0] = timeValue["ortho"];
        times[1] = timeValue["OpApply"];
        times[2] = timeValue["eigenvalue"];
        times[3] = timeValue["totalTime"];
        if (eigDiagnosticsFlag == 1)
        {
            oString.clear();
            snprintf(charBuf, 256, "\nXXXX RayleighChebyshevLMCuda Diagnostics XXXXX \n");
            oString = charBuf;

#ifdef _OPENMP
            snprintf(charBuf, 256, "XXXX --- Using OpenMP Constructs  --- XXXX\n");
            oString += charBuf;
#endif

            snprintf(charBuf, 256, "\n");
            snprintf(charBuf, 256, "Total_Iterations        : %-ld   \n", applyCountCumulative);
            oString += charBuf;
            snprintf(charBuf, 256, "Total_OpApply           : %-ld   \n", countData["OpApply"]);
            oString += charBuf;
            snprintf(charBuf, 256, "Total_SubspaceEig       : %-ld   \n", countData["eigenvalue"]);
            oString += charBuf;
            snprintf(charBuf, 256, "Total_Orthogonalization : %-ld   \n", countData["ortho"]);
            oString += charBuf;

#ifdef TIMING_
            snprintf(charBuf, 256, "TotalTime_Sec : %10.5f \n", timeValue["totalTime"]);
            oString += charBuf;
            snprintf(charBuf, 256, "OrthoTime_Sec : %10.5f \n", timeValue["ortho"]);
            oString += charBuf;
            snprintf(charBuf, 256, "ApplyTime_Sec : %10.5f \n", timeValue["OpApply"]);
            oString += charBuf;
            snprintf(charBuf, 256, "EigTime_Sec   : %10.5f \n", timeValue["eigenvalue"]);
            oString += charBuf;
#endif

            std::cout << oString << std::endl;
            if (resultsStreamPtr)
            {
                *resultsStreamPtr << oString << std::endl;
            }
        }

        if (fixedIterationCount > 0)
        {
            this->resetMaxInnerLoopCount();
        }

        if (foundSize >= 0)
            returnFlag = foundSize;
        return returnFlag;
    }

    /////////////////////////////////////////////////////////////////////
    //
    /////////////////////////////////////////////////////////////////////

    void orthogonalize(Atype &M)
    {

        if (tau_d == NULL)
        {
            cudaErrchk(cudaMalloc((void **)&tau_d, sizeof(Dtype) * M.getColSize()));
        }
        else if (M.getColSize() > tau_n)
        {
            cudaErrchk(cudaFree(tau_d));
            cudaErrchk(cudaMalloc((void **)&tau_d, sizeof(Dtype) * M.getColSize()));
            tau_n = M.getColSize();
        }

        if (M.getRowSize() > geqrf_m || M.getColSize() > geqrf_n)
        {
            cudaErrchk(cudaFree(geqrf_work_d));
            cudaErrchk(cudaFree(gqr_work_d));

            geqrf_m = M.getRowSize();
            geqrf_n = M.getColSize();
        }

        if (geqrf_work_d == NULL || (M.getRowSize() > geqrf_m || M.getColSize() > geqrf_n))
        {

            if constexpr (std::is_same<Dtype, double>::value)
            {
                cusolverErrchk(cusolverDnDgeqrf_bufferSize(
                    cusolverDn_handle,
                    M.getRowSize(),
                    M.getColSize(),
                    M.getDataPointer_d(),
                    M.getRowSize(),
                    &geqrf_lwork));

                cusolverErrchk(cusolverDnDorgqr_bufferSize(
                    cusolverDn_handle,
                    M.getRowSize(),
                    M.getColSize(),
                    M.getColSize(),
                    M.getDataPointer_d(),
                    M.getRowSize(),
                    tau_d,
                    &gqr_lwork));

                cudaErrchk(cudaMalloc((void **)&geqrf_work_d, sizeof(Dtype) * geqrf_lwork));
                cudaErrchk(cudaMalloc((void **)&gqr_work_d, sizeof(Dtype) * gqr_lwork));
            }
            else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
            {
                cusolverErrchk(cusolverDnZgeqrf_bufferSize(
                    cusolverDn_handle,
                    M.getRowSize(),
                    M.getColSize(),
                    (cuDoubleComplex *)M.getDataPointer_d(),
                    M.getRowSize(),
                    &geqrf_lwork));

                cusolverErrchk(
                    cusolverDnZungqr_bufferSize(
                        cusolverDn_handle,
                        M.getRowSize(),
                        M.getColSize(),
                        M.getColSize(),
                        (cuDoubleComplex *)M.getDataPointer_d(),
                        M.getRowSize(),
                        (cuDoubleComplex *)tau_d,
                        &gqr_lwork));
                cudaErrchk(cudaMalloc((void **)&geqrf_work_d, sizeof(Dtype) * geqrf_lwork));
                cudaErrchk(cudaMalloc((void **)&gqr_work_d, sizeof(Dtype) * gqr_lwork));
            }
        }

        if (info_d == NULL)
        {
            cudaErrchk(cudaMalloc((void **)&info_d, sizeof(int)));
        }

        if constexpr (std::is_same<Dtype, double>::value)
        {
            cusolverErrchk(cusolverDnDgeqrf(
                cusolverDn_handle,
                M.getRowSize(),
                M.getColSize(),
                M.getDataPointer_d(),
                M.getRowSize(),
                tau_d,
                geqrf_work_d,
                geqrf_lwork,
                info_d));

            cusolverErrchk(cusolverDnDorgqr(
                cusolverDn_handle,
                M.getRowSize(),
                M.getColSize(),
                M.getColSize(),
                M.getDataPointer_d(),
                M.getRowSize(),
                tau_d,
                gqr_work_d,
                gqr_lwork,
                info_d));
        }
        else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
        {
            cusolverErrchk(cusolverDnZgeqrf(
                cusolverDn_handle,
                M.getRowSize(),
                M.getColSize(),
                (cuDoubleComplex *)M.getDataPointer_d(),
                M.getRowSize(),
                (cuDoubleComplex *)tau_d,
                (cuDoubleComplex *)geqrf_work_d,
                geqrf_lwork,
                info_d));
            cusolverErrchk(cusolverDnZungqr(
                cusolverDn_handle,
                M.getRowSize(),
                M.getColSize(),
                M.getColSize(),
                (cuDoubleComplex *)M.getDataPointer_d(),
                M.getRowSize(),
                (cuDoubleComplex *)tau_d,
                (cuDoubleComplex *)gqr_work_d,
                gqr_lwork,
                info_d));
        }
    }

    // Assumes the operator is symmetric (or complex Hermitian).
    //
    // Only the upper triangular part of the projection of the
    // operator is formed and the lower part is obtained by
    // reflection.

    void formVtAV(Atype &V)
    {
        RAY_INT subspaceSize = (RAY_INT)V.getColSize();

        // tmp = A@V
        Dtype alpha;
        Dtype beta;

        if constexpr (std::is_same<Dtype, double>::value)
        {
            alpha = 1.0;
            beta = 0.0;
        }
        else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
        {
            alpha = std::complex<double>(1.0, 0.0);
            beta = std::complex<double>(0.0, 0.0);
        }

        // VtAV = Vt@tmp
        OpPtr->apply(V, mArrayTmp, alpha, beta);

        if constexpr (std::is_same<Dtype, double>::value)
        {
            cublasErrchk(
                cublasDgemm(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    mArrayTmp.getColSize(), mArrayTmp.getColSize(), mArrayTmp.getRowSize(),
                    &alpha,
                    V.getDataPointer_d(), mArrayTmp.getRowSize(),
                    mArrayTmp.getDataPointer_d(), mArrayTmp.getRowSize(),
                    &beta,
                    VtAV.getDataPointer_d(), mArrayTmp.getColSize()));
        }
        else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
        {
            cublasErrchk(
                cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_C, CUBLAS_OP_N,
                    mArrayTmp.getColSize(), mArrayTmp.getColSize(), mArrayTmp.getRowSize(),
                    (cuDoubleComplex *)&alpha,
                    (cuDoubleComplex *)V.getDataPointer_d(), mArrayTmp.getRowSize(),
                    (cuDoubleComplex *)mArrayTmp.getDataPointer_d(), mArrayTmp.getRowSize(),
                    (cuDoubleComplex *)&beta,
                    (cuDoubleComplex *)VtAV.getDataPointer_d(), mArrayTmp.getColSize()));
        }
    }

    //
    // Input
    //
    // V              : the collection of vectors in the subspace
    // VtAVeigVector  : the eigenvectors of the projected operator
    //
    // Output
    // V              : Approximate eigenvectors of the operator based
    //                  upon the eigenvectors of the projected operator
    //
    // eigVresiduals  : Residuals of the first residualCheckCount approximate eigenvectors
    //                  ordered by eigenvalues, algebraically smallest to largest.
    //
    void createEigenVectorsAndResiduals(Atype &VtAVeigVector, Atype &V,
                                        RAY_INT residualCheckCount, std::vector<double> &eigVresiduals)
    {
        RAY_INT subspaceSize = (RAY_INT)V.getColSize();

        Dtype alpha;
        Dtype beta;

        if constexpr (std::is_same<Dtype, double>::value)
        {
            alpha = 1.0;
            beta = 0.0;
        }
        else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
        {
            alpha = std::complex<double>(1.0, 0.0);
            beta = std::complex<double>(0.0, 0.0);
        }

        // e = V@e_tilde
        if constexpr (std::is_same<Dtype, double>::value)
        {
            cublasErrchk(
                cublasDgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    mArrayTmp.getRowSize(), mArrayTmp.getColSize(), mArrayTmp.getColSize(),
                    &alpha,
                    V.getDataPointer_d(), V.getRowSize(),
                    VtAVeigVector.getDataPointer_d(), VtAVeigVector.getRowSize(),
                    &beta,
                    mArrayTmp.getDataPointer_d(), mArrayTmp.getRowSize()));
        }
        else if constexpr (std::is_same<Dtype, std::complex<double>>::value)
        {
            cublasErrchk(
                cublasZgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    mArrayTmp.getRowSize(), mArrayTmp.getColSize(), mArrayTmp.getColSize(),
                    (cuDoubleComplex *)&alpha,
                    (cuDoubleComplex *)V.getDataPointer_d(), V.getRowSize(),
                    (cuDoubleComplex *)VtAVeigVector.getDataPointer_d(), VtAVeigVector.getRowSize(),
                    (cuDoubleComplex *)&beta,
                    (cuDoubleComplex *)mArrayTmp.getDataPointer_d(), mArrayTmp.getRowSize()));
        }


        cuda_kernels::normalize(
            mArrayTmp.getDataPointer_d(),
            mArrayTmp.getRowSize(),
            mArrayTmp.getColSize()
        );

        cudaErrchk(cudaMemcpy(V.getDataPointer_d(),
                              mArrayTmp.getDataPointer_d(), sizeof(Dtype) * mArrayTmp.getDimension(), cudaMemcpyDeviceToDevice));

        OpPtr->apply(V, mArrayTmp, alpha, beta);

        if (res_residualCheckCount != residualCheckCount)
        {
            if (eigVresiduals_d != NULL)
            {
                cudaErrchk(cudaFree(eigVresiduals_d));
            }
            cudaErrchk(cudaMalloc((void **)&eigVresiduals_d, sizeof(double) * residualCheckCount));

            if (VtAVeigValue_d != NULL)
            {
                cudaErrchk(cudaFree(VtAVeigValue_d));
            }
            cudaErrchk(cudaMalloc((void **)&VtAVeigValue_d, sizeof(double) * residualCheckCount));

            if (eigVresiduals_h != NULL)
            {
                delete[] eigVresiduals_h;
            }
            eigVresiduals_h = new double[residualCheckCount];

            if (VtAVeigValue_h != NULL)
            {
                delete[] VtAVeigValue_h;
            }
            VtAVeigValue_h = new double[residualCheckCount];

            res_residualCheckCount = residualCheckCount;
        }
        // for (RAY_INT i = 0; i < residualCheckCount; i++)
        // {
        //     VtAVeigValue_h[i] = VtAVeigValue[i];
        // }

        cudaErrchk(cudaMemcpy(VtAVeigValue_d,
                              VtAVeigValue.data(), sizeof(double) * residualCheckCount, cudaMemcpyHostToDevice));

        // AV - EV
        cuda_kernels::residuals(
            eigVresiduals_d,
            mArrayTmp.getDataPointer_d(),
            V.getDataPointer_d(),
            VtAVeigValue_d,
            mArrayTmp.getRowSize(),
            residualCheckCount);

        eigVresiduals.resize(residualCheckCount, 0.0);
        cudaErrchk(cudaMemcpy(eigVresiduals.data(),
                              eigVresiduals_d, sizeof(double) * residualCheckCount, cudaMemcpyDeviceToHost));
    }

    double OrthogonalityCheck(Atype &Amatrix, bool printOrthoCheck = false)
    {
        double orthoErrorMax = 0.0;

        for (size_t i = 0; i < Amatrix.getColSize(); i++)
        {
            for (size_t j = 0; j < Amatrix.getColSize(); j++)
            {

                Dtype inner_prod = Amatrix.template innerprod<Dtype>(i, j);
                if (printOrthoCheck)
                {
                    std::cout << inner_prod << " ";
                }

                if (i != j)
                {
                    orthoErrorMax = std::max(orthoErrorMax, std::abs(inner_prod));
                }
                else
                {
                    orthoErrorMax = std::max(orthoErrorMax, std::abs(inner_prod - 1.0));
                }
            }
            if (printOrthoCheck)
            {
                std::cout << std::endl;
            }
        }

        return orthoErrorMax;
    }

    void setupFinalData()
    {
        finalData["maxResidual"] = 0.0;
        finalData["maxEigValueDifference"] = 0.0;
        finalData["maxRelEigValueGap"] = 0.0;
#ifdef TIMING_
        finalData["totalTime"] = 0.0;
#endif
    }

    void setupCountData()
    {
        countData["ortho"] = 0;
        countData["OpApply"] = 0;
        countData["eigenvalue"] = 0;
    }

    void incrementCount(const std::string &countValue, RAY_INT increment = 1)
    {
        countData[countValue] += increment;
    }

    void setupTimeData()
    {
#ifdef TIMING_
        timeValue["ortho"] = 0.0;
        timeValue["OpApply"] = 0.0;
        timeValue["eigenvalue"] = 0.0;
        timeValue["totalTime"] = 0.0;
#endif
    }
    void startTimer()
    {
#ifdef TIMING_
        cudaErrchk(cudaDeviceSynchronize());
        timer.start();
#endif
    }

    void incrementTime(const std::string &timedValue)
    {
#ifdef TIMING_
        cudaErrchk(cudaDeviceSynchronize());
        timer.stop();
        timeValue[timedValue] += timer.getSecElapsedTime();
#endif
    }

    void startGlobalTimer()
    {
#ifdef TIMING_
        globalTimer.start();
#endif
    }
    void incrementTotalTime()
    {
#ifdef TIMING_
        globalTimer.stop();
        timeValue["totalTime"] = globalTimer.getSecElapsedTime();
        finalData["totalTime"] = timeValue["totalTime"];
#endif
    }

    public:
        double times[4];

    Otype *OpPtr;

    bool verboseSubspaceFlag;
    bool verboseFlag;
    bool eigDiagnosticsFlag;

    Atype mArray;
    Atype mArrayTmp;

    Vtype vTemp;

    // For storage of matrices and eigenvectors of projected system
    std::vector<Dtype> tau;
    Atype VtAV;
    Atype VtAVeigVector;

    std::vector<double> VtAVeigValue;
    std::vector<double> eigVecResiduals;

    LanczosMaxMinFinder<Vtype, Otype, VRandomizeOpType> lanczosMaxMinFinder;

    LanczosCpoly cPoly;
    LanczosCpolyOperatorLMCuda<Atype, Otype> cOp;

    JacobiDiagonalizer jacobiMethod;
    bool useJacobiFlag;

#ifndef RC_WITHOUT_LAPACK_

    SCC::DSYEV dsyev;

    SCC::ZHPEVX zhpevx;
#endif

    double guardValue;              // Value of the guard eigenvalue.
    bool intervalStopConditionFlag; // Converge based on value of guard eigenvalue
    bool hardIntervalStopFlag;
    double minEigValueEst;
    double maxEigValueEst;
    double maxMinTol;

    RAY_INT minIntervalPolyDegreeMax;
    RAY_INT maxInnerLoopCount;
    bool nonRandomStartFlag;
    bool fixedIterationCount;
    RAY_INT filterRepetitionCount;

    RC_Types::StopCondition stopCondition;

    std::map<std::string, RAY_INT> countData;

    std::map<std::string, double> finalData;

    std::ostream *resultsStreamPtr;

#ifdef TIMING_
    ClockIt timer;
    ClockIt globalTimer;
    std::map<std::string, double> timeValue;
    std::map<std::string, RAY_INT> timeCount;
#endif

    // Temporaries for multi-threading

#ifdef _OPENMP
    std::vector<Vtype> MtVarray;
#endif

    cusolverDnHandle_t cusolverDn_handle = NULL;
    int geqrf_lwork;
    Dtype *geqrf_work_d = NULL;
    int geqrf_m, geqrf_n;
    Dtype *tau_d = NULL;
    int tau_n;

    int gqr_lwork;
    Dtype *gqr_work_d = NULL;
    int *info_d = NULL;

    cublasHandle_t cublas_handle = NULL;

    int res_residualCheckCount;
    double *eigVresiduals_d = NULL;
    double *VtAVeigValue_d = NULL;
    double *eigVresiduals_h = NULL;
    double *VtAVeigValue_h = NULL;
};

#undef JACOBI_TOL
#undef DEFAULT_MAX_INNER_LOOP_COUNT
#undef RAYLEIGH_CHEBYSHEV_SMALL_TOL_
#undef DEFAULT_MAX_MIN_TOL
#undef DEFAULT_POLY_DEGREE_MAX
#undef DEFAULT_FILTER_REPETITION_COUNT
#undef DEFAULT_USE_JACOBI_FLAG
#undef DEFAULT_USE_RESIDUAL_STOP_CONDITION
#endif
