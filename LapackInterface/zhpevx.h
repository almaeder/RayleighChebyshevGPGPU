#pragma once
#include <complex>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include "SCC_LapackHeaders.h"
#include "SCC_LapackMatrix.h"
#include "SCC_LapackMatrixCmplx16.h"

#define RC_INT int

#ifndef ZHPEVX_
#define ZHPEVX_

namespace SCC
{
class ZHPEVX
{
public:
    ZHPEVX() {}

    void initialize()
    {
        AP.initialize();
        WORK.initialize();
        RWORK.clear();
        IWORK.clear();
        IFAIL.clear();
    }

    // Computes the eigCount algebraically smallest eigenvalues and eigenvectors.
    // The value returned is the number of eigenvalues found.

    RC_INT createEigensystem(SCC::LapackMatrixCmplx16 &A, std::vector<double> &eigValues, SCC::LapackMatrixCmplx16 &eigVectors)
    {
        if (A.getRowDimension() != A.getColDimension())
        {
            throw std::runtime_error("\nZHPEVX : Non-square matrix input argument  \n");
        }

        RC_INT N = A.getRowDimension();

        char JOBZ = 'V';  // Specify N for eigenvalues only
        char RANGE = 'A'; // Specify index range of eigenvalues to be find (A for all, V for interval)
        char UPLO = 'U';  // Store complex Hermetian matrix in upper trianglar packed form

        AP.initialize(A.createUpperTriPacked());

        double VL = 0;
        double VU = 0;

        RC_INT IL = 1; // Index of smallest eigenvalue returned
        RC_INT IU = N; // Index of largest  eigenvalue returned

        double ABSTOL = 2.0 * LAPACKE_dlamch('S');

        RC_INT M = 0; // Number of eigenvalues output

        eigValues.clear(); // W parameter in original call
        eigValues.resize(N, 0.0);

        RC_INT LDZ = N;
        RC_INT Mstar = N; // Maximal number of eigenvalues to be computed when using index specification

        eigVectors.initialize(LDZ, Mstar); // Matrix whose columns containing the eigenvectors (Z in original call)

        RC_INT INFO = 0;

        WORK.initialize();
        RWORK.clear();
        IWORK.clear();
        IFAIL.clear();

        WORK.initialize(2 * N, 1);
        RWORK.resize(7 * N, 0.0);
        IWORK.resize(5 * N, 0);
        IFAIL.resize(N, 0);

        std::complex<double> *APptr = reinterpret_cast<std::complex<double> *>(AP.mData.getDataPointer());
        std::complex<double> *EigVecptr = reinterpret_cast<std::complex<double> *>(eigVectors.mData.getDataPointer());
        std::complex<double> *WORKptr = reinterpret_cast<std::complex<double> *>(WORK.mData.getDataPointer());

        INFO = LAPACKE_zhpevx(LAPACK_COL_MAJOR, JOBZ, RANGE, UPLO, N, 
                reinterpret_cast<lapack_complex_double*>(APptr),
                VL, VU, IL, IU, ABSTOL,
                &M,
                eigValues.data(),
                reinterpret_cast<lapack_complex_double*>(EigVecptr),
                LDZ, IFAIL.data());

        if (INFO != 0)
        {
            std::stringstream sout;
            sout << "\nZHPEVX \nError INFO = " << INFO << "\n";
            throw std::runtime_error(sout.str());
        }

        // resize the eig values array to the number of eigenvalues found

        eigValues.resize(M);
        return M;
    }

    SCC::LapackMatrixCmplx16 AP; // For storing packed matrix in packed Hermitian form

    SCC::LapackMatrixCmplx16 WORK;
    std::vector<double> RWORK;
    std::vector<RC_INT> IWORK;
    std::vector<RC_INT> IFAIL;
};

#endif /* ZHPEVX_ */

}