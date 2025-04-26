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

#ifndef DSYEV_
#define DSYEV_

namespace SCC
{
class DSYEV
{
public:

    DSYEV()
    {
    }

    void initialize()
    {
        U.initialize();
        eigValues.clear();
    }

    void computeEigensystem(const SCC::LapackMatrix& A, std::vector<double>& eigenValues, std::vector < std::vector < double> >& eigenVectors)
    {
        assert(A.sizeCheck(A.rows,A.cols));
        computeEigensystem(A,eigenValues, U);

        // Pack eigenvectors into return argument

        std::vector <double> eigVector(A.rows);

        eigenVectors.clear();
        eigenVectors.resize(A.cols,eigVector);

        for(RC_INT j = 0; j < A.cols; j++)
        {
            for(RC_INT i = 0; i < A.rows; i++)
            {
                eigenVectors[j][i] = U(i,j);
            }
        }

    }

   
    void computeEigensystem(const SCC::LapackMatrix& A, std::vector<double>& eigenValues, SCC::LapackMatrix& eigenVectors)
    {
        assert(A.sizeCheck(A.rows,A.cols));

        eigenVectors.initialize(A);
        char JOBZ = 'V';
        char UPLO = 'U';            // Using upper triangular part of A

        RC_INT N       = A.rows;
        double* Uptr = eigenVectors.dataPtr;

        RC_INT LDA = N;

        eigenValues.resize(N);
        double*Wptr = &eigenValues[0];

        RC_INT INFO = 0;

        // Second call to create eigensystem
        INFO = LAPACKE_dsyev(LAPACK_COL_MAJOR, JOBZ, UPLO, N, Uptr, LDA, Wptr);

        if(INFO != 0)
        {
        std::cerr << "dsyev  Failed : INFO = " << INFO  << std::endl;
        exit(1);
        }
    }

    SCC::LapackMatrix                 U;
    std::vector<double>  eigValues;

};

#endif /* DSYEV_ */

}