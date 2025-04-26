/*
 * SCC_LapackMatrixCmplx16.h
 *
 *  Modified on: June 1, 2023
 *      Author: anderson
 *   Updated   : Dec. 9, 2023 (C.R. Anderson)
 */
//
// A matrix class to facilitate the use of LAPACK routines for COMPLEX*16 Fortran data
// types.
//
//
// The data for the matrix is stored by columns (Fortran convention) with each complex matrix
// element value stored as alternating doubles containing the real and imaginary part of that
// value.
//
// It is assumed that the data storage for std::complex<double> contains the
// real and imaginary components in adjacent memory location with the real component
// first. In addition it is assumed that the data for a std::vector<std::complex<double>
// is stored in contiguous memory locations with double values for the real and imaginary
// and compoments alternating, e.g. the storage convention used by FORTRAN for complex*16.
//
// Internally the data storage uses an SCC::LapackMatrix to facilitate the implementation
// of algebraic operations.
//
// Lapack routine dependencies : zgemm and zgemv
/*
#############################################################################
#
# Copyright 2021- Chris Anderson
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

#include "SCC_LapackHeaders.h"
#include "SCC_LapackMatrix.h"

#include <complex>
#include <cassert>

#ifndef LAPACK_MATRIX_CMPLX_16_H_
#define LAPACK_MATRIX_CMPLX_16_H_

namespace SCC
{
class LapackMatrixCmplx16
{
public:

    LapackMatrixCmplx16()
	{
    	this->rows = 0;
    	this->cols = 0;
	}

    LapackMatrixCmplx16(const LapackMatrixCmplx16& C)
    {
    	this->rows  = C.rows;
    	this->cols  = C.cols;
    	this->mData = C.mData;
    }

	LapackMatrixCmplx16(RC_INT M, RC_INT N)
	{
		initialize(M,N);
	}

	LapackMatrixCmplx16(const SCC::LapackMatrix& realA, const SCC::LapackMatrix& imagA)
	{
		initialize(realA,imagA);
	}

    void initialize()
	{
	    this->rows = 0;
		this->cols = 0;
		mData.initialize();
	}

	void initialize(RC_INT M, RC_INT N)
	{
	    this->rows = M;
		this->cols = N;
		mData.initialize(2*rows,cols);
		mData.setToValue(0.0);
	}

	void initialize(const LapackMatrixCmplx16& C)
    {
    	this->rows  = C.rows;
    	this->cols  = C.cols;
    	this->mData.initialize(C.mData);
    }

	void initialize(const SCC::LapackMatrix& realA, const SCC::LapackMatrix& imagA)
	{
	    this->rows = realA.getRowDimension();
		this->cols = realA.getColDimension();
		if((realA.getRowDimension() != imagA.getRowDimension())
		 ||(realA.getColDimension() != imagA.getColDimension()))
		 {
			throw std::runtime_error("\nIncorrect dimension input matrices in \nLapackMatrixCmplx (realA,imagA) constructor.\n");
		 }

		mData.initialize(2*rows,cols);

		for(RC_INT j = 0; j < cols; j++)
		{
		for(RC_INT i = 0; i < rows; i++)
		{
		mData(2*i,  j) = realA(i,j);
		mData(2*i+1,j) = imagA(i,j);
		}}
	}

	void setToValue(double val)
	{
	    for(RC_INT j = 0; j < cols; j++)
		{
		for(RC_INT i = 0; i < rows; i++)
		{
		mData(2*i,  j) = val;
		mData(2*i+1,j) = 0.0;
		}}
	}

    void setToValue(const std::complex<double>& val)
	{
	    for(RC_INT j = 0; j < cols; j++)
		{
		for(RC_INT i = 0; i < rows; i++)
		{
		mData(2*i,  j) = val.real();
		mData(2*i+1,j) = val.imag();
		}}
	}


	RC_INT getRowDimension() const {return rows;}
	RC_INT getColDimension() const {return cols;}

	inline void insert(RC_INT i, RC_INT j, double vReal, double vCplx)
	{
		 mData(2*i,j)      = vReal;
		 mData(2*i + 1,j)  = vCplx;
	}

	inline void extract(RC_INT i, RC_INT j, double& vReal, double& vCplx) const
	{
	     vReal = mData(2*i,j);
		 vCplx = mData(2*i + 1,j);
	}

    inline void insert(RC_INT i, RC_INT j, std::complex<double> z)
	{
		 mData(2*i,j)      = z.real();
		 mData(2*i + 1,j)  = z.imag();
	}

	inline void extract(RC_INT i, RC_INT j, std::complex<double>& z) const
	{
	     z = std::complex<double>(mData(2*i,j),mData(2*i + 1,j));
	}


/*!  Outputs the matrix with limited number of digits with the (0,0) element in the upper left corner */

	void printDense(std::ostream& outStream, int precision = 3)
	{
	    std::ios_base::fmtflags ff = outStream.flags();
	    int precisionCache = outStream.precision(precision);
	    std::complex<double> val;

        for(RC_INT i = 0;  i < rows; i++)
        {
        for(RC_INT j = 0; j <  cols; j++)
        {
          val = this->operator()(i,j);
          outStream <<   std::scientific <<  std::showpos << std::right << std::setw(precision+18) << val << " ";
        }
        outStream << std::endl;
        }
        outStream.flags(ff);
        outStream.precision(precisionCache);
	}

	/*!  Outputs the matrix values to the screen with the (0,0) element in the upper left corner  */

	friend std::ostream& operator<<(std::ostream& outStream, const LapackMatrixCmplx16&  V)
	{
        RC_INT i; RC_INT j;

        std::ios_base::fmtflags ff = outStream.flags();

        for(i = 0;  i < V.rows; i++)
        {
        for(j = 0; j <  V.cols; j++)
        {
          outStream <<   std::scientific << std::setprecision(15) << std::showpos << std::right << V(i,j) << " ";
        }
        outStream << std::endl;
        }

        outStream.flags(ff);
        return outStream;
	}



    /*!
    Returns a reference to the element with index (i,j) - indexing
    starting at (0,0). Using the fact that the pointer to a complex<double> value
    is a pointer to the first of two consecutive doubles storing the
    complex value.
    */

	#ifdef _DEBUG
    std::complex<double>&  operator()(RC_INT i, RC_INT j)
    {
    assert(boundsCheck(i, 0, rows-1,1));
    assert(boundsCheck(j, 0, cols-1,2));
    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i) + j*(2*rows))));
    };

    const std::complex<double>&  operator()(RC_INT i, RC_INT j) const
    {
    assert(boundsCheck(i, 0, rows-1,1));
    assert(boundsCheck(j, 0, cols-1,2));
    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i) + j*(2*rows))));
    };
#else
    /*!
    Returns a reference to the element with index (i,j) - indexing
    starting at (0,0). Using the fact that the pointer to a complex<double> value
    is a pointer to the first of two consecutive doubles storing the
    complex value.
    */
    inline std::complex<double>&  operator()(RC_INT i, RC_INT j)
    {
    	return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i) + j*(2*rows))));
    };

    inline const std::complex<double>&  operator()(RC_INT i, RC_INT j) const
    {
    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i) + j*(2*rows))));;
    };
#endif


//
// Convenience access for single column or row matrices, e.g.
// matrices initialized with (N,1) or (1,N).
//
// Indexing starts at 0;
//
//
#ifdef _DEBUG
    std::complex<double>&  operator()(RC_INT i)
    {
    assert(singleRowOrColCheck());

    RC_INT i1 = i;
    RC_INT i2 = i;
    if     (cols == 1) {i2 = 0;}
    else if(rows == 1) {i1 = 0;}

    assert(boundsCheck(i1, 0, rows-1,1));
    assert(boundsCheck(i2, 0, cols-1,2));

    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i1) + i2*(2*rows))));
    };

    const std::complex<double>&  operator()(RC_INT i) const
    {
    assert(singleRowOrColCheck());
    RC_INT i1 = i;
    RC_INT i2 = i;
    if     (cols == 1) {i2 = 0;}
    else if(rows == 1) {i1 = 0;}

    assert(boundsCheck(i1, 0, rows-1,1));
    assert(boundsCheck(i2, 0, cols-1,2));
    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i1) + i2*(2*rows))));
    };
#else

    /*!
    Returns a reference to the element with index (i) in a LapackMatrixCmplx16
    with a single row or column.
    Indexing starting at (0)
    */
    inline std::complex<double>&  operator()(RC_INT i)
    {
    RC_INT i1 = i;
    RC_INT i2 = i;
    if     (cols == 1) {i2 = 0;}
    else if(rows == 1) {i1 = 0;}

    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i1) + i2*(2*rows))));
    };

    /*!
    Returns a reference to the element with index (i) in a LapackMatrixCmplx16
    with a single row or column.
    Indexing starting at (0)
     */
    inline const std::complex<double>&  operator()(RC_INT i) const
    {

    RC_INT i1 = i;
    RC_INT i2 = i;
    if     (cols == 1) {i2 = 0;}
    else if(rows == 1) {i1 = 0;}

    return *(reinterpret_cast<std::complex<double>*>((mData.dataPtr +  (2*i1) + i2*(2*rows))));
    };


#endif



    double normFrobenius() const
    {
	double valSum = 0.0;

	for(RC_INT j = 0; j < cols; j++)
	{
	for(RC_INT i = 0; i < rows; i++)
	{
    		valSum += std::norm(this->operator()(i,j));
    }}
    return std::sqrt(valSum);
    }

    void getColumn(RC_INT colIndex, std::vector< std::complex<double>> & Mcol)
    {
    	Mcol.resize(rows);
    	for(RC_INT i = 0; i < rows; i++)
    	{
		Mcol[i] = this->operator()(i,colIndex);
    	}
    }

    void getColumn(RC_INT colIndex, LapackMatrixCmplx16 & Mcol)
    {
    	Mcol.initialize(rows,1);
    	for(RC_INT i = 0; i < rows; i++)
    	{
		Mcol(i,0) = this->operator()(i,colIndex);
    	}
    }

	void getRealAndCmplxMatrix(LapackMatrix& realA, LapackMatrix& imagA) const
	{
		realA.initialize(rows,cols);
		imagA.initialize(rows,cols);

	    for(RC_INT j = 0; j < cols; j++)
		{
		for(RC_INT i = 0; i < rows; i++)
		{
		realA(i,j) = mData(2*i,j);
		imagA(i,j) = mData(2*i+1,j);
		}}
	}

	void getRealAndCmplxColumn(RC_INT colIndex, std::vector<double>& realCol, std::vector<double>& imagCol)
	{
		assert(boundsCheck(colIndex, 0, cols-1,2));
		realCol.resize(rows);
		imagCol.resize(rows);

	    for(RC_INT i = 0; i < rows; i++)
		{
		realCol[i] = mData(2*i,colIndex);
		imagCol[i] = mData(2*i+1,colIndex);
		}
	}

    void getRealAndCmplxColumn(RC_INT colIndex, LapackMatrix& realCol, LapackMatrix& imagCol)
	{
		assert(boundsCheck(colIndex, 0, cols-1,2));

	    realCol.initialize(rows,1);
		imagCol.initialize(rows,1);


	    for(RC_INT i = 0; i < rows; i++)
		{
		realCol(i,0) = mData(2*i,colIndex);
		imagCol(i,0) = mData(2*i+1,colIndex);
		}
	}


	LapackMatrixCmplx16 createUpperTriPacked()
	{
		if(rows != cols)
		{
			throw std::runtime_error("\nLapackMatrixCmplx16: No conversion of non-square matrix \nto upper traingular packed form.\n");
		}

		LapackMatrixCmplx16 AP((rows*(rows+1))/2,1);

		RC_INT     ind; RC_INT     jnd;
		double vReal; double vImag;

		for(RC_INT j = 1; j <=cols; j++)
		{
		for(RC_INT i = 1; i <= j;   i++)
		{
            ind = i-1;
            jnd = j-1;
            extract(ind,jnd,vReal,vImag);

            ind = (i + (j-1)*j/2) - 1;
            AP.insert(ind,  0,vReal,vImag);
		}}


		return AP;
	}

//  Algebraic operators

    inline void operator=(const LapackMatrixCmplx16& B)
	{
    	if(mData.dataPtr == nullptr)
    	{
    		rows    = B.rows;
    		cols    = B.cols;
    		mData.initialize(B.mData);
    	}

        assert(sizeCheck(this->rows,B.rows));
    	assert(sizeCheck(this->cols,B.cols));
    	mData = B.mData;
	}


    inline void operator+=(const  LapackMatrixCmplx16& B)
    {
    	assert(sizeCheck(this->rows,B.rows));
    	assert(sizeCheck(this->cols,B.cols));
    	mData += B.mData;
    }

    LapackMatrixCmplx16 operator+(const LapackMatrixCmplx16& B)
    {
    	assert(sizeCheck(this->rows,B.rows));
    	assert(sizeCheck(this->cols,B.cols));

    	LapackMatrixCmplx16  C(*this);

    	C.mData += B.mData;
        return C;
    }

    LapackMatrixCmplx16 operator-(const LapackMatrixCmplx16& B)
    {
    	assert(sizeCheck(this->rows,B.rows));
    	assert(sizeCheck(this->cols,B.cols));

    	LapackMatrixCmplx16  C(*this);

    	C.mData -= B.mData;
    	return C;
    }

    inline void operator-=(const  LapackMatrixCmplx16& D)
    {
      assert(sizeCheck(this->rows,D.rows));
      assert(sizeCheck(this->cols,D.cols));

      mData -= D.mData;
    }

    inline void operator*=(const double alpha)
    {
    		mData *= alpha;
    }

    inline void operator*=(const std::complex<double> alpha)
    {
            double cReal; double cImag;
            double aReal = alpha.real();
            double aImag = alpha.imag();

            for(RC_INT i = 0; i < rows; i++)
            {
            for(RC_INT j = 0; j < cols; j++)
            {
            cReal          = mData(2*i,j);
		    cImag          = mData(2*i+1,j);
		    mData(2*i,j)   = cReal*aReal - cImag*aImag;
		    mData(2*i+1,j) = cReal*aImag + cImag*aReal;
            }}
    }

    LapackMatrixCmplx16 operator*(const double alpha)
    {
    LapackMatrixCmplx16 R(*this);
    R *= alpha;
    return R;
    }

    LapackMatrixCmplx16 operator*(const std::complex<double> alpha)
    {
    LapackMatrixCmplx16 R(*this);
    R *= alpha;
    return R;
    }


    friend LapackMatrixCmplx16 operator*(const double alpha, const LapackMatrixCmplx16& B)
    {
    LapackMatrixCmplx16 R(B);
    R *= alpha;
    return R;
    }

    friend LapackMatrixCmplx16 operator*(const std::complex<double> alpha, const LapackMatrixCmplx16& B)
    {
    LapackMatrixCmplx16 R(B);
    R *= alpha;
    return R;
    }


    inline void operator/=(const double alpha)
    {
    		mData /= alpha;
    }

    inline void operator/=(const std::complex<double> alpha)
    {
            double cReal; double cImag;
            std::complex<double> alphaInv = 1.0/alpha;

            double aReal = alphaInv.real();
            double aImag = alphaInv.imag();

            for(RC_INT i = 0; i < rows; i++)
            {
            for(RC_INT j = 0; j < cols; j++)
            {
            cReal          = mData(2*i,j);
		    cImag          = mData(2*i+1,j);
		    mData(2*i,j)   = cReal*aReal - cImag*aImag;
		    mData(2*i+1,j) = cReal*aImag + cImag*aReal;
            }}
    }


    LapackMatrixCmplx16 operator/(const double alpha)
    {
    LapackMatrixCmplx16 R(*this);
    R /= alpha;
    return R;
    }

    LapackMatrixCmplx16 operator/(const std::complex<double> alpha)
    {
    LapackMatrixCmplx16 R(*this);
    R /= alpha;
    return R;
    }


    bool isNull() const
    {
    if((rows == 0)||(cols == 0)) { return true;}
    return false;
    }


//  C := alpha*op( A )*op( B ) + beta*C,

LapackMatrixCmplx16 operator*(const LapackMatrixCmplx16& B) const
{
    assert(sizeCheck(this->cols,B.rows));

    LapackMatrixCmplx16 C(this->rows,B.cols);

    char TRANSA = 'N';
    char TRANSB = 'N';

    RC_INT M       = this->rows;
    RC_INT N       = B.cols;
    RC_INT K       = this->cols;

    std::complex<double> ALPHA = {1.0,0.0};
    std::complex<double> BETA  = {0.0,0.0};

    std::complex<double> *Aptr  = reinterpret_cast<std::complex<double>*>(mData.getDataPointer());
    std::complex<double> *Bptr  = reinterpret_cast<std::complex<double>*>(B.mData.getDataPointer());
    std::complex<double> *Cptr  = reinterpret_cast<std::complex<double>*>(C.mData.getDataPointer());
    RC_INT LDA     = this->rows;
    RC_INT LDB     = B.rows;
    RC_INT LDC     = C.rows;

    cblas_zgemm(CblasColMajor, charToCblasTranspose(TRANSA), charToCblasTranspose(TRANSB),
        M,N,K, &ALPHA,  Aptr, LDA, Bptr,LDB,  &BETA,  Cptr,LDC);
    return C;
}


std::vector< std::complex<double> > operator*(const std::vector< std::complex<double> >& x)
{
	std::vector< std::complex<double> > y(rows,0.0);

    char TRANS     = 'N';
    std::complex<double> ALPHA = {1.0,0.0};
    std::complex<double> BETA  = {0.0,0.0};

    std::complex<double> *Aptr  = reinterpret_cast<std::complex<double>*>(mData.getDataPointer());
    std::complex<double> *Cptr  = reinterpret_cast<std::complex<double>*>(y.data());
    RC_INT INCX      = 1;
    RC_INT INCY      = 1;

    cblas_zgemv(CblasColMajor, charToCblasTranspose(TRANS), rows,cols,  &ALPHA,  Aptr, rows,
          &x[0], INCX,  &BETA,  Cptr, INCY);
	return y;
}


LapackMatrixCmplx16 conjugateTranspose() const
{
	LapackMatrixCmplx16 R(cols,rows);
	for(RC_INT i = 0; i < rows; i++)
	{
		for(RC_INT j = 0; j < cols; j++)
		{
			R(j,i) = {this->operator()(i,j).real(), -this->operator()(i,j).imag()};
		}
	}
	return R;
}


#ifdef _DEBUG
        bool boundsCheck(RC_INT i, RC_INT begin, RC_INT end,int coordinate) const
        {
        if((i < begin)||(i  > end))
        {
        std::cerr << "LapackMatrix index " << coordinate << " out of bounds " << std::endl;
        std::cerr << "Offending index value : " << i << " Acceptable Range [" << begin << "," << end << "] " << std::endl;
        return false;
        }
        return true;
        }
#else
        bool boundsCheck(RC_INT, RC_INT, RC_INT,int) const {return true;}
#endif



#ifdef _DEBUG
        bool singleRowOrColCheck() const
        {
        if((rows != 1)&&(cols != 1))
        {
        std::cerr << "LapackMatrixCmplx16 Error: Use of single index access"  << std::endl;
        std::cerr << "for LapackMatrixCmplx that is not a single row or column" << std::endl;
        return false;
        }
        return true;
        }
#else
        bool singleRowOrColCheck() const {return true;}
#endif

#ifdef _DEBUG
    bool sizeCheck(RC_INT size1, RC_INT size2)
    {
    if(size1 != size2)
    {
    std::cerr << "LapackMatrixCmplx16 sizes are incompatible : " << size1 << " != " << size2 << " ";
    return false;
    }
    return true;
    }

    bool sizeCheck(RC_INT size1, RC_INT size2) const
    {
    if(size1 != size2)
    {
    std::cerr << "LapackMatrixCmplx16 sizes are incompatible : " << size1 << " != " << size2 << " ";
    return false;
    }
    return true;
    }
#else
    bool sizeCheck(RC_INT, RC_INT) {return true;}
    bool sizeCheck(RC_INT, RC_INT) const{return true;}
#endif


	RC_INT rows;
	RC_INT cols;
	SCC::LapackMatrix mData;

};
};

// LAPACK documentation

////////////////////////////////////////////////////////////////
// ZGEMV
////////////////////////////////////////////////////////////////
/*
zgemv()
subroutine zgemv	(	character 	trans,
integer 	m,
integer 	n,
complex*16 	alpha,
complex*16, dimension(lda,*) 	a,
integer 	lda,
complex*16, dimension(*) 	x,
integer 	incx,
complex*16 	beta,
complex*16, dimension(*) 	y,
integer 	incy
)

Purpose:
 ZGEMV  performs one of the matrix-vector operations

    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or

    y := alpha*A**H*x + beta*y,

 where alpha and beta are scalars, x and y are vectors and A is an
 m by n matrix.
Parameters
[in]	TRANS
          TRANS is CHARACTER*1
           On entry, TRANS specifies the operation to be performed as
           follows:

              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.

              TRANS = 'T' or 't'   y := alpha*A**T*x + beta*y.

              TRANS = 'C' or 'c'   y := alpha*A**H*x + beta*y.
[in]	M
          M is INTEGER
           On entry, M specifies the number of rows of the matrix A.
           M must be at least zero.
[in]	N
          N is INTEGER
           On entry, N specifies the number of columns of the matrix A.
           N must be at least zero.
[in]	ALPHA
          ALPHA is COMPLEX*16
           On entry, ALPHA specifies the scalar alpha.
[in]	A
          A is COMPLEX*16 array, dimension ( LDA, N )
           Before entry, the leading m by n part of the array A must
           contain the matrix of coefficients.
[in]	LDA
          LDA is INTEGER
           On entry, LDA specifies the first dimension of A as declared
           in the calling (sub) program. LDA must be at least
           max( 1, m ).
[in]	X
          X is COMPLEX*16 array, dimension at least
           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
           and at least
           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
           Before entry, the incremented array X must contain the
           vector x.
[in]	INCX
          INCX is INTEGER
           On entry, INCX specifies the increment for the elements of
           X. INCX must not be zero.
[in]	BETA
          BETA is COMPLEX*16
           On entry, BETA specifies the scalar beta. When BETA is
           supplied as zero then Y need not be set on input.
[in,out]	Y
          Y is COMPLEX*16 array, dimension at least
           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
           and at least
           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
           Before entry with BETA non-zero, the incremented array Y
           must contain the vector y. On exit, Y is overwritten by the
           updated vector y.
           If either m or n is zero, then Y not referenced and the function
           performs a quick return.
[in]	INCY
          INCY is INTEGER
           On entry, INCY specifies the increment for the elements of
           Y. INCY must not be zero.
Author
Univ. of Tennessee
Univ. of California Berkeley
Univ. of Colorado Denver
NAG Ltd.
*/

////////////////////////////////////////////////////////////////
// ZGEMM
////////////////////////////////////////////////////////////////
/*
zgemm()
subroutine zgemm	(	character 	transa,
character 	transb,
integer 	m,
integer 	n,
integer 	k,
complex*16 	alpha,
complex*16, dimension(lda,*) 	a,
integer 	lda,
complex*16, dimension(ldb,*) 	b,
integer 	ldb,
complex*16 	beta,
complex*16, dimension(ldc,*) 	c,
integer 	ldc
)

Purpose:
 ZGEMM  performs one of the matrix-matrix operations

    C := alpha*op( A )*op( B ) + beta*C,

 where  op( X ) is one of

    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,

 alpha and beta are scalars, and A, B and C are matrices, with op( A )
 an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
Parameters
[in]	TRANSA
          TRANSA is CHARACTER*1
           On entry, TRANSA specifies the form of op( A ) to be used in
           the matrix multiplication as follows:

              TRANSA = 'N' or 'n',  op( A ) = A.

              TRANSA = 'T' or 't',  op( A ) = A**T.

              TRANSA = 'C' or 'c',  op( A ) = A**H.
[in]	TRANSB
          TRANSB is CHARACTER*1
           On entry, TRANSB specifies the form of op( B ) to be used in
           the matrix multiplication as follows:

              TRANSB = 'N' or 'n',  op( B ) = B.

              TRANSB = 'T' or 't',  op( B ) = B**T.

              TRANSB = 'C' or 'c',  op( B ) = B**H.
[in]	M
          M is INTEGER
           On entry,  M  specifies  the number  of rows  of the  matrix
           op( A )  and of the  matrix  C.  M  must  be at least  zero.
[in]	N
          N is INTEGER
           On entry,  N  specifies the number  of columns of the matrix
           op( B ) and the number of columns of the matrix C. N must be
           at least zero.
[in]	K
          K is INTEGER
           On entry,  K  specifies  the number of columns of the matrix
           op( A ) and the number of rows of the matrix op( B ). K must
           be at least  zero.
[in]	ALPHA
          ALPHA is COMPLEX*16
           On entry, ALPHA specifies the scalar alpha.
[in]	A
          A is COMPLEX*16 array, dimension ( LDA, ka ), where ka is
           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
           part of the array  A  must contain the matrix  A,  otherwise
           the leading  k by m  part of the array  A  must contain  the
           matrix A.
[in]	LDA
          LDA is INTEGER
           On entry, LDA specifies the first dimension of A as declared
           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
           LDA must be at least  max( 1, m ), otherwise  LDA must be at
           least  max( 1, k ).
[in]	B
          B is COMPLEX*16 array, dimension ( LDB, kb ), where kb is
           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
           part of the array  B  must contain the matrix  B,  otherwise
           the leading  n by k  part of the array  B  must contain  the
           matrix B.
[in]	LDB
          LDB is INTEGER
           On entry, LDB specifies the first dimension of B as declared
           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
           LDB must be at least  max( 1, k ), otherwise  LDB must be at
           least  max( 1, n ).
[in]	BETA
          BETA is COMPLEX*16
           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
           supplied as zero then C need not be set on input.
[in,out]	C
          C is COMPLEX*16 array, dimension ( LDC, N )
           Before entry, the leading  m by n  part of the array  C must
           contain the matrix  C,  except when  beta  is zero, in which
           case C need not be set on entry.
           On exit, the array  C  is overwritten by the  m by n  matrix
           ( alpha*op( A )*op( B ) + beta*C ).
[in]	LDC
          LDC is INTEGER
           On entry, LDC specifies the first dimension of C as declared
           in  the  calling  (sub)  program.   LDC  must  be  at  least
           max( 1, m ).
Author
Univ. of Tennessee
Univ. of California Berkeley
Univ. of Colorado Denver
NAG Ltd.
 */
#endif /* LAPACK_MATRIX_CMPLX_16_H__ */
