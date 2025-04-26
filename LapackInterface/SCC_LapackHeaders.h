/*
 * SCC_LapackHeaders.h
 *
 * LAPACK function prototypes for LAPACK routines used by the collection of classes
 * contained in LapackInterface
 *
 *
 *  Created on: Oct 25, 2017
 *      Author: anderson
 *
 *
 *  Updated : July 27, 2018 (CRA)
 *  Updated : Dec. 09, 2023 (CRA)
 */

/*
#############################################################################
#
# Copyright  2015-2018 Chris Anderson
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

#ifndef SCC_LAPACK_HEADERS_
#define SCC_LAPACK_HEADERS_

#pragma once
#ifdef USE_MKL
extern "C" {    
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
}    
#else
#include <lapacke.h>
#include <cblas.h>
#endif

#define RC_INT int


inline CBLAS_TRANSPOSE charToCblasTranspose(char transChar)
{
    switch (transChar)
    {
        case 'N':
        case 'n': return CblasNoTrans;
        case 'T':
        case 't': return CblasTrans;
        case 'C':
        case 'c': return CblasConjTrans;
        default:
            throw std::invalid_argument("Invalid transpose character");
    }
}

inline CBLAS_LAYOUT charToCblasLayout(char layoutChar)
{
    switch (layoutChar)
    {
        case 'C':
        case 'c': return CblasColMajor;
        case 'R':
        case 'r': return CblasRowMajor;
        default:
            throw std::invalid_argument("Invalid layout character");
    }
}


#endif /* SCC_LAPACKHEADERS_H_ */


