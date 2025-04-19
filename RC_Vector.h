/*
 * RC_Vector.h
 *
 * A wrapper class for std::vector<T> that adds the minimal number of
 * vector operations required to instantiate instances of the
 * RayleighChebyshev class.
 *
 *
 * Note : Individual vector element access is not needed by the RayleighChebyshev
 * procedure; the eigensystem procedure implemented in the RayleighChebyshev
 * procedure is therefore agnostic to any particular representation of the data
 * within the vector.
 *
 *
 *  Created on: Oct 11, 2024
 *      Author: anderson
 */
//#############################################################################
//#
//# Copyright  2024 Chris Anderson
//#
//# This program is free software: you can redistribute it and/or modify
//# it under the terms of the Lesser GNU General Public License as published by
//# the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# For a copy of the GNU General Public License see
//# <http://www.gnu.org/licenses/>.
//#
//#############################################################################
#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <complex>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <random>
#include <iostream>
#include "RC_Types.h"

#ifndef RCVECTOR_
#define RCVECTOR_

template <typename T>
class RCvector
{
	public:

//////////////////////////////////////////////////////////
//  Required constructors
//////////////////////////////////////////////////////////

	RCvector()
	{
	    vData.clear();
	}

    RCvector(const RCvector& W)
    {
        initialize(W);
    }

//////////////////////////////////////////////////////////
// Constructor not required for RayleighChebyshve
// but useful for creating test program
//////////////////////////////////////////////////////////

    RCvector(RC_INT dimension)
    {
        vData.resize(dimension);
    }


	virtual ~RCvector(){}

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////

	void initialize(const RCvector<T>& W)
	{
	    vData = W.vData;
	}

	T dot(const RCvector<T>& W) const
	{
		T dotSum = 0.0;
		for(size_t k = 0; k < vData.size(); k++)
		{
			if constexpr (std::is_same<T, double>::value)
				dotSum += vData[k]*W.vData[k];
			else if constexpr (std::is_same<T, std::complex<double>>::value)
			{
				dotSum += vData[k]*std::conj(W.vData[k]);				
			}

		}
		return dotSum;
	}

	void operator *=(T alpha)
	{
	    for(size_t k = 0; k < vData.size(); k++)
	    {
	       vData[k] *= alpha;
	    }
	}

    void operator +=(const RCvector<T>& W)
	{
	    for(size_t k = 0; k < vData.size(); k++)
	    {
	        vData[k] += W.vData[k];
	    }
	}

	void operator -=(const RCvector<T>& W)
	{
	    for(size_t k = 0; k < vData.size(); k++)
	    {
	        vData[k] -= W.vData[k];
	    }
	}


	T operator [](RC_INT k) const
	{
		return vData[k];
	}

	T norm2() const
	{
		T normSquared =  (*this).dot(*this);
		return std::sqrt(std::abs(normSquared));
	}

	size_t get_size() const
	{
	return vData.size();
	}

	std::vector<T> vData;
};


#endif /* AVECTORCLASS_ */
