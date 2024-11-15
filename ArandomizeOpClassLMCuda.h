/*
 * ArandomizeOpClassLMCuda.h
 *
 *  Created on: Oct 11, 2024
 *      Author: anderson
 *
 * An operator class that provides the member function randomize(...) required
 * by the RayleighChebyshev procedure to initialize instances of the vector
 * class being used with random entries.
 *
 * This class uses the C++ random number generator std::mt19937_64 and currently
 * has a fixed seed.
 *
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
#include <random>
#include "AvectorClassLMCuda.h"
#include "AmatrixClassLMCuda.h"
#include <complex>

#ifndef ARANDOMIZEOPCLASSLMCuda_
#define ARANDOMIZEOPCLASSLMCuda_

template <typename T>
class ArandomizeOpClassLMCuda
{
    public:

	ArandomizeOpClassLMCuda()
	{
	    seed = 3141592;
	    randomGenerator.seed(seed);

	    // Initialize the distribution to be uniform in the interval [-1,1]

	    std::uniform_real_distribution<double>::param_type distParams(-1.0,1.0);
	    distribution.param(distParams);
	}

	void randomize(AvectorClassLMCuda<T>& V)
	{
        for(size_t i = 0; i < V.getDimension(); i++)
        {
			if constexpr (std::is_same<T, double>::value)
			{
				V.vData[i] = distribution(randomGenerator);
			}
			else if constexpr (std::is_same<T, std::complex<double>>::value)
			{
				std::complex<double> random_complex(distribution(randomGenerator), distribution(randomGenerator));
				V.vData[i] = random_complex;
			}
		}
	}

	void randomize(AmatrixClassLMCuda<T>& M)
	{
        for(size_t i = 0; i < M.getDimension(); i++)
        {
			if constexpr (std::is_same<T, double>::value)
			{
				M.mData[i] = distribution(randomGenerator);
			}
			else if constexpr (std::is_same<T, std::complex<double>>::value)
			{
				std::complex<double> random_complex(distribution(randomGenerator), distribution(randomGenerator));
				M.mData[i] = random_complex;
			}
        }


	}



    int                                    seed;
    std::mt19937_64                        randomGenerator;
	std::uniform_real_distribution<double>      distribution;
};




#endif /* ArandomizeOpClassLMCuda */
