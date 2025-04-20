
#pragma once
#include "RC_Vector.h"

#ifndef RC_ABSTRACT_MATRIX_
#define RC_ABSTRACT_MATRIX_

template <typename T, typename Derived>
class RC_abstract_matrix
{
public:
	//////////////////////////////////////////////////////////
	//  Required constructors
	//////////////////////////////////////////////////////////

	RC_abstract_matrix() {};

	RC_abstract_matrix(const Derived &W) {}

	//////////////////////////////////////////////////////////
	// Constructor not required for RayleighChebyshve
	// but useful for creating test program
	//////////////////////////////////////////////////////////

	RC_abstract_matrix(RC_INT m, RC_INT n) {}

	RC_abstract_matrix(RC_INT m, RC_INT n, std::vector<T> data) {}

	virtual ~RC_abstract_matrix() {}

	//////////////////////////////////////////////////////////////////
	//  Member functions required to use this class as a
	//  RayleighChebyshev template parameter
	//////////////////////////////////////////////////////////////////

	virtual void host_to_device_copy() = 0;

	virtual void device_to_host_copy() = 0;

	virtual void device_to_device_copy(const Derived &W) = 0;

	virtual void initialize(const Derived &W) = 0;

	virtual void initialize(const RCvector<T> &V, RC_INT n) = 0;

	virtual void orthogonalize() = 0;

	virtual Derived &operator=(const Derived &W) = 0;

	virtual inline T &operator()(RC_INT i, RC_INT j) = 0;

	virtual const inline T &operator()(RC_INT i, RC_INT j) const = 0;

	virtual void normalize() = 0;

	T inner_product(const RC_INT k, const RC_INT l) const {};

	virtual void resize_rows(RC_INT n) = 0;

	virtual void resize_rows(RC_INT n, T value) = 0;

	virtual void resize_rows(RC_INT n, RCvector<T> &V) = 0;

	virtual void resize(RC_INT m, RC_INT n) = 0;

	virtual void matmult(const Derived &A, const Derived &B, T alpha, T beta) = 0;

	virtual void matmult(const Derived &A, const Derived &B, T alpha, T beta, std::string conj_A, std::string conj_B) = 0;

	virtual void residuals(const Derived &OpA, const std::vector<double> &eig_values, std::vector<double> &eig_residuals, RC_INT residualCheckCount) = 0;

	virtual void substract(const Derived &A) = 0;
	
	virtual void scale(const T alpha) = 0;

	size_t get_size() const
	{
		return m * n;
	}

	RC_INT get_row_size() const
	{
		return m;
	}

	RC_INT get_col_size() const
	{
		return n;
	}

protected:
	RC_INT n;
	RC_INT m;
	size_t data_type_size = sizeof(T);

	// currently only column major is supported
	bool column_major = true;
};

#endif /* RC_ABSTRACT_MATRIX_ */

#ifndef RC_ABSTRACT_RANDOMIZE_
#define RC_ABSTRACT_RANDOMIZE_

template <typename T, typename Matrix>
class RC_abstract_randomize
{
public:
	RC_abstract_randomize() {}

	virtual void randomize(RCvector<T> &V) = 0;

	virtual void randomize(Matrix &M) = 0;
};

#endif /* RC_abstract_randomize */
