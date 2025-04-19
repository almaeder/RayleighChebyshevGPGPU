
#pragma once
#include "RC_Vector.h"
#include "RC_abstract_matrix.h"

#ifndef RCMATRIX_
#define RCMATRIX_

template <typename T>
class RC_host_matrix : public RC_abstract_matrix<T, RC_host_matrix<T>>
{
	public:

//////////////////////////////////////////////////////////
//  Required constructors
//////////////////////////////////////////////////////////

	RC_host_matrix();

    RC_host_matrix(const RC_host_matrix<T>& W);

//////////////////////////////////////////////////////////
// Constructor not required for RayleighChebyshve
// but useful for creating test program
//////////////////////////////////////////////////////////

    RC_host_matrix(RC_INT m, RC_INT n);

    RC_host_matrix(RC_INT m, RC_INT n, std::vector<T> data);

	~RC_host_matrix();

//////////////////////////////////////////////////////////////////
//  Member functions required to use this class as a
//  RayleighChebyshev template parameter
//////////////////////////////////////////////////////////////////


	void host_to_device_copy();

	void device_to_host_copy();


	void initialize(const RC_host_matrix<T>& W) override;

	void initialize(const RCvector<T>& V, RC_INT n);

	void orthogonalize();

    RC_host_matrix<T>& operator=(const RC_host_matrix<T>& W) override;

	void normalize();

	T inner_product(const RC_INT k, const RC_INT l) const;

	void _scale(const RC_INT k, const T alpha);
	void _scale_add(const RC_INT k, const RC_INT l, const T alpha);


	void resize_rows(RC_INT n);


	void resize_rows(RC_INT n, T value);
	

	void resize_rows(RC_INT n, RCvector<T>& V);

	void resize(RC_INT m, RC_INT n);

	std::vector<T> mData;

};



#endif /* AmatrixClassLMCuda_ */
