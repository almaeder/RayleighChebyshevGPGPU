
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

    RC_abstract_matrix(const Derived& W) {}

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


	virtual void initialize(const Derived& W) = 0;

	virtual void initialize(const RCvector<T>& V, RC_INT n) = 0;

	virtual void orthogonalize() = 0;


    virtual Derived& operator=(const Derived& W) = 0;


	virtual void normalize() = 0;

	template <typename T1>
	T1 innerprod(const RC_INT k, const RC_INT l) const {};


	virtual void resize_rows(RC_INT n) = 0;

	virtual void resize_rows(RC_INT n, T value) = 0;

	virtual void resize_rows(RC_INT n, RCvector<T>& V) = 0;

	virtual void resize(RC_INT m, RC_INT n) = 0;


	size_t get_size() const
	{
	return m*n;
	}

    RC_INT get_row_size() const
    {
    return m;
    }

    RC_INT get_col_size() const
    {
    return n;
    }

    // inline T& operator()(RC_INT i, RC_INT j)
    // {
    // return mData[i  + j*m];
    // };

    // const inline T& operator()(RC_INT i, RC_INT j) const
    // {
    // return mData[i + j*m];
    // };

    // T* getDataPointer(){return mData.data();};

    // const T* getDataPointer() const {return mData.data();};

    // T* getDataPointer_d(){return mData_d;};

    // const T* getDataPointer_d() const {return mData_d;};


	// std::vector<T> mData;
	// T *mData_d = NULL;
	// cusparseDnMatDescr_t matrix_desc = NULL;

	protected:
		RC_INT n;
		RC_INT m;
		size_t data_type_size = sizeof(T);

		// currently only column major is supported
		bool column_major = true;

};



#endif /* AmatrixClassLMCuda_ */
