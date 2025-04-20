#pragma once
#include "RC_Vector.h"

#ifndef RC_ABSTRACT_OPERATOR_
#define RC_ABSTRACT_OPERATOR_

template <typename T, typename Matrix>
class RC_abstract_operator
{
    public:

	RC_abstract_operator() {};

	~RC_abstract_operator() {};

	virtual void apply(RCvector<T>& V) = 0;

	virtual void apply(Matrix& M) = 0;

    virtual void apply(Matrix& Min, Matrix& Mout, T alpha, T beta) = 0;

};

#endif /* RC_ABSTRACT_OPERATOR_ */
