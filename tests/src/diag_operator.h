#pragma once
#include "../../RC_abstract_operator.h"
#include "../../RC_host_matrix.cpp"

using CPX = std::complex<double>;

#ifndef DIAG_OPERATOR_
#define DIAG_OPERATOR_

template <typename T, typename Matrix>
class diag_operator
{
    public:
    std::vector<T> data;

	diag_operator();

    diag_operator(RC_INT n);

	~diag_operator();

	virtual void apply(RCvector<T>& V);

	virtual void apply(Matrix& M);

    virtual void apply(Matrix& Min, Matrix& Mout, T alpha, T beta);

};

#endif /* DIAG_OPERATOR_ */
