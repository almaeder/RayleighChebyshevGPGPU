#pragma once
#include "../../RC_host_matrix.h"

using CPX = std::complex<double>;

#ifndef DIAG_HOST_OPERATOR_
#define DIAG_HOST_OPERATOR_

template <typename T, typename Matrix>
class diag_host_operator
{
    public:
    std::vector<T> data;

	diag_host_operator();

    diag_host_operator(RC_INT n);

	~diag_host_operator();

	virtual void apply(RCvector<T>& V);

	virtual void apply(Matrix& M);

    virtual void apply(Matrix& Min, Matrix& Mout, T alpha, T beta);

};

#endif /* DIAG_HOST_OPERATOR_ */
