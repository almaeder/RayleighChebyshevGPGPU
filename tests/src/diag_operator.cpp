#include "diag_operator.h"

template <typename T, typename Matrix>
diag_operator<T, Matrix>::diag_operator() {}

template <typename T, typename Matrix>
diag_operator<T, Matrix>::diag_operator(RC_INT n) {

    data.resize(n);
    for (RC_INT i = 0; i < n; i++) {
        if constexpr (std::is_same<T, std::complex<double>>::value) {
            data[i] = std::complex<double>(i + 1, 0);
        } else if constexpr (std::is_same<T, double>::value) {
            data[i] = static_cast<T>(i + 1);
        } 
        else
            throw std::runtime_error("Unsupported type");
    }
}

template <typename T, typename Matrix>
diag_operator<T, Matrix>::~diag_operator() {}

template <>
void diag_operator<CPX, RC_host_matrix<CPX>>::apply(RCvector<CPX>& V) {
    for (RC_INT i = 0; i < V.get_size(); i++) {
        V.vData[i] = data[i] * V.vData[i];
    }
}

template <>
void diag_operator<CPX, RC_host_matrix<CPX>>::apply(RC_host_matrix<CPX>& M) {

    // assert that M.get_row_size() == data.size()
    if (M.get_row_size() != data.size()) {
        std::cout << "M.get_row_size() = " << M.get_row_size() << std::endl;
        std::cout << "data.size() = " << data.size() << std::endl;
        throw std::runtime_error("Matrix and vector sizes do not match");
    }
    for (RC_INT j = 0; j < M.get_col_size(); j++) {
        for (RC_INT i = 0; i < M.get_row_size(); i++) {
            M.mData[j * M.get_row_size() + i] = data[i] * M.mData[j * M.get_row_size() + i];
        }
    }
}

template <>
void diag_operator<CPX, RC_host_matrix<CPX>>::apply(RC_host_matrix<CPX>& Min, RC_host_matrix<CPX>& Mout, CPX alpha, CPX beta) {
    // assert that Min.get_row_size() == data.size()
    if (Min.get_row_size() != data.size()) {
        std::cout << "Min.get_row_size() = " << Min.get_row_size() << std::endl;
        std::cout << "data.size() = " << data.size() << std::endl;
        throw std::runtime_error("Matrix and vector sizes do not match");
    }

    for (RC_INT i = 0; i < Min.get_row_size(); i++) {
        for (RC_INT j = 0; j < Min.get_col_size(); j++) {
            Mout.mData[j * Min.get_row_size() + i] = alpha * data[i] * Min.mData[j * Min.get_row_size() + i] + beta * Mout.mData[j * Min.get_row_size() + i];
        }
    }
}


template class diag_operator<CPX, RC_host_matrix<CPX>>;
