#pragma once
#include <cuda_runtime.h>
#include <complex>
#include <cuComplex.h>
#include <type_traits>

namespace cuda_kernels
{

template <typename T, typename D>
void scale(
    T *data_d,
    const int n,
    const D scale
);

void scale(
    std::complex<double> *data_d,
    const int n,
    const double scale
);

template <typename T>
void substract(
    T *a_d,
    T *b_d,
    const int n
);

template <typename T>
void residuals(
    double *residuals_d,
    T *Av_d,
    T *v_d,
    double *energies,
    const int m,
    const int n
);

template <typename T>
void normalize(
    T *a_d,
    const int m,
    const int n
);

} // namespace cuda_kernels