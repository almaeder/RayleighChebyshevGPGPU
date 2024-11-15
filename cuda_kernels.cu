#include "cuda_kernels.h"

#define THREADS 1024


namespace cuda_kernels
{

template <typename T>
__global__ void _scale(
    T *data_d,
    const int n,
    const T scale
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        data_d[i] *= scale;
    }
}

__global__ void _scale(
    cuDoubleComplex *data_d,
    const int n,
    const cuDoubleComplex scale
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        data_d[i] = cuCmul(data_d[i], scale);
    }
}

template <typename T, typename D>
void scale(
    T *data_d,
    const int n,
    const D scale
){
    int blocks = (n + THREADS - 1) / THREADS;

    _scale<<<blocks, THREADS>>>(data_d, n, scale);

}
template void scale<double,double>(double *data_d, const int n, const double scale);

void scale(
    std::complex<double> *data_d,
    const int n,
    const double scale
){
    int blocks = (n + THREADS - 1) / THREADS;

    cuDoubleComplex scale_T = make_cuDoubleComplex(scale, 0.0);
    _scale<<<blocks, THREADS>>>(reinterpret_cast<cuDoubleComplex*>(data_d), n,
        scale_T);

}

template <typename T>
__global__ void _substract(
    T *a_d,
    T *b_d,
    const int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        a_d[i] -= b_d[i];
    }
}

__global__ void _substract(
    cuDoubleComplex *a_d,
    cuDoubleComplex *b_d,
    const int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        a_d[i] = cuCsub(a_d[i],b_d[i]);
    }
}


// a - b = a
template <typename T>
void substract(
    T *a_d,
    T *b_d,
    const int n
){
    int blocks = (n + THREADS - 1) / THREADS;

    if constexpr (std::is_same<T, std::complex<double>>::value){
        _substract<<<blocks, THREADS>>>(reinterpret_cast<cuDoubleComplex*>(a_d),
            reinterpret_cast<cuDoubleComplex*>(b_d), n);
    }
    else{
        _substract<<<blocks, THREADS>>>(a_d, b_d, n);
    }

}
template void substract<double>(double *a_d, double *b_d, const int n);
template void substract<std::complex<double>>(std::complex<double> *a_d, std::complex<double> *b_d, const int n);


template <typename T>
__device__ T _sumWarp(T a) {
    a += __shfl_down_sync(0xFFFFFFFF, a, 1);
    a += __shfl_down_sync(0xFFFFFFFF, a, 2);
    a += __shfl_down_sync(0xFFFFFFFF, a, 4);
    a += __shfl_down_sync(0xFFFFFFFF, a, 8);
    a += __shfl_down_sync(0xFFFFFFFF, a, 16);
    return a;
}

__device__ cuDoubleComplex _sumWarp(cuDoubleComplex a) {
    double real_part = cuCreal(a);
    double imag_part = cuCimag(a);

    real_part += __shfl_down_sync(0xFFFFFFFF, real_part, 1);
    real_part += __shfl_down_sync(0xFFFFFFFF, real_part, 2);
    real_part += __shfl_down_sync(0xFFFFFFFF, real_part, 4);
    real_part += __shfl_down_sync(0xFFFFFFFF, real_part, 8);
    real_part += __shfl_down_sync(0xFFFFFFFF, real_part, 16);
    imag_part += __shfl_down_sync(0xFFFFFFFF, imag_part, 1);
    imag_part += __shfl_down_sync(0xFFFFFFFF, imag_part, 2);
    imag_part += __shfl_down_sync(0xFFFFFFFF, imag_part, 4);
    imag_part += __shfl_down_sync(0xFFFFFFFF, imag_part, 8);
    imag_part += __shfl_down_sync(0xFFFFFFFF, imag_part, 16);

    a = make_cuDoubleComplex(real_part, imag_part);

    return a;
}


__device__ cuDoubleComplex _sumBlock(cuDoubleComplex a) {
    __shared__ cuDoubleComplex warpSums[32];
    int warpIdx = threadIdx.x / warpSize;
    cuDoubleComplex warpSum = _sumWarp(a);
    if (threadIdx.x % warpSize == 0)
        warpSums[warpIdx] = warpSum;
    __syncthreads();
    cuDoubleComplex blockSum = make_cuDoubleComplex(0.0, 0.0);
    if (warpIdx == 0)
        blockSum = _sumWarp(warpSums[threadIdx.x]);
    return blockSum;
}

template <typename T>
__device__ T _sumBlock(T a) {
    __shared__ T warpSums[32];
    int warpIdx = threadIdx.x / warpSize;
    T warpSum = _sumWarp(a);
    if (threadIdx.x % warpSize == 0)
        warpSums[warpIdx] = warpSum;
    __syncthreads();
    T blockSum = 0;
    if (warpIdx == 0)
        blockSum = _sumWarp(warpSums[threadIdx.x]);
    return blockSum;
}


template <typename T>
__global__ void sum1MKernel(const T *a, T *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T value = idx < N ? a[idx] : 0.0;
    T sum = _sumBlock(value);
    if (threadIdx.x == 0)
        b[blockIdx.x] = sum;
}

template <typename T>
__global__ void _reduce_norms(
    const T *a_d,
    T *b_d,
    const int m
){
    int idx_row = threadIdx.x;
    int idx_col = blockIdx.x;

    T a = 0.0;
    for(int i = idx_row; i < m; i += blockDim.x)
    {
        T tmp = a_d[i + idx_col * m];
        a += tmp * tmp;
    }
    T sum = _sumBlock(a);
    if (idx_row == 0){
        b_d[idx_col] = sqrt(a);        
    }
}

__global__ void _reduce_norms_complex(
    const cuDoubleComplex *a_d,
    cuDoubleComplex *b_d,
    const int m
){
    int idx_row = threadIdx.x;
    int idx_col = blockIdx.x;
    cuDoubleComplex a = make_cuDoubleComplex(0.0, 0.0);

    for(int i = idx_row; i < m; i += blockDim.x)
    {
        cuDoubleComplex tmp2 = a_d[i + idx_col * m];
        a = cuCadd(a,cuCmul(cuConj(tmp2),tmp2));
    }
    
    a = _sumBlock(a);

    if(idx_row == 0){
        b_d[idx_col] = make_cuDoubleComplex(sqrt(cuCabs(a)), 0.0);
    }
}


template <typename T>
__global__ void _reduce_norms_block(
    T *b_d
){
    int idx_row = threadIdx.x;
    int idx_col = blockIdx.x;

    T value = b_d[idx_row + idx_col * THREADS];
    T sum = _sumBlock(value);
    __syncthreads();
    if(threadIdx.x == 0){
        b_d[idx_col * THREADS] = sqrt(sum);
    }

}


template <typename T>
__global__ void _normalize(
    T *a_d,
    T *b_d,
    const int m,
    const int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < m*n; i += blockDim.x * gridDim.x)
    {
        int idx_col = i / m;
        T norm_value = b_d[idx_col];
        a_d[i] /= norm_value;
    }

}

__global__ void _normalize_complex(
    cuDoubleComplex *a_d,
    cuDoubleComplex *b_d,
    const int m,
    const int n
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < m*n; i += blockDim.x * gridDim.x)
    {

        int idx_col = i / m;
        cuDoubleComplex norm_value = b_d[idx_col];
        a_d[i] = cuCdiv(a_d[i], norm_value);
    }

}

template <typename T>
void normalize(
    T *a_d,
    const int m,
    const int n
){

    T *b_d;
    cudaMalloc(&b_d,  n * sizeof(T));

    if constexpr (std::is_same<T, std::complex<double>>::value){
        _reduce_norms_complex<<<n, THREADS>>>(
            reinterpret_cast<cuDoubleComplex*>(a_d),
            reinterpret_cast<cuDoubleComplex*>(b_d),
            m);

        int blocks = (m*n +  THREADS - 1) / THREADS;
        _normalize_complex<<<blocks, THREADS>>>(
            reinterpret_cast<cuDoubleComplex*>(a_d),
            reinterpret_cast<cuDoubleComplex*>(b_d),
            m, n);
    }
    else{
        _reduce_norms<<<n, THREADS>>>(a_d, b_d, m);

        int blocks = (m*n +  THREADS - 1) / THREADS;
        _normalize<<<blocks, THREADS>>>(a_d, b_d, m, n);
    }

    cudaFree(b_d);
}
template void normalize<double>(double *a_d, const int m, const int n);
template void normalize<std::complex<double>>(std::complex<double> *a_d, const int m, const int n);

template <typename T>
__global__ void _residuals(
    double *residuals_d,
    T *Av_d,
    T *v_d,
    double *energies,
    const int m
){
    int idx_row = threadIdx.x;
    int idx_col = blockIdx.x;
    T a = 0.0;

    for(int i = idx_row; i < m; i += blockDim.x)
    {
        T tmp = Av_d[i + idx_col * m] - energies[idx_col] * v_d[i + idx_col * m];
        a += tmp * tmp;
    }
    
    a = _sumBlock(a);

    if(idx_row == 0){
        residuals_d[idx_col] = sqrt(fabs(a));
    }

}

__global__ void _residuals_complex(
    double *residuals_d,
    cuDoubleComplex *Av_d,
    cuDoubleComplex *v_d,
    double *energies,
    const int m
){
    int idx_row = threadIdx.x;
    int idx_col = blockIdx.x;
    cuDoubleComplex a = make_cuDoubleComplex(0.0, 0.0);

    for(int i = idx_row; i < m; i += blockDim.x)
    {
        cuDoubleComplex eng = make_cuDoubleComplex(energies[idx_col], 0.0);
        cuDoubleComplex tmp = cuCsub(Av_d[i + idx_col * m], cuCmul(eng, v_d[i + idx_col * m]));
        a = cuCadd(a,cuCmul(cuConj(tmp),tmp));
    }
    
    a = _sumBlock(a);

    if(idx_row == 0){
        residuals_d[idx_col] = sqrt(cuCabs(a));
    }
}


template <typename T>
void residuals(
    double *residuals_d,
    T *Av_d,
    T *v_d,
    double *energies,
    const int m,
    const int n
){

    if constexpr (std::is_same<T, std::complex<double>>::value){
       _residuals_complex<<<n, THREADS>>>(residuals_d,
        reinterpret_cast<cuDoubleComplex*>(Av_d),
        reinterpret_cast<cuDoubleComplex*>(v_d),
        energies, m);
    }
    else{
        _residuals<<<n, THREADS>>>(residuals_d, Av_d, v_d, energies, m);
    }

}
template void residuals<double>(double *residuals_d, double *Av_d, double *v_d, double *energies, const int m, const int n);
template void residuals<std::complex<double>>(double *residuals_d, std::complex<double> *Av_d,
    std::complex<double> *v_d, double *energies, const int m, const int n);



} // namespace cuda_kernels