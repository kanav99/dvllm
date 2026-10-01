#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "constants.hpp"

__global__ void modular_integer_matmul_kernel(const integer_t *d_x,
                                             const integer_w_t *d_w,
                                             integer_t *d_y,
                                             int n,
                                             int d)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= d)
    {
        return;
    }

    integer_t sum = 0;
    for (int col = 0; col < n; ++col)
    {
        integer_t product = static_cast<integer_t>(d_w[row * n + col]) * d_x[col];
        sum += product;
    }

    d_y[row] = sum;
}

void rocblas_integer_matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d)
{
    dim3 block(256);
    dim3 grid((d + block.x - 1) / block.x);

    modular_integer_matmul_kernel<<<grid, block>>>(d_x, d_w, d_y, n, d);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess)
    {
        std::cerr << "HIP kernel launch error: " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
    status = hipDeviceSynchronize();
    if (status != hipSuccess)
    {
        std::cerr << "HIP synchronize error: " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}