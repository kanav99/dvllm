#pragma once

#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <string>

#include <backend/backend.hpp>
#include <backend/cpu.hpp>
#include "constants.hpp"
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>

class MetalBackend : public CpuBackend
{
public:
    void matmul(float *d_y, float *d_x, float *d_w, int n, int d) override
    {
        // cblas_sgemv computes: y = alpha * W * x + beta * y
        // W is our weight matrix with `d` rows and `n` columns, stored row-major.
        // x is our input vector of size `n`.
        // y is our output vector of size `d`.

        cblas_sgemv(CblasRowMajor, // C-style row-major memory layout
                    CblasNoTrans,  // Do not transpose W
                    d,             // M: number of rows in W
                    n,             // N: number of columns in W
                    1.0f,          // alpha: scalar multiplier for W*x
                    d_w,           // A: the weight matrix
                    n,             // lda: leading dimension (elements per row)
                    d_x,           // X: the input vector
                    1,             // incX: stride between elements in X
                    0.0f,          // beta: scalar multiplier for initial Y
                    d_y,           // Y: the output vector
                    1);            // incY: stride between elements in Y
    }

    const char *getDeviceName() override
    {
        return "Metal";
    }
};
