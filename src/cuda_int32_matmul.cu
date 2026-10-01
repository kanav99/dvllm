#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#include "constants.hpp"

__global__ void modular_integer_matmul_kernel(const integer_t *d_x,
                                             const integer_w_t *d_w,
                                             integer_t *d_y,
                                             int n,
                                             int d,
                                             int num_chunks,
                                             int batch_size)
{
    int chunk = blockIdx.y;
    int batch = blockIdx.z;

    if (chunk >= num_chunks || batch >= batch_size) return;

    // A block of 256 threads has 8 warps (32 threads each).
    // Each warp processes 4 rows simultaneously (Thread Tiling).
    // Total rows processed per block = 8 * 4 = 32.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_start = blockIdx.x * 32 + warp_id * 4;

    // Prevent out-of-bounds blocks from doing work
    if (row_start >= d) return;

    int start_col = chunk * CHUNK_SIZE;
    int end_col = min(start_col + CHUNK_SIZE, n);
    
    const integer_t *x_tok = d_x + batch * n;

    // Opt 2: Shared Memory Caching
    __shared__ int64_t s_x[128]; 
    
    // Opt 3: 1D Thread Tiling (4 rows computed per thread)
    int64_t sum[4] = {0, 0, 0, 0}; 

    // Process the chunk in tiles of 128 columns
    for (int col = start_col; col < end_col; col += 128)
    {
        int valid_cols = min(128, end_col - col);

        // Opt 1 & 5: Cooperative coalesced load of X into shared memory
        // Only the first 128 threads of the 256-thread block need to do this
        if (threadIdx.x < valid_cols) {
            s_x[threadIdx.x] = x_tok[col + threadIdx.x];
        }
        __syncthreads(); // Ensure shared memory is fully loaded

        // Opt 4: Vectorized Memory Access
        // If we have a full 128-column tile, vectorize it.
        if (col + 128 <= end_col) {
            // Each warp of 32 threads processes 128 columns (4 columns per thread)
            int c = col + lane_id * 4;
            int s_c = lane_id * 4;

            // Preload the shared X values into fast local registers
            int64_t x_vals[4];
            x_vals[0] = s_x[s_c + 0];
            x_vals[1] = s_x[s_c + 1];
            x_vals[2] = s_x[s_c + 2];
            x_vals[3] = s_x[s_c + 3];

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int current_row = row_start + r;
                if (current_row < d) {
                    // Vectorized load: Fetch 4 int16_t weights as a single 64-bit load
                    const int64_t* w_vec_ptr = reinterpret_cast<const int64_t*>(&d_w[current_row * n + c]);
                    int64_t w_vec = *w_vec_ptr;

                    // Unpack the 64-bit load back into 4 x int16_t
                    int16_t w_vals[4];
                    w_vals[0] = w_vec & 0xFFFF;
                    w_vals[1] = (w_vec >> 16) & 0xFFFF;
                    w_vals[2] = (w_vec >> 32) & 0xFFFF;
                    w_vals[3] = (w_vec >> 48) & 0xFFFF;

                    // Compute math for this row
                    int64_t local_sum = 0;
                    local_sum += static_cast<int64_t>(w_vals[0]) * x_vals[0];
                    local_sum += static_cast<int64_t>(w_vals[1]) * x_vals[1];
                    local_sum += static_cast<int64_t>(w_vals[2]) * x_vals[2];
                    local_sum += static_cast<int64_t>(w_vals[3]) * x_vals[3];

                    sum[r] += local_sum;
                }
            }
        } 
        // Boundary case: For the final remainder of the chunk, fall back to scalar math
        else {
            int c = col + lane_id;
            for (int step = 0; step < valid_cols; step += 32) {
                int current_c = c + step;
                bool valid = current_c < end_col;
                int64_t x_val = valid ? s_x[lane_id + step] : 0;
                
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int current_row = row_start + r;
                    if (current_row < d && valid) {
                        sum[r] += static_cast<int64_t>(d_w[current_row * n + current_c]) * x_val;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Standard CUDA Warp Reduction to sum up the 128 columns
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int64_t val = sum[r];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // Lane 0 holds the final reduction for this chunk/row combination
        if (lane_id == 0) {
            int current_row = row_start + r;
            if (current_row < d) {
                d_y[(batch * num_chunks + chunk) * d + current_row] = val;
            }
        }
    }
}

void cuda_integer_matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d, cudaStream_t stream, int batch_size = 1)
{
    int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    dim3 block(256, 1, 1); 
    
    // We scaled up arithmetic intensity: 1 block handles 32 rows instead of 256.
    int rows_per_block = 32;
    dim3 grid((d + rows_per_block - 1) / rows_per_block, num_chunks, batch_size);

    modular_integer_matmul_kernel<<<grid, block, 0, stream>>>(d_x, d_w, d_y, n, d, num_chunks, batch_size);
    
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA stream synchronize error: " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}