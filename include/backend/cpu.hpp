#pragma once

#include <backend/backend.hpp>
#include "constants.hpp"
#include <cstring>
#include <thread>
#include <algorithm> // for std::min

class CpuBackend: public Backend
{
    std::thread handles[128]; 
public:
    CpuBackend()
    {
    }

    ~CpuBackend()
    {
    }
    
    void matmul(float *d_y, float *d_x, float *d_w, int n, int d) override {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += d_w[i * n + j] * d_x[j];
            }
            d_y[i] = val;
        }
    }

    void matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d, int batch_size = 1) override {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
        for (int b = 0; b < batch_size; b++) {
            const integer_t *x_tok = d_x + b * n;
            integer_t *y_tok = d_y + b * num_chunks * d;
            int i;
            #pragma omp parallel for private(i)
            for (i = 0; i < d; i++) {
                for (int c = 0; c < num_chunks; ++c) {
                    integer_t val = 0;
                    int start_col = c * CHUNK_SIZE;
                    int end_col = std::min(start_col + CHUNK_SIZE, n);
                    for (int j = start_col; j < end_col; j++) {
                        val += static_cast<integer_t>(d_w[i * n + j]) * x_tok[j];
                    }
                    y_tok[c * d + i] = val;
                }
            }
        }
    }

    void malloc(void **d_ptr, size_t size) override
    {
        *d_ptr = new char[size];
    }

    void free(void *d_ptr) override
    {
        char *ptr = static_cast<char *>(d_ptr);
        delete[] ptr;
    }

    void copyHostToDevice(void *d_ptr, void *h_ptr, size_t size) override
    {
        std::memcpy(d_ptr, h_ptr, size);
    }

    void copyDeviceToHost(void *h_ptr, void *d_ptr, size_t size) override
    {
        std::memcpy(h_ptr, d_ptr, size);

    }

    // void quantizeHostToDevice(integer_w_t *d_ptr, float *h_ptr, size_t num_elements) override
    // {
    //     for (size_t i = 0; i < num_elements; i++) {
    //         d_ptr[i] = integer_w_t(h_ptr[i] * FP_W_SCALE);
    //     }
    // }

    void copyHostToDeviceAsync(void *d_ptr, void *h_ptr, size_t size, int handle) override
    {
        // Safety check: If a previous connection left a thread running in this slot, 
        // join it before overwriting it to prevent std::terminate()
        if (handles[handle].joinable()) {
            handles[handle].join();
        }
        
        handles[handle] = std::thread([d_ptr, h_ptr, size]() {
            std::memcpy(d_ptr, h_ptr, size);
        });
    }

    void waitForOperation(int handle) override
    {
        // Check if the thread actually exists and is running before trying to join
        if (handles[handle].joinable()) {
            handles[handle].join();
        }
    }

    bool isDeviceAvailable() override
    {
        return true;
    }

    const char* getDeviceName() override
    {
        return "CPU";
    }

    bool isCpuBackend() override { return true; }
};
