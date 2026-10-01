#pragma once

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <backend/backend.hpp>
#include "constants.hpp"

void rocblas_integer_matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d);

#define CHECK_ROCBLAS_ERROR(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << "Error: " << hipGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class RocblasBackend: public Backend
{
private:
    rocblas_handle handle;
    hipStream_t streamHandles[128]; // Assuming a maximum of 128 streams for simplicity

public:
    RocblasBackend()
    {
        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
        for (int i = 0; i < 128; ++i) {
            HIP_CHECK(hipStreamCreate(&streamHandles[i]));
        }
    }

    ~RocblasBackend()
    {
        CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
        for (int i = 0; i < 128; ++i) {
            HIP_CHECK(hipStreamDestroy(streamHandles[i]));
        }
    }
    
    void matmul(float *d_y, float *d_x, float *d_w, int n, int d) {
        float alpha = 1.0f;
        float beta = 0.0f;

        // sgemm
        CHECK_ROCBLAS_ERROR(
            rocblas_sgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,    
                d, 1, n,
                &alpha,
                d_w, n,
                d_x, n,
                &beta,
                d_y, d
            )
        );
    }

    void matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d) {
        rocblas_integer_matmul(d_y, d_x, d_w, n, d);
    }

    void malloc(void **d_ptr, size_t size) override
    {
        hipError_t status = hipMalloc(d_ptr, size);
        if (status != hipSuccess) {
            std::cerr << "hipMalloc error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void free(void *d_ptr) override
    {
        hipError_t status = hipFree(d_ptr);
        if (status != hipSuccess) {
            std::cerr << "hipFree error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void copyHostToDevice(void *d_ptr, void *h_ptr, size_t size) override
    {
        hipError_t status = hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
        if (status != hipSuccess) {
            std::cerr << "hipMemcpy error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void copyDeviceToHost(void *h_ptr, void *d_ptr, size_t size) override
    {
        hipError_t status = hipMemcpy(h_ptr, d_ptr, size, hipMemcpyDeviceToHost);
        if (status != hipSuccess) {
            std::cerr << "hipMemcpy error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // void quantizeHostToDevice(integer_w_t *d_ptr, float *h_ptr, size_t num_elements) override
    // {
    //     // Allocate temporary host buffer for quantized values
    //     integer_w_t* temp_buffer = new integer_w_t[num_elements];
    //     for (size_t i = 0; i < num_elements; i++) {
    //         temp_buffer[i] = static_cast<integer_w_t>(h_ptr[i] * FP_W_SCALE);
    //     }
    //     // Copy quantized values to device
    //     copyHostToDevice(d_ptr, temp_buffer, num_elements * sizeof(integer_w_t));
    //     delete[] temp_buffer;
    // }

    void copyHostToDeviceAsync(void *d_ptr, void *h_ptr, size_t size, int handle) override
    {
        if (handle < 0 || handle >= 128) {
            std::cerr << "Invalid handle: " << handle << std::endl;
            std::exit(EXIT_FAILURE);
        }
        HIP_CHECK(hipMemcpyAsync(d_ptr, h_ptr, size, hipMemcpyHostToDevice, streamHandles[handle]));
    }

    void waitForOperation(int handle) override
    {
        if (handle < 0 || handle >= 128) {
            std::cerr << "Invalid handle: " << handle << std::endl;
            std::exit(EXIT_FAILURE);
        }
        HIP_CHECK(hipStreamSynchronize(streamHandles[handle]));
    }

    void pinMemory(void *ptr, size_t size) override
    {
        HIP_CHECK(hipHostRegister(ptr, size, hipHostRegisterDefault));
    }

    void unpinMemory(void *ptr) override
    {
        HIP_CHECK(hipHostUnregister(ptr));
    }

    bool isDeviceAvailable() override
    {
        int device_count;
        hipError_t status = hipGetDeviceCount(&device_count);
        if (status != hipSuccess || device_count == 0) {
            return false;
        }
        return true;
    }

    const char* getDeviceName() override
    {
        static char device_name[256];
        hipDevice_t device;
        hipError_t status = hipGetDevice(&device);
        if (status != hipSuccess) {
            std::cerr << "hipGetDevice error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
        status = hipDeviceGetName(device_name, sizeof(device_name), device);
        if (status != hipSuccess) {
            std::cerr << "hipDeviceGetName error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
        return device_name;
    }
};
