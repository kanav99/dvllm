#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <backend/backend.hpp>

#define CHECK_CUBLAS_ERROR(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << status << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

void cuda_integer_matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d, cudaStream_t stream, int batch_size = 1);

const int MAX_NUM_STREAMS = 1024; // Maximum number of CUDA streams for async operations

class CublasBackend : public Backend
{
private:
    int num_devices_ = 1;
    int current_device_ = 0;
    std::vector<cublasHandle_t> handles_;
    std::vector<std::vector<cudaStream_t>> streamHandles_;

public:
    CublasBackend() {
        initDevices(1);
    }

    void initDevices(int num_devices) override {
        // Clean up if already initialized
        if (!handles_.empty()) {
            for (int d = 0; d < num_devices_; d++) {
                cudaSetDevice(d);
                cublasDestroy(handles_[d]);
                for (int i = 0; i < MAX_NUM_STREAMS; i++) {
                    cudaStreamDestroy(streamHandles_[d][i]);
                }
            }
        }

        int actual_devices = 0;
        cudaGetDeviceCount(&actual_devices);
        if (num_devices > actual_devices) {
            std::cerr << "Requested " << num_devices << " devices, but only " << actual_devices << " are available. Scaling down." << std::endl;
            num_devices = actual_devices;
        }
        num_devices_ = num_devices;

        handles_.resize(num_devices_);
        streamHandles_.resize(num_devices_, std::vector<cudaStream_t>(MAX_NUM_STREAMS));

        for (int d = 0; d < num_devices_; d++) {
            cudaSetDevice(d);
            CHECK_CUBLAS_ERROR(cublasCreate(&handles_[d]));
            for (int i = 0; i < MAX_NUM_STREAMS; i++) {
                cudaError_t status = cudaStreamCreate(&streamHandles_[d][i]);
                if (status != cudaSuccess) {
                    std::cerr << "cudaStreamCreate error: " << status << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
        cudaSetDevice(0);
        current_device_ = 0;
    }

    ~CublasBackend() {
        for (int d = 0; d < num_devices_; d++) {
            cudaSetDevice(d);
            cublasDestroy(handles_[d]);
            for (int i = 0; i < MAX_NUM_STREAMS; i++) {
                cudaStreamDestroy(streamHandles_[d][i]);
            }
        }
    }

    void setDevice(int device) override {
        if (device >= 0 && device < num_devices_) {
            current_device_ = device;
            cudaSetDevice(device);
        }
    }

    int getNumDevices() override {
        return num_devices_;
    }

    void matmul(float *d_y, float *d_x, float *d_w, int n, int d) override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        // Set the active stream on cuBLAS handle to match the device context
        CHECK_CUBLAS_ERROR(cublasSetStream(handles_[current_device_], streamHandles_[current_device_][0]));

        CHECK_CUBLAS_ERROR(
            cublasSgemm(
                handles_[current_device_],
                CUBLAS_OP_T, CUBLAS_OP_N,
                d, 1, n,
                &alpha,
                d_w, n,   
                d_x, n,   
                &beta,
                d_y, d    
            )
        );
    }

    void matmul(integer_t *d_y, integer_t *d_x, integer_w_t *d_w, int n, int d, int batch_size = 1) override {
        // Pass the stream associated with the active device execution track
        cuda_integer_matmul(d_y, d_x, d_w, n, d, streamHandles_[current_device_][0], batch_size);
    }

    void malloc(void **d_ptr, size_t size) override
    {
        cudaError_t status = cudaMalloc(d_ptr, size);
        if (status != cudaSuccess) {
            std::cerr << "cudaMalloc error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void free(void *d_ptr) override
    {
        cudaError_t status = cudaFree(d_ptr);
        if (status != cudaSuccess) {
            std::cerr << "cudaFree error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void copyHostToDevice(void *d_ptr, void *h_ptr, size_t size) override
    {
        cudaError_t status = cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpy (Host to Device) error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void copyDeviceToHost(void *h_ptr, void *d_ptr, size_t size) override
    {
        cudaError_t status = cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpy (Device to Host) error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void copyHostToDeviceAsync(void *d_ptr, void *h_ptr, size_t size, int handle) override
    {
        int s_idx = handle % MAX_NUM_STREAMS;
        cudaError_t status = cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, streamHandles_[current_device_][s_idx]);
        if (status != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void waitForOperation(int handle) override
    {
        int s_idx = handle % MAX_NUM_STREAMS;
        cudaError_t status = cudaStreamSynchronize(streamHandles_[current_device_][s_idx]);
        if (status != cudaSuccess) {
            std::cerr << "cudaStreamSynchronize error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void pinMemory(void *ptr, size_t size) override
    {
        // Register host memory as page-locked for async DMA transfers
        cudaError_t status = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
        if (status != cudaSuccess) {
            std::cerr << "cudaHostRegister error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void unpinMemory(void *ptr) override
    {
        cudaError_t status = cudaHostUnregister(ptr);
        if (status != cudaSuccess) {
            std::cerr << "cudaHostUnregister error: " << status << std::endl;
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

    bool isDeviceAvailable() override
    {
        int device_count = 0;
        cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess) {
            return false;
        }
        return device_count > 0;
    }

    const char* getDeviceName() override
    {
        static char device_name[256];
        cudaDeviceProp prop;
        cudaError_t status = cudaGetDeviceProperties(&prop, current_device_);
        if (status != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties error: " << status << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        // Ensure string copy doesn't overflow
        strncpy(device_name, prop.name, sizeof(device_name) - 1);
        device_name[sizeof(device_name) - 1] = '\0';
        
        return device_name;
    }
};