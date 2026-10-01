#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#define CHECK_ROCBLAS_ERROR(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_HIP_ERROR(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

    // function that copies matrices from host to device, does matmul, and copies result back
void matmul(const float *h_A, const float *h_B, float *h_C, float *d_A, float *d_B, float *d_C, int N, rocblas_handle &handle) {
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Matrix dimensions: C = alpha * A * B + beta * C
    // A: N x N, B: N x N, C: N x N
    CHECK_ROCBLAS_ERROR(rocblas_sgemm(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ));

    // Wait for computation to finish
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Copy result back to host
    CHECK_HIP_ERROR(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));
}

void matmul_cpu(const float *h_A, const float *h_B, float *h_C, int N) {
    Eigen::Map<const Eigen::MatrixXf> A(h_A, N, N);
    Eigen::Map<const Eigen::MatrixXf> B(h_B, N, N);
    Eigen::Map<Eigen::MatrixXf> C(h_C, N, N);

    C = A * B;
}

void print_device()
{
    char name[256];
    hipDevice_t device;
    CHECK_HIP_ERROR(hipGetDevice(&device));
    CHECK_HIP_ERROR(hipDeviceGetName(name, 256, device));
    std::cout << "Device: " << name << std::endl;
}

int main() {
    print_device();
    const int N = 2048;
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    // Host matrices
    std::vector<float> h_A(size, 1.0f); // A filled with 1s
    std::vector<float> h_B(size, 2.0f); // B filled with 2s
    std::vector<float> h_C(size, 0.0f); // Output matrix

    // Device matrices
    float *d_A, *d_B, *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, bytes));
    CHECK_HIP_ERROR(hipMalloc(&d_B, bytes));
    CHECK_HIP_ERROR(hipMalloc(&d_C, bytes));

    // Create rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    // warm up the GPU
    for (int i = 0; i < 10; ++i) {
        matmul(h_A.data(), h_B.data(), h_C.data(), d_A, d_B, d_C, N, handle);
    }

    // Measure performance
    int num_iterations = 20;
    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));
    CHECK_HIP_ERROR(hipEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        matmul(h_A.data(), h_B.data(), h_C.data(), d_A, d_B, d_C, N, handle);
        // matmul_cpu(h_A.data(), h_B.data(), h_C.data(), N);
    }
    CHECK_HIP_ERROR(hipEventRecord(stop));
    CHECK_HIP_ERROR(hipEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_HIP_ERROR(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time taken for " << num_iterations << " iterations: " << milliseconds << " ms" << std::endl;
    std::cout << "Time per iteration: " << milliseconds / num_iterations << " ms" << std::endl;

    // Cleanup
    rocblas_destroy_handle(handle);
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));

    return 0;
}
