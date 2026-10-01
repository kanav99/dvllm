// Standalone HIP sample for modular int32 matrix multiplication.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_HIP_ERROR(status)                                                                 \
	do                                                                                          \
	{                                                                                           \
		if ((status) != hipSuccess)                                                             \
		{                                                                                       \
			std::cerr << "HIP error: " << (status) << " in " << __FILE__ << ":" << __LINE__  \
					  << std::endl;                                                             \
			std::exit(EXIT_FAILURE);                                                            \
		}                                                                                       \
	} while (0)

static void fill_matrix(std::vector<int32_t> &matrix, int stride)
{
	for (size_t i = 0; i < matrix.size(); ++i)
	{
		matrix[i] = static_cast<int32_t>((static_cast<int>(i % stride) - stride / 2));
	}
}

static void cpu_matmul(const std::vector<int32_t> &a,
					   const std::vector<int32_t> &b,
					   std::vector<int32_t> &c,
					   int n)
{
	for (int row = 0; row < n; ++row)
	{
		for (int col = 0; col < n; ++col)
		{
			uint32_t sum = 0;
			for (int k = 0; k < n; ++k)
			{
				sum += static_cast<uint32_t>(static_cast<int64_t>(a[row * n + k]) * static_cast<int64_t>(b[k * n + col]));
			}
			c[row * n + col] = static_cast<int32_t>(sum);
		}
	}
}

__global__ void modular_matmul_kernel(const int32_t *a, const int32_t *b, int32_t *c, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= n || col >= n)
	{
		return;
	}

	uint32_t sum = 0;
	for (int k = 0; k < n; ++k)
	{
		int64_t product = static_cast<int64_t>(a[row * n + k]) * static_cast<int64_t>(b[k * n + col]);
		sum += static_cast<uint32_t>(product);
	}

	c[row * n + col] = static_cast<int32_t>(sum);
}

static void print_device_name()
{
	hipDevice_t device;
	char device_name[256];

	CHECK_HIP_ERROR(hipGetDevice(&device));
	CHECK_HIP_ERROR(hipDeviceGetName(device_name, sizeof(device_name), device));

	std::cout << "Device: " << device_name << std::endl;
}

int main(int argc, char **argv)
{
	int n = 512;
	if (argc >= 2)
	{
		n = std::stoi(argv[1]);
	}

	if (n <= 0)
	{
		std::cerr << "Matrix dimension must be positive" << std::endl;
		return EXIT_FAILURE;
	}

	print_device_name();

	const size_t element_count = static_cast<size_t>(n) * static_cast<size_t>(n);
	const size_t int32_bytes = element_count * sizeof(int32_t);

	std::vector<int32_t> h_a(element_count);
	std::vector<int32_t> h_b(element_count);
	std::vector<int32_t> h_c(element_count, 0);
	std::vector<int32_t> h_reference(element_count, 0);

	fill_matrix(h_a, 11);
	fill_matrix(h_b, 13);

	int32_t *d_a = nullptr;
	int32_t *d_b = nullptr;
	int32_t *d_c = nullptr;

	CHECK_HIP_ERROR(hipMalloc(&d_a, int32_bytes));
	CHECK_HIP_ERROR(hipMalloc(&d_b, int32_bytes));
	CHECK_HIP_ERROR(hipMalloc(&d_c, int32_bytes));

	CHECK_HIP_ERROR(hipMemcpy(d_a, h_a.data(), int32_bytes, hipMemcpyHostToDevice));
	CHECK_HIP_ERROR(hipMemcpy(d_b, h_b.data(), int32_bytes, hipMemcpyHostToDevice));
	CHECK_HIP_ERROR(hipMemset(d_c, 0, int32_bytes));

	dim3 block(16, 16);
	dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
	modular_matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
	CHECK_HIP_ERROR(hipGetLastError());

	CHECK_HIP_ERROR(hipDeviceSynchronize());
	CHECK_HIP_ERROR(hipMemcpy(h_c.data(), d_c, int32_bytes, hipMemcpyDeviceToHost));

	cpu_matmul(h_a, h_b, h_reference, n);

	size_t mismatch_count = 0;
	for (size_t i = 0; i < element_count; ++i)
	{
		if (h_c[i] != h_reference[i])
		{
			++mismatch_count;
			if (mismatch_count <= 8)
			{
				std::cerr << "Mismatch at index " << i << ": got " << h_c[i]
						  << ", expected " << h_reference[i] << std::endl;
			}
		}
	}

	std::cout << "Computed C = A x B for " << n << "x" << n << " int32_t matrices" << std::endl;
	std::cout << "First element: " << h_c[0] << std::endl;
	if (mismatch_count == 0)
	{
		std::cout << "Validation: passed" << std::endl;
	}
	else
	{
		std::cout << "Validation: failed with " << mismatch_count << " mismatches" << std::endl;
	}

	CHECK_HIP_ERROR(hipFree(d_a));
	CHECK_HIP_ERROR(hipFree(d_b));
	CHECK_HIP_ERROR(hipFree(d_c));

	return mismatch_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
