#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "SieveSortCUDA.cuh"

const size_t _8 = 1 << 3;			//3
const size_t _64 = _8 << 3;			//6
const size_t _512 = _64 << 3;		//9
const size_t _4K = _512 << 3;		//12
const size_t _32K = _4K << 3;		//15
const size_t _256K = _32K << 3;		//18
const size_t _2M = _256K << 3;		//21
const size_t _16M = _2M << 3;		//24
const size_t _128M = _16M << 3;		//27
const size_t _1G = _128M << 3;		//30
const size_t _8G = _1G << 3;		//33
const size_t _64G = _8G << 3;		//36
const size_t _512G = _64G << 3;		//39
const size_t _4T = _512G << 3;		//42
const size_t _32T = _4T << 3;		//45
const size_t _256T = _32T << 3;		//48
const size_t _2P = _256T << 3;		//51
const size_t _16P = _2P << 3;		//54
const size_t _128P = _16P << 3;		//57
const size_t _1E = _128P << 3;		//60
const size_t _8E = _1E << 3;		//63

__device__ static bool sieve_sort_64(uint32_t* a/*[64]*/, size_t n, uint32_t* result) {
	for (size_t i = 1; i < n; i++) {
		// 选择要插入的元素
		uint32_t key = a[i];
		// 从已排序序列的最后一个元素开始，向前比较
		size_t j = i - 1;
		// 如果当前元素小于它前面的元素，则将它前面的元素向后移动一位
		while (j >= 0 && a[j] > key) {
			a[j + 1] = a[j];
			j--;
		}
		// 将选择的元素插入到正确的位置
		a[j + 1] = key;
	}
	for (size_t i = 0; i < n; i++) result[i] = a[i];
	return true;
}

__device__ static bool sieve_collect(size_t n, size_t loops, size_t stride, size_t reminder,
	uint32_t* source, uint32_t* destination) {
	if (n == 0 || loops == 0 || loops > 16 || source == nullptr || destination == nullptr)
		return false;
	else
	{
		size_t ptr[16] = { 0 };
		size_t top[16] = { 0 };
		size_t p = 0;
		for (size_t i = 0; i < loops; i++)
		{
			ptr[i] = p;
			top[i] = p + ((i == loops - 1 && reminder > 0) ? reminder : stride);;
			p += stride;
		}

		size_t q = 0;
		uint32_t buffer[16] = { 0 };
		uint32_t min = ~0;
		uint32_t count = 0;
		while (q < n) {
			min = ~0;
			for (size_t i = 0; i < loops; i++) {
				if (ptr[i] < top[i]) {
					if ((buffer[i] = source[ptr[i]]) < min) min = buffer[i];
					ptr[i]++;
				}
			}
			for (size_t i = 0; i < loops; i++) {
				if ((ptr[i] < top[i]) && (buffer[i] == min)) {
					ptr[i]++;
					count++;
				}
			}
			for (int i = 0; i < count; i++) {
				destination[q++] = min;
			}
		}
		return true;
	}
	return false;
}

__global__ static void sieve_sort_kernel(uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth);
__global__ static void sieve_sort_kernel_(
	uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth,
	size_t loops, size_t stride, size_t reminder) {
	unsigned int i = threadIdx.x;
	sieve_sort_kernel<<<1,1>>>(
		a + i * stride,
		(i == loops - 1 && reminder > 0) ? reminder : stride,
		result + i * stride,
		max_depth, depth - 1);
}
__global__ static void sieve_sort_kernel(uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth) {
	if (n <= 64) {
		sieve_sort_64(a, n, result);
		return;
	}
	size_t loops = 0, stride = 0, reminder = 0;
	if (!get_config(n, loops, stride, reminder, 8, 4)) return;
	sieve_sort_kernel_ <<<1, loops >>> (a, n, result, max_depth, depth, loops, stride, reminder);
	__syncthreads();

	int delta_depth = max_depth - depth;
	if ((delta_depth & 1) == 1) {
		uint32_t* p = result;
		result = a;
		a = p;
	}
	sieve_collect(n, loops, stride, reminder, result, a);
}
__host__ bool sieve_sort_cuda(uint32_t* a, size_t n)
{
	//max(n)==256P (2^60)
	if (a == nullptr)
		return false;
	else if (n <= 1)
		return true;
	else if (n == 2) {
		uint32_t a0 = *(a + 0);
		uint32_t a1 = *(a + 1);
		*(a + 0) = (a0 <= a1) ? a0 : a1;
		*(a + 1) = (a0 >= a1) ? a0 : a1;
		return true;
	}
	else {
		uint32_t* input = nullptr;
		uint32_t* result = nullptr;
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		cudaStatus = cudaMalloc((void**)&input, n * sizeof(uint32_t));
		cudaStatus = cudaMalloc((void**)&result, n * sizeof(uint32_t));

		if (result != nullptr && input!=nullptr) {
			cudaStatus = cudaMemcpy(input, a, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

			int max_depth = get_depth(n, 4);
			sieve_sort_kernel<<<1,1>>>(input, n, result, max_depth, max_depth);

			cudaStatus = cudaDeviceSynchronize();
			
			cudaStatus = cudaMemcpy(a, input, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		}
		cudaFree(result);
		cudaFree(input);

		cudaStatus = cudaDeviceReset();

	}
	return true;
}
