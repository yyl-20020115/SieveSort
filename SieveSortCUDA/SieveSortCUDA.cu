#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "SieveSortCUDA.cuh"
#include <vector>
#include <map>

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

__device__ static bool sieve_sort_256(uint32_t* a/*[256]*/, size_t n, uint32_t* result) {
	for (size_t i = 1; i < n; i++) {
		uint32_t key = a[i];
		size_t j = i - 1;
		while (j >= 0 && a[j] > key) {
			a[j + 1] = a[j];
			j--;
			if (j == ~0ULL) break;
		}
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
			count = 0;
			for (size_t i = 0; i < loops; i++) {
				if (ptr[i] < top[i]) {
					buffer[i] = source[ptr[i]];
					if (buffer[i] < min) min = buffer[i];
				}
				else {
					buffer[i] = ~0;
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



struct config {
	uint32_t* a;
	uint32_t* result;
	size_t n;
	size_t loops;
	size_t stride;
	size_t reminder;
	config(uint32_t* a, uint32_t* result, size_t n, size_t loops, size_t stride, size_t reminder) {
		this->a = a;
		this->result = result;
		this->n = n;
		this->loops = loops;
		this->stride = stride;
		this->reminder = reminder;
	}
};



void make_configs(uint32_t* a, uint32_t* result, size_t n, int depth, std::map<int, std::vector<config>>& configs, int min_bits = 8, int shift = 4) {
	size_t loops = 0, stride = 0, reminder = 0;
	if (get_config(n, loops, stride, reminder, min_bits, shift))
	{
		auto f = configs.find(depth);
		if (f == configs.end()) {
			configs.insert({ depth, { config(a, result, n,loops,stride,reminder) } });
		}
		else {
			f->second.push_back(config(a, result, n, loops, stride, reminder));
		}
		for (size_t i = 0; i < loops; i++) {
			make_configs(a + i * stride, result + i * stride,
				(i < loops - 1) ? stride : reminder, depth + 1, configs, min_bits, shift);
		}
	}
	else {
		auto f = configs.find(depth);
		if (f == configs.end()) {
			configs.insert({ depth, { config(a,result,n,1,n,0) } });
		}
		else {
			f->second.push_back(config(a, result, n, 1, n, 0));
		}
	}
}

__global__ static void sieve_sort_kernel(uint32_t* p_atomic, uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth);
__global__ static void sieve_sort_kernel_bridge(uint32_t* p_atomic, uint32_t* a, uint32_t* result, int max_depth, int depth, size_t loops, size_t stride, size_t reminder) {
	unsigned int i = threadIdx.x;
	sieve_sort_kernel << <1, 1 >> > (
		p_atomic,
		a + i * stride,
		(i == loops - 1 && reminder > 0) ? reminder : stride,
		result + i * stride,
		max_depth, depth - 1);
}
__global__ static void sieve_sort_kernel(uint32_t* p_atomic, uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth) {
	int delta_depth = max_depth - depth;
	if (n <= 256) {
		sieve_sort_256(a, n, result);
	}
	else {
		size_t loops = 0, stride = 0, reminder = 0;

		if (!get_config(n, loops, stride, reminder, 8, 4)) return;

		__shared__ uint32_t d_atomic;
		d_atomic = 0;

		uint32_t* p_atomic = &d_atomic;
		sieve_sort_kernel_bridge << <1, loops >> > (p_atomic, a, result, max_depth, depth, loops, stride, reminder);

		while (*p_atomic < loops)
			;

		if ((delta_depth & 1) == 1) {
			uint32_t* p = result;
			result = a;
			a = p;
		}
		sieve_collect(n, loops, stride, reminder, result, a);
	}
	atomicAdd(p_atomic, 1);

}

__global__ static void sieve_sort_kerenl_with_config(config* configs, int max_depth, int depth) {
	unsigned int index =
		blockDim.x * blockIdx.x + threadIdx.x;

	config* pc = configs + index;
	if (pc->n <= 256) {
		sieve_sort_256(pc->a, pc->n, pc->result);
	}
	else {
		uint32_t* a = pc->a;
		uint32_t* result = pc->result;

		int delta_depth = max_depth - depth;
		if ((delta_depth & 1) == 1) {
			uint32_t* p = result;
			result = a;
			a = p;
		}
		sieve_collect(pc->n, pc->loops, pc->stride, pc->reminder, result, a);
	}
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
		std::map<int, std::vector<config>> configs;


		uint32_t* input = nullptr;
		uint32_t* result = nullptr;
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaMalloc((void**)&input, n * sizeof(uint32_t));
		cudaStatus = cudaMalloc((void**)&result, n * sizeof(uint32_t));

		if (result != nullptr && input != nullptr) {
			config* configs_ = nullptr;
			cudaStatus = cudaMemcpy(input, a, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
			make_configs(input, result, n, 0, configs, 8, 4);
			int max_depth = configs.size();
			
			for (int i = max_depth - 1; i >= 0; i--) {
				std::vector<config>& config_list = configs[i];
				size_t list_size = config_list.size();
				if (list_size > 0) {
					void* pd = config_list.data();
					cudaStatus = cudaMalloc((void**)&configs_, list_size * sizeof(config));
					cudaStatus = cudaMemcpy(configs_, pd, list_size * sizeof(config), cudaMemcpyHostToDevice);

					if (list_size <= THREAD_NUM) {
						sieve_sort_kerenl_with_config << <1, list_size >> > (configs_, max_depth, i);
						cudaThreadSynchronize();
					}
					else {
						dim3 grid(ceil(list_size / (double)THREAD_NUM), 1, 1);
						dim3 block(THREAD_NUM, 1, 1);
						sieve_sort_kerenl_with_config << <grid, block >> > (configs_, max_depth, i);
						cudaThreadSynchronize();
					}
					cudaFree(configs_);
				}
			}

			cudaStatus = cudaMemcpy(a, input, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		}
		cudaFree(result);
		cudaFree(input);
	}
	return true;
}
