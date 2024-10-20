#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "SieveSortCUDA.cuh"
#include <vector>
#include <map>
#include <algorithm>

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
			result[j + 1] = result[j];
			j--;
			if (j == ~0ULL) break;
		}
		a[j + 1] = key;
		result[j + 1] = key;
	}

	//for (size_t i = 0; i < n; i++) result[i] = a[i];
	//memcpy(result, a, n * sizeof(uint32_t));

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
			top[i] = p + ((i == loops - 1 && reminder > 0) ? reminder : stride);
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

struct partition {
	uint32_t* a;
	uint32_t* result;
	size_t n;
	size_t loops;
	size_t stride;
	size_t reminder;
	partition(uint32_t* a, uint32_t* result, size_t n, size_t loops, size_t stride, size_t reminder) {
		this->a = a;
		this->result = result;
		this->n = n;
		this->loops = loops;
		this->stride = stride;
		this->reminder = reminder;
	}
};


static void make_partitions(uint32_t* a, uint32_t* result, size_t n, int depth, std::map<int, std::vector<partition>>& partitions, int min_bits = 8, int shift = 4) {
	size_t loops = 0, stride = 0, reminder = 0;
	if (get_config(n, loops, stride, reminder, min_bits, shift))
	{
		auto f = partitions.find(depth);
		if (f == partitions.end()) {
			partitions.insert({ depth, { partition(a, result, n,loops,stride,reminder) } });
		}
		else {
			f->second.push_back(partition(a, result, n, loops, stride, reminder));
		}
		for (size_t i = 0; i < loops; i++) {
			make_partitions(a + i * stride, result + i * stride,
				(i == loops - 1 && reminder>0) ? reminder: stride, depth + 1, partitions, min_bits, shift);
		}
	}
	else {
		auto f = partitions.find(depth);
		if (f == partitions.end()) {
			partitions.insert({ depth, { partition(a,result,n,1,n,0) } });
		}
		else {
			f->second.push_back(partition(a, result, n, 1, n, 0));
		}
	}
}

__global__ static void sieve_sort_kerenl_with_config(partition* partitions, int max_depth, int depth) {
	unsigned int index =
		blockDim.x * blockIdx.x + threadIdx.x;

	partition* part = partitions + index;
	//printf("n=%lld,index=%d\n", pc->n, index);
	if (part->n <= 256) {
		sieve_sort_256(part->a, part->n, part->result);
	}
	else {
		uint32_t* destination = part->a;
		uint32_t* source = part->result;
		int delta_depth = max_depth - depth;
		bool flip = ((delta_depth & 1) == 1);
		flip = (((max_depth) & 1) == 1) ? !flip : flip;
		if (flip) {
			uint32_t* p = source;
			source = destination;
			destination = p;
		}
		sieve_collect(part->n, part->loops, part->stride, part->reminder, source, destination);
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
		std::map<int, std::vector<partition>> _partitions;
		uint32_t* input = nullptr;
		uint32_t* result = nullptr;
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(0);
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaMalloc((void**)&input, n * sizeof(uint32_t));
		cudaStatus = cudaMalloc((void**)&result, n * sizeof(uint32_t));

		if (result != nullptr && input != nullptr) {
			partition* partitions = nullptr;
			cudaStatus = cudaMemcpy(input, a, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
			cudaStatus = cudaMemcpy(result, input, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
			//cudaStatus = cudaMemset(result, 0, n * sizeof(uint32_t));

			make_partitions(input, result, n, 0, _partitions, 8, 4);
			int max_depth = _partitions.size() - 1;
			size_t max_list_size = 0;
			for (auto& partition:_partitions) {
				size_t s = partition.second.size();
				max_list_size = s > max_list_size ? s : max_list_size;
			}
			//printf("n = %lld, max_depth=%d\n",n, max_depth);
			cudaStatus = cudaMalloc((void**)&partitions, max_list_size * sizeof(partition));

			for (int i = max_depth; i >= 0; i--) {
				std::vector<partition>& partitions_list = _partitions[i];
				size_t list_size = partitions_list.size();
				if (list_size > 0) {
					cudaStatus = cudaMemcpy(partitions, (void*)partitions_list.data(), list_size * sizeof(partition), cudaMemcpyHostToDevice);
					if (list_size <= THREAD_NUM) {
						sieve_sort_kerenl_with_config << <1, list_size >> > (partitions, max_depth, i);
					}
					else {
						dim3 grid(ceil(list_size / (double)THREAD_NUM), 1, 1);
						dim3 block(THREAD_NUM, 1, 1);
						sieve_sort_kerenl_with_config << <grid, block >> > (partitions, max_depth, i);
					}
					cudaThreadSynchronize();
				}
			}

			cudaFree(partitions);
			cudaStatus = cudaMemcpy(a, input, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		}
		cudaFree(result);
		cudaFree(input);
	}
	return true;
}
