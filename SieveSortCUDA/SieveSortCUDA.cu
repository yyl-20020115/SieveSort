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

__device__ static bool sieve_sort_insert(uint32_t* a/*[256]*/, size_t n, uint32_t* result) {
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
	return true;
}

__device__ static bool sieve_collect(size_t n, size_t loops, size_t stride, size_t reminder,
	uint32_t* source, uint32_t* destination) {
	if (n == 0 || loops == 0 || source == nullptr || destination == nullptr)
		return false;
	else
	{
		size_t* ptr = new size_t[loops];
		size_t* top = new size_t[loops];
		uint32_t* buffer = new uint32_t[loops];// { 0 };
		memset(ptr, 0, loops * sizeof(size_t));
		memset(top, 0, loops * sizeof(size_t));
		memset(buffer, 0, sizeof(uint32_t) * loops);
		size_t p = 0;
		for (size_t i = 0; i < loops; i++)
		{
			ptr[i] = p;
			top[i] = p + ((i == loops - 1 && reminder > 0) ? reminder : stride);
			p += stride;
		}

		size_t q = 0;
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
		delete[] buffer;
		delete[] ptr;
		delete[] top;
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


static void make_partitions(uint32_t* a, uint32_t* result, size_t n, int depth, std::map<int, std::vector<partition>>& partitions, int min_bits = 8, int shift_bits = 4) {
	size_t loops = 0, stride = 0, reminder = 0;
	if (get_config(n, loops, stride, reminder, min_bits, shift_bits))
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
				(i == loops - 1 && reminder > 0) ? reminder : stride, depth + 1, partitions, min_bits, shift_bits);
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

__global__ static void sieve_sort_kerenl_with_config(partition* partitions, int max_depth, int depth, int min_bits) {
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	partition* part = partitions + index;
	if (part->n <= (1ULL << min_bits)) {
		sieve_sort_insert(part->a, part->n, part->result);
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
__host__ bool sieve_sort_cuda(uint32_t* a, size_t n, const int min_bits, const int shift_bits)
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
		cudaStatus = cudaMalloc((void**)&input, n * sizeof(uint32_t));
		cudaStatus = cudaMalloc((void**)&result, n * sizeof(uint32_t));

		if (result != nullptr && input != nullptr) {
			make_partitions(input, result, n, 0, _partitions, min_bits, shift_bits);
			if (_partitions.size() == 0) goto exit_me;

			cudaStatus = cudaMemcpy(input, a, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
			cudaStatus = cudaMemcpy(result, input, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

			int max_depth = _partitions.size() - 1;
			size_t max_list_size = _partitions.at(max_depth).size();
			partition* partitions = nullptr;
			cudaStatus = cudaMalloc((void**)&partitions, max_list_size * sizeof(partition));

			for (int i = max_depth; i >= 0; i--) {
				std::vector<partition>& partitions_list = _partitions[i];
				size_t list_size = partitions_list.size();
				if (list_size > 0) {
					cudaStatus = cudaMemcpy(partitions, (void*)partitions_list.data(), list_size * sizeof(partition), cudaMemcpyHostToDevice);
					if (list_size <= THREAD_NUM) {
						sieve_sort_kerenl_with_config <<<
							1, list_size 
							>>> (partitions, max_depth, i, min_bits);
					}
					else {
						sieve_sort_kerenl_with_config <<<
							dim3(ceil(list_size / (double)THREAD_NUM), 1, 1),
							dim3(THREAD_NUM, 1, 1) 
							>>> (partitions, max_depth, i, min_bits);
					}
					cudaThreadSynchronize();
				}
			}

			cudaFree(partitions);
			cudaStatus = cudaMemcpy(a, input, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		exit_me:
			;
		}
		cudaFree(result);
		cudaFree(input);
	}
	return true;
}
