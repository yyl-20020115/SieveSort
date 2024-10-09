#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

uint32_t generate_random_32(size_t range = 0ULL);
uint64_t generate_random_64(uint64_t range = 0ULL);

__device__ __host__ int get_depth(size_t n, int shift);
__device__ bool get_config(size_t n, size_t& loops, size_t& stride, size_t& reminder, int min_bits, int shift);
__host__ bool sieve_sort_cuda(uint32_t* a, size_t n);
