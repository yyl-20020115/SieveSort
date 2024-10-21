#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <immintrin.h>
uint32_t generate_random_32(size_t range) {
	uint64_t v0 = 0ULL;
	int r = _rdrand64_step(&v0);
	if (range >= (1ULL << 32)) range = (~0U);
	uint32_t value = ((v0 >> 32) & (~0U)) ^ (v0 & (~0U));
	return range == 0 ? value : (uint32_t)((value / (double)(~0U)) * range);
}
uint64_t generate_random_64(uint64_t range) {
	uint64_t v0 = 0ULL;
	int r = _rdrand64_step(&v0);
	return range == 0ULL ? v0 : (uint64_t)((v0 / (double)(~0ULL)) * range);
}

__device__ __host__ int get_depth(size_t n, int shift) {
	int c = 0;
	for (int t = 0; ; t += shift) {
		n >>= shift;
		c++;
		if (n == 0ULL) break;
	}
	return c;
}
__device__ __host__ bool get_config(size_t n, size_t& loops, size_t& stride, size_t& reminder, int min_bits, int shift_bits) {
	if (n <= ((1ULL) << min_bits)) return false;
	int depths = get_depth(n, shift_bits);
	int max_bits = depths * shift_bits;
	stride = (1ULL) << (max_bits - shift_bits);
	if (stride == n) {
		stride = n >> shift_bits;
		reminder = 0;
	}
	else {
		reminder = n & (~((~0ULL) << (max_bits - shift_bits)));
	}
	loops = (n - reminder) / stride + (reminder > 0);
	return true;
}

