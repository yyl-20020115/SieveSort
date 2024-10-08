#include "SieveSort.h"

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
int get_top_bit_index(size_t n) {
	int c = (int)_lzcnt_u64(n);
	return n == 0ULL ? 0 : 64 - c;
}

int get_depth(size_t n, int shift) {
	int c = 0;
	for (int t = 0; ; t += shift) {
		n >>= shift;
		c++;
		if (n == 0ULL) break;
	}
	return c;
}