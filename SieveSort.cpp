#include <iostream>
#include <immintrin.h>
#include <algorithm>
#include <iomanip>
#include <chrono>

const __m256i _zero = _mm256_set1_epi16(0);

__forceinline void transpose(uint32_t* a, size_t n, size_t m) {
	for (size_t i = 0; i < n; i++) {
		for (size_t j = i; j < n; j++) {
			std::swap(a[i * n + j], a[j * n + i]);
		}
	}
}
__forceinline int get_lsb_index(__mmask16 mask) {
	return _tzcnt_u16(mask) & 0x0f;
}
__forceinline __mmask16 single_bit(int leading_or_trailing, __mmask16 old_mask, __mmask16 mask) {
	if (mask == 0) return mask;
	unsigned short lz = __lzcnt16(mask);
	unsigned short tz = _tzcnt_u16(mask);
	unsigned short pc = __popcnt16(mask);
	__mmask16 and_mask = mask & old_mask;
	unsigned short ac = __popcnt16(and_mask);
	if (ac > 1) {
		__mmask16 cover = ~((leading_or_trailing
			? ((lz == 0 || lz >= 16) ? mask : (1 << (15 - lz)))
			: ((tz == 0 || tz >= 16) ? mask : (1 << tz))
			));
		mask &= cover;
		lz = __lzcnt16(mask);
		tz = _tzcnt_u16(mask);
		pc = __popcnt16(mask);

	}
	switch (pc) {
	case 0:
		//return mask;
		break;
	case 1:
		mask = (lz == 0 || lz >= 16) ? mask : (1 << (15 - lz));
		break;
	default:
		//count of 1 >=2
		mask = (leading_or_trailing
			? ((lz == 0 || lz >= 16) ? mask : (1 << (15 - lz)))
			: ((tz == 0 || tz >= 16) ? mask : (1 << tz))
			);
		break;
	}
	return mask;
}
__forceinline __mmask8 single_bit(int leading_or_trailing, __mmask8 old_mask, __mmask8 mask) {
	if (mask == 0) return mask;
	unsigned short lz = __lzcnt16(mask);
	unsigned short tz = _tzcnt_u16(mask);
	unsigned short pc = __popcnt16(mask);
	__mmask8 and_mask = mask & old_mask;
	unsigned short ac = __popcnt16(and_mask);
	if (ac > 1) {
		__mmask8 cover = ~((leading_or_trailing
			? ((lz == 0 || lz >= 8) ? mask : (1 << (8 - lz)))
			: ((tz == 0 || tz >= 8) ? mask : (1 << tz))
			));
		mask &= cover;
		lz = __lzcnt16(mask);
		tz = _tzcnt_u16(mask);
		pc = __popcnt16(mask);
	}

	switch (pc) {
	case 0:
		break;
	case 1:
		mask = (lz == 0 || lz >= 8) ? mask : (1 << (8 - lz));
		break;
	default:
		//count of 1 >=2
		mask = leading_or_trailing
			? ((lz == 0 || lz >= 8) ? mask : (1 << (8 - lz)))
			: ((tz == 0 || tz >= 8) ? mask : (1 << tz))
			;
		break;
	}

	return mask;
}

__m512i sieve_sort32x16(__m512i a, uint32_t* result = nullptr) {
	uint32_t buffer[16] = { 0 };
	if (result == nullptr) _mm512_store_epi32(result = buffer, a);
	__mmask16 mask = 0xffff;
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[15] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[0] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[14] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[1] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[13] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[2] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[12] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[3] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[11] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[4] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[10] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[5] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[9] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[6] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[8] = _mm512_mask_reduce_max_epu32(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(result[7] = _mm512_mask_reduce_min_epu32(mask, a))))
		);
	return _mm512_loadu_epi32(result);
}
__forceinline bool sieve_get_min(__mmask16 mask, __m512i a, uint32_t* p_min, __mmask16* p_mask_min) {
	if (mask != 0) {
		*p_min = _mm512_mask_reduce_min_epu32(mask, a);
		*p_mask_min = _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(*p_min));
		return true;
	}
	return false;
}

__forceinline bool sieve_get_max(__mmask16 mask, __m512i a, uint32_t* p_max, __mmask16* p_mask_max) {
	if (mask != 0) {
		*p_max = _mm512_mask_reduce_max_epu32(mask, a);
		*p_mask_max = _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(*p_max));
		return true;
	}
	return false;
}
__forceinline bool sieve_get_max_min(__mmask16 mask, __m512i a, uint32_t* p_max, uint32_t* p_min, __mmask16* p_mask_max, __mmask16* p_mask_min) {
	if (mask != 0) {
		*p_max = _mm512_mask_reduce_max_epu32(mask, a);
		*p_min = _mm512_mask_reduce_min_epu32(mask, a);
		*p_mask_max = _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(*p_max));
		*p_mask_min = _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(*p_min));
		return true;
	}
	return false;
}

int seive_get_min(uint32_t* p_min, __mmask16* p_all_masks, __mmask16 masks[16], __m512i values[16]) {
	if (p_all_masks == 0 || *p_all_masks==0) return 0;
	int count = 0;
	uint32_t  _mines[16] = { 0 };
	__mmask16 mask_mines[16] = { 0 };
	for (size_t i = 0; i < 16; i++) {
		if ((*p_all_masks & (1 << i)) != 0) {
			if (sieve_get_min(masks[i],
				values[i],
				_mines + i, mask_mines + i))
			{
				//OK
			}
		}
	}
	__mmask16 found_mask = 0;
	if (sieve_get_min(*p_all_masks, _mm512_loadu_epi32(_mines), p_min, &found_mask)){
		//update masks
		while (found_mask != 0) {
			int idx = get_lsb_index(found_mask);
			count += __popcnt16(mask_mines[idx]);
			masks[idx] &= ~mask_mines[idx];
			found_mask &= ~(1 << idx);
		}
		*p_all_masks &= ~_mm256_cmpeq_epu16_mask(
			_mm256_loadu_epi16(masks), _zero);
	}
	
	return count;
}
//[16u32]x16
void sieve_sort_256(uint32_t a[256]) {
	__m512i values[16];
	for (size_t i = 0; i < 16; i++) {
		values[i] = _mm512_loadu_epi32(a + (i<<4));
	}
	__mmask16 masks[16];
	memset(masks, 0xff, sizeof(masks));
	__mmask16 all_masks = 0xffff;
	int p = 0;
	while(p<256) {
		uint32_t _min = 0;
		int count = seive_get_min(&_min, &all_masks, masks, values);
		if (count == 0)break;
		for (int i = 0; i < count;i++) {
			a[p++] = _min;
		}
	}
}


__m512i sieve_sort64x8(__m512i a, uint64_t* result = nullptr) {
	uint64_t buffer[8] = { 0 };
	if (result == nullptr) _mm512_store_epi64(result = buffer, a);
	__mmask8 mask = ~0;
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[7] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[0] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[6] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[1] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[5] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[2] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[4] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_cmpeq_epi64_mask(a, _mm512_set1_epi64(result[3] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	return _mm512_loadu_epi64(result);
}

const __m512i zero = _mm512_setzero_si512();
const __m512i ones = _mm512_set1_epi32(1);
const __m512i mones = _mm512_set1_epi32(~0);
const __m512i hexes = _mm512_set1_epi32(16);
const __m512i sequence = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

__forceinline void sieve_sort_base16(uint32_t* a, size_t n, bool flip = false) {
	if (flip) {
		__m512i idx = sequence;
		for (size_t i = 0; i < n; i++) {
			__m512i values = _mm512_i32gather_epi32(idx, a, sizeof(uint32_t));
			_mm512_i32scatter_epi32(a, idx,
				sieve_sort32x16(values), sizeof(uint32_t));
			idx = _mm512_add_epi32(idx, hexes);
		}
	}
	else {
		for (size_t i = 0; i < n; i++) {
			sieve_sort32x16(_mm512_loadu_epi32(a + (i << 4)), a + (i << 4));
		}
	}
}


//sort the data from 2d to liner 

__forceinline bool square_sort_base16(uint32_t* buffer, uint32_t* a, size_t count, size_t total) {
	//a[256]:a[16][16]
	const size_t stride = 16;
	if (total > stride * count) return false;
	size_t k = 0;
	__m512i indices = zero, top_indices = zero;
	for (size_t i = 0; i < stride; i++) {
		indices.m512i_u32[i] = i;// (uint32_t)(i * count);
		top_indices.m512i_u32[i] = i + stride * (count - 1);// (uint32_t)((i + 1) * count);
	}
	__mmask16 mask = 0xffff;
	while (k < total) {
		__m512i values = _mm512_mask_i32gather_epi32(mones, mask, indices, a, sizeof(uint32_t));
		uint32_t _min = _mm512_mask_reduce_min_epu32(mask, values);
		__mmask16 locations = _mm512_cmpeq_epu32_mask(values, _mm512_set1_epi32(_min));
		indices = _mm512_mask_add_epi32(indices, locations, indices, hexes);
		mask &= ~_mm512_mask_cmpeq_epu32_mask(mask, indices, top_indices);
		//while (locations != 0) {
		//	int idx = get_lsb_index(locations);
		//	buffer[k++] = _min;
		//	locations &= ~(1 << idx);
		//}
		unsigned short cbits = __popcnt16(locations);
		for (uint16_t c = 0; c < cbits; c++) {
			buffer[k++] = _min;
		}
	}
	return true;
}
bool sieve_sort_base256(uint32_t* a, size_t n) {
	const int _base = 256;
	bool done = false;
	if (n <= 1) return true;
	else if (n > _base) return false;
	else if (n == _base) {
		uint32_t test[_base] = { 0 };
		memcpy(test, a, sizeof(test));
		std::sort(test, test + _base);
		sieve_sort_base16(a, n >> 4, false);
		sieve_sort_base16(a, n >> 4, true);
		uint32_t result[_base] = { 0 };
		if (done = square_sort_base16(result, a, n >> 4, n)) {
			memcpy(a, result, sizeof(result));
		}
		bool beq = std::equal(test, test + _base, result);
		if (!beq) {
			for (int i = 0; i < _base; i++) {
				if (test[i] != result[i]) {
					std::cout << "found" << std::endl;
				}
			}
		}

	}
	else //n<_base
	{
		size_t c = n >> 8;
		size_t r = n & 0xffULL;
		c = r != 0 ? c + 1 : c;

		size_t m = c << 8;
		uint32_t* _a = new uint32_t[m];
		uint32_t* _r = new uint32_t[m];
		memset(_r, 0, sizeof(uint32_t) * m);
		memset(_a, 0, sizeof(uint32_t) * n);
		memcpy(_a + (m - n), a, sizeof(uint32_t) * n);

		sieve_sort_base16(_a, m >> 4, false);
		done = square_sort_base16(_r, _a, m >> 4, m);
		delete[] _a;
		if (done) {
			memcpy(a, _r + (m - n), sizeof(uint32_t) * n);
		}
		delete[] _r;
	}
	return done;
}

void compose_sort(uint32_t* result, uint32_t* a, size_t w, size_t h) {
	for (size_t i = 0; i < h; i++) {
		std::sort(a + i * w, a + (i + 1) * w);
	}
	square_sort_base16(result, a, w * h >> 4, w * h);
}

__forceinline uint32_t generate_random_32() {
	int r;
	uint64_t v0;
	r = _rdrand64_step(&v0);
	return ((v0 >> 32) & 0xffffffff ^ (v0 & 0xffffffff));
}
__forceinline uint64_t generate_random_64() {
	int r;
	uint64_t v0;
	r = _rdrand64_step(&v0);
	return v0;
}
__forceinline uint32_t generate_random_32(uint32_t range) {
	return (uint32_t)((generate_random_32() / (double)(~0U)) * range);
}
__forceinline uint64_t generate_random_64(uint64_t range) {
	return (uint64_t)((generate_random_64() / (double)(~0ULL)) * range);
}

void print1d(uint32_t a[], int m) {
	for (int i = 0; i < m; i++) {
		std::cout << a[i];
		if (i < m - 1) std::cout << " ";
	}
	std::cout << std::endl;
}
void print2d(uint32_t a[], int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << "\t" << a[i * n + j];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
void tests() {
	const int max_retries = 100000;
	uint64_t result64x8[8] = { 0 };
	uint64_t compare64x8[8] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < 8; i++) {
			compare64x8[i] = result64x8[i] = generate_random_64();
		}
		sieve_sort64x8(_mm512_loadu_epi64(result64x8), result64x8);
		std::sort(compare64x8, compare64x8 + 8);
		bool ex = std::equal(compare64x8, result64x8 + 16, compare64x8);
		if (!ex)
		{
			std::cout << "failed" << std::endl;
		}
	}
	uint32_t result32x16[16] = { 0 };
	uint32_t compare32x16[16] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < 16; i++) {
			compare32x16[i] = result32x16[i] = generate_random_32();
		}
		sieve_sort32x16(_mm512_loadu_epi32(result32x16), result32x16);
		std::sort(compare32x16, compare32x16 + 16);
		bool ex = std::equal(result32x16, result32x16 + 16, compare32x16);
		if (!ex)
		{
			std::cout << "failed" << std::endl;
			for (int i = 0; i < 16; i++) {
				if (compare32x16[i] != result32x16[i]) {
					std::cout << i << std::endl;
				}
			}
		}
	}
}



int main(int argc, char* argv[])
{
#if 1
	const int _length = 256;

	uint32_t a[_length], b[_length], c[_length];
	for (int t = 0; t < _length; t++) {
		//uint32_t result32[count];
		c[t] = b[t] = a[t] = generate_random_32(_length);
	}
	//print1d(a, _length);

	sieve_sort_256(a);
	print1d(a, _length);

	std::sort(c, c + _length);
	print1d(c, _length);
	std::cout << std::endl;
#endif
#if 0
	const int _64K = 65536;
	const int _test_count = 10;
	uint32_t* _a = new uint32_t[_64K];
	uint32_t* _b = new uint32_t[_64K];
	uint32_t* _c = new uint32_t[_64K];
	for (int c = 0; c < _64K; c++) {
		_a[c] = _b[c] = generate_random_32(_64K);
	}

	auto start = std::chrono::high_resolution_clock::now();

	for (int c = 0; c < _test_count; c++) {
		compose_sort(_b, _a, 16, 4096);
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed4 = end - start;
	double d4 = ((double)_test_count / elapsed4.count()) / 1000.0;
	std::cout << "compose sort speed:" << d4 << "K/s" << std::endl;

	start = std::chrono::high_resolution_clock::now();

	for (int c = 0; c < _test_count; c++) {
		std::sort(_c, _c + 256);
	}

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed3 = end - start;

	double d3 = ((double)_test_count / elapsed3.count()) / 1000.0;
	std::cout << "std sort speed:" << d3 << "K/s" << std::endl;


	return 0;
#endif
	const int count = 256;
	const int max_repeats = 10000;
	uint32_t** values = new uint32_t * [max_repeats];

	for (int c = 0; c < max_repeats; c++) {
		//uint32_t result32[count];
		values[c] = new uint32_t[count];
		for (int i = 0; i < count; i++) {
			//result32[i] = (256 - i);
			values[c][i] = generate_random_32(count);
		}
	}

	//ok for 16x

	auto start = std::chrono::high_resolution_clock::now();

	for (int c = 0; c < max_repeats; c++) {
		uint32_t result[count], compare[count];
		memcpy(result, values[c], sizeof(result));
		memcpy(compare, values[c], sizeof(compare));
		sieve_sort_256(result);
		std::sort(compare, compare + count);
		bool beq = std::equal(result, result + count, compare);
		if (!beq) {
			int bad = 0;
			std::cout << "failed c=" << c << std::endl;
			for (int j = 0; j < count; j++) {
				if (result[j] != compare[j]) {
					std::cout << "found failed:" << j << std::endl;
					bad++;
				}
			}
			std::cout << "bad=" << bad << std::endl;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	double d1 = ((double)max_repeats / elapsed.count()) / 1000.0;
	std::cout << "sieve sort speed:" << d1 << "K/s" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		std::sort(values[c], values[c] + count);
	}

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed2 = end - start;
	double d2 = ((double)max_repeats / elapsed2.count() / (1000.0));
	std::cout << "std sort speed:  " << d2 << "K/s" << std::endl;

	for (int c = 0; c < max_repeats; c++) {
		//uint32_t result32[count];
		delete[] values[c];
	}
	delete[] values;

	std::cout << "ratio:" << (d1 / d2 * 100.0) << "%" << std::endl;

	std::cout << "1/r:" << (d2 / d1) << std::endl;

	return 0;
}

