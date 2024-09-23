#include <iostream>
#include <immintrin.h>
#include <algorithm>
#include <iomanip>
#include <chrono>

const __m256i _zero = _mm256_setzero_si256();
const __m512i zero = _mm512_setzero_si512();
const __m512i ones = _mm512_set1_epi32(1);
const __m512i mones = _mm512_set1_epi32(~0);
const __m512i hexes = _mm512_set1_epi32(16);
const __m512i sequence = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

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
	if (mask == 0 || old_mask == 0) return mask;
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
		return mask;
	case 1:
		return mask = (lz == 0 || lz >= 16) ? mask : (1 << (15 - lz));
	default:
		//count of 1 >=2
		return mask = (leading_or_trailing
			? ((lz == 0 || lz >= 16) ? mask : (1 << (15 - lz)))
			: ((tz == 0 || tz >= 16) ? mask : (1 << tz))
			);
	}
}
__forceinline __mmask8 single_bit(int leading_or_trailing, __mmask8 old_mask, __mmask8 mask) {
	if (mask == 0 || old_mask == 0) return mask;
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
		return mask;
	case 1:
		return mask = (lz == 0 || lz >= 8) ? mask : (1 << (8 - lz));
	default:
		//count of 1 >=2
		return mask = leading_or_trailing
			? ((lz == 0 || lz >= 8) ? mask : (1 << (8 - lz)))
			: ((tz == 0 || tz >= 8) ? mask : (1 << tz))
			;
	}
}
__forceinline __m512i sieve_sort32x16(__m512i a, uint32_t* result = nullptr) {
	//TODO:FIX
	uint32_t buffer[16] = { 0 };
	if (result == nullptr) _mm512_store_epi32(result = buffer, a);
	__mmask16 mask = 0xffff;
	__mmask16 c_mask = 0;
	__mmask16 min_mask = 0, max_mask = 0;
	__mmask16 min_single_mask = 0, max_single_mask = 0;

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[15] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[0] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[14] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[1] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[13] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[2] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[12] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[3] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[11] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[4] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[10] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[5] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[9] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[6] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);

	min_single_mask = single_bit(0, mask, max_mask = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(result[8] = _mm512_mask_reduce_max_epu32(mask, a))));
	max_single_mask = single_bit(1, mask, min_mask = _mm512_mask_cmpeq_epi32_mask(mask & ~max_mask, a, _mm512_set1_epi32(result[7] = _mm512_mask_reduce_min_epu32(mask & ~max_mask, a))));
	mask &= ~(min_single_mask | max_single_mask);


	return _mm512_loadu_epi32(result);
}
__forceinline bool sieve_get_min(__mmask16 mask, __m512i a, uint32_t& _min, __mmask16& _mask_min) {
	if (mask != 0) {
		_mask_min = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask, a)));
		return true;
	}
	return false;
}
__forceinline int sieve_get_min_index(__mmask16 mask, uint32_t& _min, __m512i a) {
	return get_lsb_index(
		_mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask, a))));
}
__forceinline int sieve_get_min_index(__mmask16 mask, uint32_t& _min, uint32_t a[16]) {
	return sieve_get_min_index(mask, _min, _mm512_loadu_epi32(a));
}
__forceinline bool sieve_get_min(__mmask16 mask, uint32_t a[16], uint32_t& _min, __mmask16& _mask_min) {
	return sieve_get_min(mask, _mm512_loadu_epi32(a), _min, _mask_min);
}
__forceinline bool sieve_get_max(__mmask16 mask, __m512i a, uint32_t& _max, __mmask16& _mask_max) {
	if (mask != 0) {
		_mask_max = _mm512_cmpeq_epi32_mask(a, _mm512_set1_epi32(
			_max = _mm512_mask_reduce_max_epu32(mask, a)));
		return true;
	}
	return false;
}
__forceinline bool sieve_get_min_max(__mmask16 mask, __m512i a, uint32_t& _min, uint32_t& _max, __mmask16& _mask_min, __mmask16& _mask_max) {
	if (mask != 0) {
		_mask_max = _mm512_mask_cmpeq_epi32_mask(mask, a, _mm512_set1_epi32(
			_max = _mm512_mask_reduce_max_epu32(mask, a)));
		_mask_min = _mm512_mask_cmpeq_epi32_mask(mask & (~_mask_max), a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask & (~_mask_max), a)));
		return true;
	}
	return false;
}
__forceinline bool sieve_get_min_max(__mmask8 mask, __m512i a, uint32_t& _min, uint32_t& _max, __mmask8& _mask_min, __mmask8& _mask_max) {
	if (mask != 0) {
		_mask_max = _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(
			_max = _mm512_mask_reduce_max_epu64(mask, a)));
		_mask_min = _mm512_mask_cmpeq_epi64_mask(mask & (~_mask_max), a, _mm512_set1_epi64(
			_min = _mm512_mask_reduce_min_epu64(mask & (~_mask_max), a)));
		return true;
	}
	return false;
}

__forceinline __m512i sieve_sort32x16_loop(__m512i a, uint32_t* result = nullptr) {

	__m512i target = zero;
	__mmask16 mask = 0xffff;
	__mmask16 _min_mask = 0, _max_mask = 0;
	uint32_t _min = 0, _max = 0;

	int i = 0, j = 16;
	while (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		int c_min = __popcnt16(_min_mask);
		int c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (result != nullptr) {
		_mm512_storeu_epi32(result, target);
	}
	return target;
}
__forceinline __m512i sieve_sort64x8_loop(__m512i a, uint64_t* result = nullptr) {

	__m512i target = zero;
	__mmask8 mask = 0xffff;
	__mmask8 _min_mask = 0, _max_mask = 0;
	uint32_t _min = 0, _max = 0;

	int i = 0, j = 8;
	while (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		int c_min = __popcnt16(_min_mask);
		int c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi64((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi64(_min));

		target = _mm512_mask_blend_epi64((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi64(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (result != nullptr) {
		_mm512_storeu_epi64(result, target);
	}
	return target;
}

__forceinline int seive_get_min(uint32_t& p_min, __mmask16& _all_masks, __mmask16 masks[16], __m512i values[16]) {
	if (_all_masks == 0) return 0;
	int count = 0;
	uint32_t  _mines[16] = { 0 };
	__mmask16 mask_mines[16] = { 0 };
	for (size_t i = 0; i < 16; i++) {
		if ((_all_masks & (1 << i)) != 0) {
			if (sieve_get_min(masks[i],
				values[i],
				_mines[i], mask_mines[i]))
			{
				//OK
			}
		}
	}
	__mmask16 found_mask = 0;
	if (sieve_get_min(_all_masks, _mm512_loadu_epi32(_mines), p_min, found_mask)) {
		__m256i __mask_mines = _mm256_loadu_epi16(mask_mines);
		count = _mm512_reduce_add_epu16(_mm512_castsi256_si512(
			_mm256_maskz_popcnt_epi16(found_mask, __mask_mines)));
		__m512i _mask_mines = _mm512_cvtepu16_epi32(__mask_mines);
		__m512i _masks = _mm512_cvtepu16_epi32(_mm256_loadu_epi16(masks));
		_masks = _mm512_mask_andnot_epi32(_masks, found_mask, _mask_mines, _masks);
		_all_masks &= ~_mm512_cmpeq_epu32_mask(
			_masks, zero);
		_mm256_storeu_epi16(masks, _mm512_cvtepi32_epi16(_masks));
	}

	return count;
}
__forceinline int seive_get_min_index(uint32_t& p_min, uint32_t values[256]) {
	int index = -1;
	uint32_t  _mines[16] = { 0 };
	__mmask16 mask_mines[16] = { 0 };
	for (size_t i = 0; i < 16; i++) {
		if (sieve_get_min(0xffff,
			values + (i << 4),
			_mines[i], mask_mines[i]))
		{
			//OK
		}
	}
	__mmask16 found_mask = 0;
	if (sieve_get_min(0xffff, _mm512_loadu_epi32(_mines), p_min, found_mask)) {
		int p = get_lsb_index(found_mask);
		index = (p << 4) + get_lsb_index(mask_mines[p]);
	}
	return index;
}
__forceinline void seive_get_min_max(
	int use_mask,
	uint32_t& p_min,
	uint32_t& p_max,
	uint32_t& _min_count,
	uint32_t& _max_count,
	__mmask16& _all_masks,
	__mmask16 masks[16],
	__m512i values[16]) {

	_min_count = 0;
	_max_count = 0;
	if (_all_masks == 0) return;
	uint32_t  _mines[16] = { 0 };
	__mmask16 mask_mines[16] = { 0 };
	uint32_t  _maxes[16] = { 0 };
	__mmask16 mask_maxes[16] = { 0 };
	for (int i = 0; i < 16; i++) {
		if ((_all_masks & (1 << i)) != 0) {
			if (sieve_get_min_max(
				masks[i],
				values[i],
				_mines[i], _maxes[i], mask_mines[i], mask_maxes[i]))
			{
				//OK
			}
		}
	}
	__mmask16 found_mask = 0;
	if ((use_mask & 1) != 0 && sieve_get_min(_all_masks, _mm512_loadu_epi32(_mines), p_min, found_mask)) {
		__m256i __mask_mines = _mm256_loadu_epi16(mask_mines);
		_min_count = _mm512_reduce_add_epu16(_mm512_castsi256_si512(
			_mm256_maskz_popcnt_epi16(found_mask, __mask_mines)));
		__m512i _mask_mines = _mm512_cvtepu16_epi32(__mask_mines);
		__m512i _masks = _mm512_cvtepu16_epi32(_mm256_loadu_epi16(masks));
		_masks = _mm512_mask_andnot_epi32(_masks, found_mask, _mask_mines, _masks);
		_all_masks &= ~_mm512_cmpeq_epu32_mask(
			_masks, zero);
		_mm256_storeu_epi16(masks, _mm512_cvtepi32_epi16(_masks));
	}
	if ((use_mask & 2) != 0 && sieve_get_max(_all_masks, _mm512_loadu_epi32(_maxes), p_max, found_mask)) {
		__m256i __mask_maxes = _mm256_loadu_epi16(mask_maxes);
		_max_count = _mm512_reduce_add_epu16(_mm512_castsi256_si512(
			_mm256_maskz_popcnt_epi16(found_mask, __mask_maxes)));
		__m512i _mask_maxes = _mm512_cvtepu16_epi32(__mask_maxes);
		__m512i _masks = _mm512_cvtepu16_epi32(_mm256_loadu_epi16(masks));
		_masks = _mm512_mask_andnot_epi32(_masks, found_mask, _mask_maxes, _masks);
		_all_masks &= ~_mm512_cmpeq_epu32_mask(
			_masks, zero);
		_mm256_storeu_epi16(masks, _mm512_cvtepi32_epi16(_masks));
	}
}

const size_t _256 = 1 << 8;
//[16u32]x16
void sieve_sort_256_dual(uint32_t a[256], uint32_t* result = nullptr) {
	__m512i values[16];
	for (size_t i = 0; i < 16; i++) {
		values[i] = _mm512_loadu_epi32(a + (i << 4));
	}
	__mmask16 masks[16];
	memset(masks, 0xff, sizeof(masks));

	result = result == nullptr ? a : result;

	__mmask16 all_masks = 0xffff;
	uint32_t _min = 0, _max = 0;
	uint32_t _min_count = 0, _max_count = 0;
	int i = 0, j = 255;
	while (i <= j) {
		seive_get_min_max(
			3,
			_min, _max,
			_min_count, _max_count,
			all_masks, masks, values);

		for (size_t t = 0; t < _min_count; t++) {
			result[i++] = _min;
		}
		for (size_t t = 0; t < _max_count; t++) {
			result[j--] = _max;
		}
	}
}

void sieve_sort_256(uint32_t a[_256], uint32_t* result = nullptr) {
	__m512i values[16];
	for (size_t i = 0; i < 16; i++) {
		values[i] = _mm512_loadu_epi32(a + (i << 4));
	}
	__mmask16 masks[16];
	memset(masks, 0xff, sizeof(masks));
	__mmask16 all_masks = 0xffff;
	result = result == nullptr ? a : result;

	int p = 0;
	while (p < _256) {
		uint32_t _min = 0;
		int count = seive_get_min(_min, all_masks, masks, values);
		if (count == 0) break;
		for (int i = 0; i < count; i++) {
			result[p++] = _min;
		}
	}
}

const size_t _4K = _256 << 4;
void sieve_sort_4K(uint32_t result[_4K], uint32_t a[_4K], int omp_depth = 1) {
	__m512i idx = zero;
	for (int i = 0; i < 16; i++) {
		idx.m512i_u32[i] = i << 8;
	}
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 8);
			sieve_sort_256_dual(pa);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 8);
			sieve_sort_256_dual(pa);
		}
	}

	__mmask16 mask = 0x0ffff;

	for (int i = 0; i < _4K; i++) {
		__m512i values = _mm512_mask_i32gather_epi32(zero, mask, idx, a, sizeof(uint32_t));
		int p = sieve_get_min_index(mask, result[i], values);
		idx.m512i_u32[p]++;
		if ((idx.m512i_u32[p] & 0xff) == 0) {
			mask &= ~(1 << p);
		}
	}
}
const size_t _64K = _4K << 4;
void sieve_sort_64K(uint32_t result[_64K], uint32_t a[_64K], int omp_depth = 2) {
	uint32_t idx[16] = { 0 };
	uint32_t** lines = new uint32_t * [16];
	for (int i = 0; i < 16; i++) {
		lines[i] = new uint32_t[_4K];
	}
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 12);
			sieve_sort_4K(lines[i], pa, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 12);
			sieve_sort_4K(lines[i], pa, 0);
		}
	}

	__mmask16 mask = 0x0ffff;

	__m512i values = zero;
	for (int i = 0; i < _64K; i++) {
		for (int j = 0; j < 16; j++) {
			if ((mask & (1 << j)) != 0) {
				values.m512i_u32[j] = lines[j][idx[j]];
			}
		}
		int p = sieve_get_min_index(mask, result[i], values);
		idx[p]++;
		if (idx[p] == _4K) {
			mask &= ~(1 << p);
		}
	}
	for (int i = 0; i < 16; i++) {
		delete[] lines[i];
	}
	delete[] lines;
}


const size_t _1M = _64K << 4;
void sieve_sort_1M(uint32_t result[_1M], uint32_t a[_1M], int omp_depth = 3) {
	uint32_t idx[16] = { 0 };
	uint32_t** lines = new uint32_t * [16];
	for (int i = 0; i < 16; i++) {
		lines[i] = new uint32_t[_64K];
	}
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 16);
			sieve_sort_64K(lines[i], pa, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 16);
			sieve_sort_64K(lines[i], pa, 0);
		}
	}

	__mmask16 mask = 0x0ffff;

	uint32_t _min = 0;
	__m512i values = zero;
	for (int i = 0; i < _1M; i++) {
		for (int j = 0; j < 16; j++) {
			if ((mask & (1 << j)) != 0) {
				values.m512i_u32[j] = lines[j][idx[j]];
			}
		}
		int p = sieve_get_min_index(mask, result[i], values);
		idx[p]++;
		if (idx[p] == _64K) {
			mask &= ~(1 << p);
		}
	}
	for (int i = 0; i < 16; i++) {
		delete[] lines[i];
	}
	delete[] lines;
}
const size_t _16M = _1M << 4;
void sieve_sort_16M(uint32_t result[_16M], uint32_t a[_16M], int omp_depth = 4) {
	uint32_t idx[16] = { 0 };
	uint32_t** lines = new uint32_t * [16];
	for (int i = 0; i < 16; i++) {
		lines[i] = new uint32_t[_1M];
	}
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 20);
			sieve_sort_1M(lines[i], pa, omp_depth - 1);
		}
	}
	else
	{
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 20);
			sieve_sort_1M(lines[i], pa, 0);
		}
	}

	__mmask16 mask = 0x0ffff;

	__m512i values = zero;
	for (int i = 0; i < _16M; i++) {
		for (int j = 0; j < 16; j++) {
			if ((mask & (1 << j)) != 0) {
				values.m512i_u32[j] = lines[j][idx[j]];
			}
		}
		int p = sieve_get_min_index(mask, result[i], values);
		idx[p]++;
		if (idx[p] == _1M) {
			mask &= ~(1 << p);
		}
	}
	for (int i = 0; i < 16; i++) {
		delete[] lines[i];
	}
	delete[] lines;
}

const size_t _256M = _16M << 4;
void sieve_sort_256M(uint32_t result[_256M], uint32_t a[_256M], int omp_depth = 5) {
	uint32_t idx[16] = { 0 };
	uint32_t** lines = new uint32_t * [16];
	for (int i = 0; i < 16; i++) {
		lines[i] = new uint32_t[_16M];
	}
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 24);
			sieve_sort_16M(lines[i], pa, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 24);
			sieve_sort_16M(lines[i], pa, 0);
		}
	}

	__mmask16 mask = 0x0ffff;

	__m512i values = zero;
	for (int i = 0; i < _256M; i++) {
		for (int j = 0; j < 16; j++) {
			if ((mask & (1 << j)) != 0) {
				values.m512i_u32[j] = lines[j][idx[j]];
			}
		}
		int p = sieve_get_min_index(mask, result[i], values);
		idx[p]++;
		if (idx[p] == _16M) {
			mask &= ~(1 << p);
		}
	}
	for (int i = 0; i < 16; i++) {
		delete[] lines[i];
	}
	delete[] lines;
}

const size_t _1G = _256M << 4;
void sieve_sort_1G(uint32_t result[/*_1G*/], uint32_t a[/*_1G*/], int omp_depth = 6) {
	uint32_t idx[16] = { 0 };
	uint32_t** lines = new uint32_t * [16];
	for (int i = 0; i < 16; i++) {
		lines[i] = new uint32_t[_256M];
	}
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 28);
			sieve_sort_256M(lines[i], pa, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 28);
			sieve_sort_256M(lines[i], pa, 0);
		}
	}

	__mmask16 mask = 0x0ffff;

	__m512i values = zero;
	for (int i = 0; i < _256M; i++) {
		for (int j = 0; j < 16; j++) {
			if ((mask & (1 << j)) != 0) {
				values.m512i_u32[j] = lines[j][idx[j]];
			}
		}
		int p = sieve_get_min_index(mask, result[i], values);
		idx[p]++;
		if (idx[p] == _256M) {
			mask &= ~(1 << p);
		}
	}
	for (int i = 0; i < 16; i++) {
		delete[] lines[i];
	}
	delete[] lines;
}

__m512i sieve_sort64x8(__m512i a, uint64_t* result = nullptr) {
	//TODO:
	uint64_t buffer[8] = { 0 };
	if (result == nullptr) _mm512_store_epi64(result = buffer, a);
	__mmask8 mask = ~0;
	mask &= ~(
		single_bit(0, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[7] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[0] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[6] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[1] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[5] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[2] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	mask &= ~(
		single_bit(0, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[4] = _mm512_mask_reduce_max_epu64(mask, a))))
		| single_bit(1, mask, _mm512_mask_cmpeq_epi64_mask(mask, a, _mm512_set1_epi64(result[3] = _mm512_mask_reduce_min_epu64(mask, a))))
		);
	return _mm512_loadu_epi64(result);
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
			compare64x8[i] = result64x8[i] = generate_random_64(16);
		}
		sieve_sort64x8_loop(_mm512_loadu_epi64(result64x8), result64x8);

		std::sort(compare64x8, compare64x8 + 8);
		bool ex = std::equal(compare64x8, result64x8 + 16, compare64x8);
		if (!ex)
		{
			std::cout << "failed" << std::endl;
		}
	}
	std::cout << "64 pass" << std::endl;

	uint32_t original32x16[16] = { 0 };
	uint32_t result32x16[16] = { 0 };
	uint32_t compare32x16[16] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < 16; i++) {
			original32x16[i] = compare32x16[i] = result32x16[i] = generate_random_32(32);
		}
		__m512i t = sieve_sort32x16(_mm512_loadu_epi32(result32x16));
		__m512i r = sieve_sort32x16_loop(_mm512_loadu_epi32(result32x16));
		if (_mm512_cmpeq_epi32_mask(t, r) != 0xffff) {
			//std::cout << "bad" << std::endl;
		}
		_mm512_storeu_epi32(result32x16, r);
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
	std::cout << "32 pass" << std::endl;
	std::cout << "all pass" << std::endl;
}



int main(int argc, char* argv[])
{
	tests();

#if 0
	const int _length = 256;
	for (int s = 0; s < 10000; s++) {
		uint32_t a[_length], b[_length], c[_length];
		for (int t = 0; t < _length; t++) {
			//uint32_t result32[count];
			c[t] = b[t] = a[t] = generate_random_32(_length);
		}
		//print1d(a, _length);
		std::sort(b, b + 256);
		sieve_sort_256(a);
		bool bd = std::equal(a, a + 256, b);
		if (!bd) {
			std::cout << "bad" << std::endl;
			sieve_sort_256(c);
		}
	}
#endif
#if 0
	const int _length = 65536;

	uint32_t a[_length], b[_length], c[_length];
	for (int t = 0; t < _length; t++) {
		//uint32_t result32[count];
		c[t] = b[t] = a[t] = generate_random_32(_length);
	}
	//print1d(a, _length);

	sieve_sort_64K(a);
	print1d(a, _length);

	std::sort(c, c + _length);
	print1d(c, _length);
	std::cout << std::endl;
#endif

#if 0
	const int _length = 4096;

	uint32_t a[_length], b[_length], c[_length];
	for (int t = 0; t < _length; t++) {
		//uint32_t result32[count];
		c[t] = b[t] = a[t] = generate_random_32(_length);
	}
	//print1d(a, _length);

	//sieve_sort_256_dual(a);
	sieve_sort_4K(a);
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
	const int count = _16M;
	const int max_repeats = 10;
	uint32_t** values = new uint32_t * [max_repeats];
	uint32_t** results = new uint32_t * [max_repeats];
#pragma omp parallel for
	for (int c = 0; c < max_repeats; c++) {
		values[c] = new uint32_t[count];
		results[c] = new uint32_t[count];
		memset(results[c], 0, sizeof(uint32_t) * count);
		for (int i = 0; i < count; i++) {
			//result32[i] = (256 - i);
			values[c][i] = generate_random_32(count);
		}
	}

	//ok for 16x
	auto start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		sieve_sort_16M(results[c], values[c]);
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = end - start;
	double d1 = ((double)max_repeats / elapsed1.count()) / 1000.0;
	std::cout << "sieve sort speed:" << d1 << "K/s" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		std::sort(values[c], values[c] + count);
	}

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed2 = end - start;
	double d2 = ((double)max_repeats / elapsed2.count() / (1000.0));
	std::cout << "std sort speed:  " << d2 << "K/s" << std::endl;
#pragma omp parallel for
	for (int c = 0; c < max_repeats; c++) {
		//uint32_t result32[count];
		for (int d = 0; d < count; d++) {
			if (results[c][d] != values[c][d]) {
				std::cout << "found bad value at repeat " << c << " index " << d << std::endl;
			}
		}
		delete[] results[c];
		delete[] values[c];
	}
	delete[] values;
	std::cout << "seive sort test: omp 16 threads" << std::endl;
	std::cout << "samples:" << count << std::endl;
	std::cout << "repeats:" << max_repeats << std::endl;
	std::cout << "t1(seive):" << elapsed1.count() << " s" << std::endl;
	std::cout << "t2(std::):" << elapsed2.count() << " s" << std::endl;
	std::cout << "ratio:" << (d1 / d2 * 100.0) << "%" << std::endl;

	return 0;
}

