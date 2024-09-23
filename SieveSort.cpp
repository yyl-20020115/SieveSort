#include <iostream>
#include <immintrin.h>
#include <algorithm>
#include <iomanip>
#include <chrono>

const __m256i _zero = _mm256_setzero_si256();
const __m512i zero = _mm512_setzero_si512();
const __m512i ones = _mm512_set1_epi32(1);

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

__forceinline bool sieve_get_min(__mmask16 mask, __m512i a, uint32_t& _min, __mmask16& _mask_min) {
	if (mask != 0) {
		_mask_min = _mm512_mask_cmpeq_epu32_mask(mask, a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask, a)));
		return true;
	}
	return false;
}
__forceinline int sieve_get_min_index(__mmask16 mask, uint32_t& _min, __m512i a) {
	return get_lsb_index(
		_mm512_mask_cmpeq_epu32_mask(mask, a, _mm512_set1_epi32(
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
		_mask_max = _mm512_mask_cmpeq_epu32_mask(mask, a, _mm512_set1_epi32(
			_max = _mm512_mask_reduce_max_epu32(mask, a)));
		_mask_min = _mm512_mask_cmpeq_epu32_mask(mask & (~_mask_max), a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask & (~_mask_max), a)));
		return true;
	}
	return false;
}
__forceinline bool sieve_get_min_max(__mmask8 mask, __m512i a, uint64_t& _min, uint64_t& _max, __mmask8& _mask_min, __mmask8& _mask_max) {
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
	uint32_t i = 0, j = 16;
	uint32_t _min = 0, _max = 0;
	uint32_t c_min = 0, c_max = 0;
	while (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

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
	__mmask8 mask = 0xff;
	__mmask8 _min_mask = 0, _max_mask = 0;
	uint32_t i = 0, j = 8;
	uint64_t _min = 0, _max = 0;
	uint32_t c_min = 0, c_max = 0;
	while (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

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

__forceinline __m512i sieve_sort32x16(__m512i a, uint32_t* result = nullptr) {
	__m512i target = zero;
	__mmask16 mask = 0xffff;
	__mmask16 _min_mask = 0, _max_mask = 0;
	uint32_t i = 0, j = 16;
	uint32_t _min = 0, _max = 0;
	uint32_t c_min = 0, c_max = 0;
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi32((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi32(_min));

		target = _mm512_mask_blend_epi32((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

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
__forceinline __m512i sieve_sort64x8(__m512i a, uint64_t* result = nullptr) {
	__m512i target = zero;
	__mmask8 mask = 0xff;
	__mmask8 _min_mask = 0, _max_mask = 0;
	uint32_t i = 0, j = 8;
	uint64_t _min = 0, _max = 0;
	uint32_t c_min = 0, c_max = 0;
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi64((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi64(_min));

		target = _mm512_mask_blend_epi64((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi64(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi64((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi64(_min));

		target = _mm512_mask_blend_epi64((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi64(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

		target = _mm512_mask_blend_epi64((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm512_set1_epi64(_min));

		target = _mm512_mask_blend_epi64((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm512_set1_epi64(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	if (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);

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
__forceinline bool seive_get_min_max(
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
	if (_all_masks == 0) return false;
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
	return true;
}

const size_t _256 = 1 << 8;
//[16u32]x16
bool sieve_sort_256_dual(uint32_t* a/*[256]*/, size_t n, uint32_t* result = nullptr) {
	if (n != _256) return false;
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
	return true;
}

bool sieve_sort_256(uint32_t* a/*[_256]*/, size_t n, uint32_t* result = nullptr) {
	if (n != _256) return false;
	__m512i values[16];
	for (size_t i = 0; i < 16; i++) {
		values[i] = _mm512_loadu_epi32(a + (i << 4));
	}
	__mmask16 masks[16];
	memset(masks, 0xff, sizeof(masks));
	__mmask16 all_masks = 0xffff;
	result = result == nullptr ? a : result;
	int p = 0, count = 0;
	uint32_t _min = 0;
	while (p < n) {
		if (0 == (count = seive_get_min(_min, all_masks, masks, values))) break;
		for (int i = 0; i < count; i++) {
			result[p++] = _min;
		}
	}
	return true;
}

bool sieve_collect(size_t n, uint32_t* b, uint32_t* result) {
	//N=4K, cb=12
	//N=64K,cb = 16
	//N >0 and it has to be 2^n
	if (__popcnt64(n) != 1) return false;

	int cb = (64 - (int)__lzcnt64(n)) - 1;
	if (cb < 4) return false;

	__m512i idx = zero;
	__m512i top = zero;

	for (int i = 0; i < 16; i++) {
		idx.m512i_u32[i] = i << (cb - 4);
		top.m512i_u32[i] = (i + 1) << (cb - 4);
	}

	__mmask16 mask = 0x0ffff, _mask_min = 0;
	int pc = 0, i = 0;
	uint32_t _min = 0;
	while (i < n) {
		__m512i values = _mm512_mask_i32gather_epi32(zero, mask, idx, b, sizeof(uint32_t));
		if (!sieve_get_min(mask, values, _min, _mask_min)) break;
		idx = _mm512_mask_add_epi32(idx, _mask_min, idx, ones);
		mask &= (~_mm512_mask_cmpeq_epu32_mask(_mask_min, idx, top));
		pc = __popcnt16(_mask_min);
		for (int j = 0; j < pc; j++) {
			result[i++] = _min;
		}
		if (mask == 0) break;
	}
	return true;
}
const size_t _4K = _256 << 4;
bool sieve_sort_4K(uint32_t* a/*[_4K]*/, size_t n, uint32_t* result = nullptr, int omp_depth = 1) {
	if (n != _4K) return false;
	uint32_t* b = new uint32_t[n];
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 8);
			uint32_t* pb = b + ((size_t)i << 8);
			sieve_sort_256(pa, n >> 4, pb);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 8);
			uint32_t* pb = b + ((size_t)i << 8);
			sieve_sort_256(pa, n >> 4, pb);
		}
	}
	sieve_collect(n, b, result == nullptr ? a : result);
	delete[] b;
	return true;
}
const size_t _64K = _4K << 4;
bool sieve_sort_64K(uint32_t* a/*[_64K]*/, size_t n, uint32_t* result, int omp_depth = 2) {
	if (n != _64K) return false;
	uint32_t* b = new uint32_t[n];
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 12);
			uint32_t* pb = b + ((size_t)i << 12);
			sieve_sort_4K(pa, n >> 4, pb, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 12);
			uint32_t* pb = b + ((size_t)i << 12);
			sieve_sort_4K(pa, n >> 4, pb, omp_depth - 1);
		}
	}

	sieve_collect(n, b, result == nullptr ? a : result);
	delete[] b;
	return true;
}


const size_t _1M = _64K << 4;
bool sieve_sort_1M(uint32_t* a/*[_1M]*/, size_t n, uint32_t* result, int omp_depth = 3) {
	if (n != _1M) return false;
	uint32_t* b = new uint32_t[n];
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 16);
			uint32_t* pb = b + ((size_t)i << 16);
			sieve_sort_64K(pa, n >> 4, pb, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 16);
			uint32_t* pb = b + ((size_t)i << 16);
			sieve_sort_64K(pa, n >> 4, pb, omp_depth - 1);
		}
	}
	sieve_collect(n, b, result == nullptr ? a : result);

	delete[] b;
	return true;
}
const size_t _16M = _1M << 4;
bool sieve_sort_16M(uint32_t* a/*[_16M]*/, size_t n, uint32_t* result, int omp_depth = 4) {
	if (n != _16M) return false;
	uint32_t* b = new uint32_t[n];
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 20);
			uint32_t* pb = b + ((size_t)i << 20);
			sieve_sort_1M(pa, n >> 4, pb, omp_depth - 1);
		}
	}
	else
	{
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 20);
			uint32_t* pb = b + ((size_t)i << 20);
			sieve_sort_1M(pa, n >> 4, pb, omp_depth - 1);
		}
	}

	sieve_collect(n, b, result == nullptr ? a : result);

	delete[] b;
	return true;
}

const size_t _256M = _16M << 4;
bool sieve_sort_256M(uint32_t* a/*[_256M]*/, size_t n, uint32_t* result, int omp_depth = 5) {
	if (n != _256M) return false;
	uint32_t* b = new uint32_t[n];

	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 24);
			uint32_t* pb = b + ((size_t)i << 24);
			sieve_sort_16M(pa, n >> 4, pb, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 24);
			uint32_t* pb = b + ((size_t)i << 24);
			sieve_sort_16M(pa, n >> 4, pb, omp_depth - 1);
		}
	}

	sieve_collect(n, b, result == nullptr ? a : result);

	delete[] b;
	return true;
}

const size_t _1G = _256M << 4;
bool sieve_sort_1G(uint32_t* a/*[_1G]*/, size_t n, uint32_t* result, int omp_depth = 6) {
	if (n != _1G) return false;
	uint32_t* b = new uint32_t[n];
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 28);
			uint32_t* pb = b + ((size_t)i << 28);
			sieve_sort_256M(pa, n >> 4, pb, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < 16; i++) {
			uint32_t* pa = a + ((size_t)i << 28);
			uint32_t* pb = b + ((size_t)i << 28);
			sieve_sort_256M(pa, n >> 4, pb, omp_depth - 1);
		}
	}

	sieve_collect(n, b, result == nullptr ? a : result);

	delete[] b;
	return true;
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
		//sieve_sort64x8_loop(_mm512_loadu_epi64(result64x8), result64x8);

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
		_mm512_storeu_epi32(result32x16, r);
		std::sort(compare32x16, compare32x16 + 16);
		bool ex = std::equal(result32x16, result32x16 + 16, compare32x16);
		if (!ex)
		{
			std::cout << "failed" << std::endl;
		}
	}
	std::cout << "32 pass" << std::endl;
	std::cout << "all pass" << std::endl;
}



int main(int argc, char* argv[])
{
#if 1
	tests();
#endif
	const int count = _64K;
	const int max_repeats = 100;
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
		sieve_sort_64K(values[c], count, results[c]);
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

