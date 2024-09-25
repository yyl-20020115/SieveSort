#include <iostream>
#include <immintrin.h>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <memory.h>
#include <omp.h>

const size_t _16 = 1 << 4;      //4
const size_t _256 = _16 << 4;   //8
const size_t _4K = _256 << 4;   //12
const size_t _64K = _4K << 4;   //16
const size_t _1M = _64K << 4;   //20
const size_t _16M = _1M << 4;   //24
const size_t _256M = _16M << 4; //28
const size_t _4G = _256M << 4;  //32
const size_t _64G = _4G << 4;   //36
const size_t _256G = _64G << 4; //40

const __m256i _zero = _mm256_setzero_si256();
const __m512i zero = _mm512_setzero_si512();
const __m512i ones = _mm512_set1_epi32(1);
const __m512i ones64 = _mm512_set1_epi64(1);

#ifdef __GNUC__
#define __forceinline 
#endif

__forceinline bool sieve_get_min(__mmask16 mask, __m512i a, uint32_t& _min, __mmask16& _mask_min) {
	if (mask != 0) {
		_mask_min = _mm512_mask_cmpeq_epu32_mask(mask, a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask, a)));
		return true;
	}
	return false;
}
__forceinline bool sieve_get_min(__mmask16 mask, uint32_t a[16], uint32_t& _min, __mmask16& _mask_min) {
	return sieve_get_min(mask, _mm512_loadu_epi32(a), _min, _mask_min);
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
__forceinline __m512i sieve_sort16_32_loop(__m512i a, uint32_t* result = nullptr) {
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
__forceinline __m512i sieve_sort8_64_loop(__m512i a, uint64_t* result = nullptr) {

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

__forceinline __m512i sieve_sort16_32_direct(__m512i a, uint32_t* result = nullptr) {
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
__forceinline __m512i sieve_sort8_64_direct(__m512i a, uint64_t* result = nullptr) {
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

bool sieve_sort_16(uint32_t* a, size_t n, uint32_t* result = nullptr) {
	if (n > _16)
		return false;
	else if (n == 0)
		return true;
	else if (n == 1) {
		if (result != nullptr) result[0] = a[0];
		return true;
	}
	else if (n == 2) {
		uint32_t a0 = a[0], a1 = a[1];
		a[0] = std::min(a0, a1);
		a[1] = std::max(a0, a1);
		if (result != nullptr) {
			result[0] = a[0];
			result[1] = a[1];
		}
		return true;
	}
	else { //2=<n<=16
		uint32_t b[_16];
		memcpy(b, a, sizeof(uint32_t) * n);
		memset(b + n, 0xff, sizeof(uint32_t) * (_16 - n));
		sieve_sort16_32_loop(_mm512_loadu_epi32(b), b);
		memcpy(a, b, sizeof(uint32_t) * n);
		if (result != nullptr) {
			memcpy(result, b, sizeof(uint32_t) * n);
		}
		return true;
	}
}
bool sieve_sort_256(uint32_t* a/*[_256]*/, size_t n, uint32_t* result = nullptr) {
	if (n <= _16)
		return sieve_sort_16(a, n, result);
	else if (n > _256)
		return false;
	else { //16<n<=256
		uint32_t b[_256];
		memcpy(b, a, sizeof(uint32_t) * n);
		memset(b + n, 0xff, sizeof(uint32_t) * (_256 - n));
		__m512i values[16] = { 0 };
		for (size_t i = 0; i < 16; i++) {
			values[i] = _mm512_loadu_epi32(b + (i << 4));
		}
		__mmask16 masks[16];
		memset(masks, 0xff, sizeof(masks));
		__mmask16 all_masks = 0xffff;

		int p = 0, count = 0;
		uint32_t _min = 0;
		while (all_masks != 0 && p < n) {
			if (0 == (count = seive_get_min(_min, all_masks, masks, values)))
				break;
			for (int i = 0; i < count; i++)
				a[p++] = _min;
		}
		if (result != nullptr) {
			memcpy(result, a, sizeof(uint32_t) * n);
		}
		return true;
	}
}

__forceinline int get_top_bit_index(size_t n) {
	int c = (int)_lzcnt_u64(n);
	return n == 0ULL ? 0 : 64 - c;
}
__forceinline bool get_config(size_t n, size_t& loops, size_t& stride, size_t& reminder, __mmask16& mask, bool& flip, int min_bits = 8) {
	if (n < ((1ULL) << min_bits)) return false;
	int top_bits = get_top_bit_index(n);
	int nb_count = (int)__popcnt64(n);
	int all_bits = nb_count == 1 ? (top_bits - 1) : (top_bits & ~0x3) == top_bits ? top_bits : (top_bits + ((n - ((n >> 4) << 4)) != 0ULL));
	int max_bits = ((all_bits >> 2) + ((all_bits & 0x3) != 0)) << 2;

	stride = (1ULL) << (max_bits - 4);
	reminder = n & (~((~0ULL) << (max_bits - 4)));
	loops = (n - reminder) / stride + (reminder > 0);
	mask = ~((~0U) << (loops));

	flip = (max_bits > 12) && (((max_bits >> 2) & 1) == 0);

	return true;
}

bool sieve_collect(size_t n, size_t loops, size_t stride, size_t reminder, __mmask16 mask,
	bool flip,
	uint32_t* source, uint32_t* destination) {
	if (n == 0 || loops == 0 || loops > 16 || mask == 0 || source == nullptr || destination == nullptr)
		return false;
	if (flip) {
		std::swap(source, destination);
	}
	const size_t large_stride_threshold = (1ULL << 24); //(1ULL << 12))
	if (stride <= large_stride_threshold) {
		__m512i idx = zero;
		__m512i top = zero;
		uint32_t p = 0;
		for (int i = 0; i < loops; i++) {
			idx.m512i_u32[i] = p;
			top.m512i_u32[i] = p + (uint32_t)((i == loops - 1 && reminder > 0) ? reminder : stride);
			p += (uint32_t)stride;
		}
		int pc = 0, i = 0;
		uint32_t _min = 0;
		__mmask16 _mask_min = 0;
		while (mask != 0 && i < n) {
			__m512i values = _mm512_mask_i32gather_epi32(zero, mask, idx, source, sizeof(uint32_t));
			if (!sieve_get_min(mask, values, _min, _mask_min)) break;
			idx = _mm512_mask_add_epi32(idx, _mask_min, idx, ones);
			mask &= (~_mm512_mask_cmpeq_epu32_mask(_mask_min, idx, top));
			pc = __popcnt16(_mask_min);
			for (int j = 0; j < pc; j++)
				destination[i++] = _min;
		}
	}
	else {
		__m512i _idx_low_ = zero;
		__m512i _idx_high = zero;
		__m512i top_low_ = zero;
		__m512i top_high = zero;
		size_t loops_low_ = loops >= 8 ? 8 : loops;
		size_t loops_high = loops >= 8 ? (loops - loops_low_) : 0;
		size_t p = 0;
		for (int i = 0; i < loops_low_; i++) {
			_idx_low_.m512i_u64[i] = p;
			top_low_.m512i_u64[i] = p + ((loops_high == 0 && i == loops - 1 && reminder > 0) ? reminder : stride);
			p += stride;
		}
		for (int i = 0; i < loops_high; i++) {
			_idx_high.m512i_u64[i] = p;
			top_high.m512i_u64[i] = p + ((i == loops - 1 && reminder > 0) ? reminder : stride);
			p += stride;
		}

		int pc = 0, i = 0;
		uint32_t _min = 0;
		__mmask8 _mask_low_ = 0;
		__mmask8 _mask_high = 0;
		__mmask8 _mask_min_low_ = 0;
		__mmask8 _mask_min_high = 0;
		__mmask16 _mask_min = 0;

		while (mask != 0 && i < n) {
			_mask_low_ = (__mmask8)(mask & 0xff);
			_mask_high = (__mmask8)(mask >> 8);
			__m256i values_low_ = _mm512_mask_i64gather_epi32(
				_zero, _mask_low_, _idx_low_, source, sizeof(uint32_t));
			__m256i values_high = _mm512_mask_i64gather_epi32(
				_zero, _mask_high, _idx_high, source, sizeof(uint32_t));
			__m512i values = _mm512_inserti64x4(
				_mm512_castsi256_si512(values_low_), values_high, 1);
			if (!sieve_get_min(mask, values, _min, _mask_min)) break;

			_mask_min_low_ = (__mmask8)(_mask_min & 0xff);
			_mask_min_high = (__mmask8)(_mask_min >> 8);

			_idx_low_ = _mm512_mask_add_epi64(_idx_low_, _mask_min_low_, _idx_low_, ones64);
			_idx_high = _mm512_mask_add_epi64(_idx_high, _mask_min_high, _idx_high, ones64);
			_mask_low_ &= _mm512_mask_cmpeq_epu64_mask(_mask_min_low_, _idx_low_, top_low_);
			_mask_high &= _mm512_mask_cmpeq_epu64_mask(_mask_min_high, _idx_high, top_high);

			mask &= ~((((__mmask16)_mask_high) << 8) | (__mmask16)_mask_low_);
			pc = __popcnt16(_mask_min);
			for (int j = 0; j < pc; j++)
				destination[i++] = _min;
		}
	}
	if (flip) {
		memcpy(source, destination, sizeof(uint32_t) * n);
	}
	return true;
}

bool sieve_sort_core(uint32_t* a, size_t n, uint32_t* result, int depth);
bool sieve_sort_omp(uint32_t* a, size_t n, uint32_t* result, int depth) {
	size_t loops = 0, stride = 0, reminder = 0;
	__mmask16 mask = 0;
	bool flip = false;
	if (!get_config(n, loops, stride, reminder, mask, flip)) return false;

	if (depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < loops; i++) {
			sieve_sort_core(a + i * stride,
				(i == loops - 1 && reminder > 0) ? reminder : stride,
				result + i * stride,
				depth - 1);
		}
	}
	else {
		for (int i = 0; i < loops; i++) {
			sieve_sort_core(a + i * stride,
				(i == loops - 1 && reminder > 0) ? reminder : stride,
				result + i * stride,
				depth - 1);
		}
	}
	return sieve_collect(n, loops, stride, reminder, mask, flip, result, a);
}
bool sieve_sort_core(uint32_t* a, size_t n, uint32_t* result, int depth) {
	return (n <= _256)
		? sieve_sort_256(a, n, result)
		: sieve_sort_omp(a, n, result, depth)
		;
}

bool sieve_sort(uint32_t* a, size_t n, int depth = 32)
{
	bool done = false;
	if (a == nullptr)
		return false;
	else if (n <= 1)
		return true;
	else {
		uint32_t* result = new uint32_t[n];
		if (result != nullptr) {
			//memset(result, 0xff, n);
			done = sieve_sort_core(a, n, result, depth);
			delete[] result;
		}
	}
	return done;
}

__forceinline uint32_t generate_random_32(size_t range = 0ULL) {
	uint64_t v0 = 0ULL;
	int r = _rdrand64_step(&v0);
	if (range >= (1ULL << 32)) range = (~0U);
	uint32_t value = ((v0 >> 32) & (~0U)) ^ (v0 & (~0U));
	return range == 0 ? value : (uint32_t)((value / (double)(~0U)) * range);
}
__forceinline uint64_t generate_random_64(uint64_t range = 0ULL) {
	uint64_t v0 = 0ULL;
	int r = _rdrand64_step(&v0);
	return range == 0ULL ? v0 : (uint64_t)((v0 / (double)(~0ULL)) * range);
}

void tests() {
	const int max_retries = 100000;
	uint64_t result64x8[8] = { 0 };
	uint64_t compare64x8[8] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < 8; i++) {
			compare64x8[i] = result64x8[i] = generate_random_64();
		}
		sieve_sort8_64_direct(_mm512_loadu_epi64(result64x8), result64x8);
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
		__m512i t = sieve_sort16_32_direct(_mm512_loadu_epi32(result32x16));
		__m512i r = sieve_sort16_32_loop(_mm512_loadu_epi32(result32x16));
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
void do_test(const size_t count = 256, const int max_repeats = 1, const int use_omp = 0) {
	uint32_t** results_sieve = new uint32_t * [max_repeats];
	uint32_t** results_stdst = new uint32_t * [max_repeats];
#pragma omp parallel for
	for (int c = 0; c < max_repeats; c++) {
		results_sieve[c] = new uint32_t[count];
		results_stdst[c] = new uint32_t[count];
		//regard >256M as random memory already
		for (size_t i = 0; i < count; i++) {
			results_stdst[c][i] = results_sieve[c][i] = generate_random_32(count);
		}
	}

	//ok for 16x
	auto start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		sieve_sort(results_sieve[c], count, (use_omp ? 32 : -1));
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed1 = end - start;
	double d1 = ((double)max_repeats / elapsed1.count()) / 1000.0;

	start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		std::sort(results_stdst[c], results_stdst[c] + count);
	}

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed2 = end - start;
	double d2 = ((double)max_repeats / elapsed2.count() / (1000.0));
	std::cout << "==================================" << std::endl;
	//#pragma omp parallel for
	for (int c = 0; c < max_repeats; c++) {
		//uint32_t result32[count];
		for (int d = 0; d < count; d++) {
			if (results_sieve[c][d] != results_stdst[c][d]) {
				std::cout << "found bad value at repeat " << c << " index " << d << std::endl;
			}
		}
		delete[] results_sieve[c];
		delete[] results_stdst[c];
	}
	delete[] results_sieve;
	delete[] results_stdst;

	std::cout << "samples:" << count << std::endl;
	std::cout << "repeats:" << max_repeats << std::endl;
	std::cout << "omp: " << omp_get_max_threads() << " threads" << std::endl;
	std::cout << "sieve sort speed:" << d1 << "K/s" << std::endl;
	std::cout << "std sort speed:  " << d2 << "K/s" << std::endl;
	std::cout << "t1(seive):" << elapsed1.count() << " s" << std::endl;
	std::cout << "t2(std::):" << elapsed2.count() << " s" << std::endl;
	std::cout << "ratio:" << (d1 / d2 * 100.0) << "%" << std::endl;
}
int main(int argc, char* argv[])
{
#if 0
	tests();
#endif
#if 0
	size_t t = 1ULL << 32;
	do_test(t, 1, 1);
#endif
	for (int i = 0; i < 24; i += 4) {
		int t = 1 << i;
		std::cout << std::endl;
		std::cout << "i=" << i << ",t=" << t << std::endl;
		do_test(t - 1, 1, 1);
		do_test(t + 0, 1, 1);
		do_test(t + 1, 1, 1);
		if (t == 1) {
			for (int j = 2; j < 16; j++) {
				do_test(j, 1, 1);
			}
		}
	}

	return 0;
}

