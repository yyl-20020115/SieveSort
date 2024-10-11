#include "SieveSort.h"

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
const size_t _4T = _256G << 4;  //44
const size_t _64T = _4T << 4;   //48
const size_t _1P = _64T << 4;   //52
const size_t _16P = _1P << 4;   //56
const size_t _256P = _16P << 4; //60

static const __m256i _zero = _mm256_setzero_si256();
static const __m512i zero = _mm512_setzero_si512();
static const __m512i ones = _mm512_set1_epi32(1);
static const __m512i ones64 = _mm512_set1_epi64(1);


static __forceinline bool sieve_get_min(__mmask16 mask, __m512i a, uint32_t& _min, __mmask16& _mask_min) {
	if (mask != 0) {
		_mask_min = _mm512_mask_cmpeq_epu32_mask(mask, a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask, a)));
		return true;
	}
	return false;
}

static __forceinline bool sieve_get_min_max(__mmask16 mask, __m512i a, uint32_t& _min, uint32_t& _max, __mmask16& _mask_min, __mmask16& _mask_max) {
	if (mask != 0) {
		_mask_max = _mm512_mask_cmpeq_epu32_mask(mask, a, _mm512_set1_epi32(
			_max = _mm512_mask_reduce_max_epu32(mask, a)));
		_mask_min = _mm512_mask_cmpeq_epu32_mask(mask & (~_mask_max), a, _mm512_set1_epi32(
			_min = _mm512_mask_reduce_min_epu32(mask & (~_mask_max), a)));
		return true;
	}
	return false;
}
static __forceinline bool sieve_get_min_max(__mmask8 mask, __m512i a, uint64_t& _min, uint64_t& _max, __mmask8& _mask_min, __mmask8& _mask_max) {
	if (mask != 0) {
		_mask_max = _mm512_mask_cmpeq_epu64_mask(mask, a, _mm512_set1_epi64(
			_max = _mm512_mask_reduce_max_epu64(mask, a)));
		_mask_min = _mm512_mask_cmpeq_epu64_mask(mask & (~_mask_max), a, _mm512_set1_epi64(
			_min = _mm512_mask_reduce_min_epu64(mask & (~_mask_max), a)));
		return true;
	}
	return false;
}
__m512i sieve_sort16_32_loop(__m512i a, uint32_t* result) {
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
	_mm512_storeu_epi32(result, target);
	return target;
}
__m512i sieve_sort8_64_loop(__m512i a, uint64_t* result) {
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
	_mm512_storeu_epi64(result, target);
	return target;
}
__m512i sieve_sort16_32_direct(__m512i a, uint32_t* result) {
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
	_mm512_storeu_epi32(result, target);
	return target;
}
__m512i sieve_sort8_64_direct(__m512i a, uint64_t* result) {
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
	_mm512_storeu_epi64(result, target);
	return target;
}
static __forceinline int seive_get_min(uint32_t& p_min, __mmask16& _all_masks, __mmask16 masks[16], __m512i values[16]) {
	if (_all_masks == 0) return 0;
	int count = 0;
	uint32_t  _mines[16];
	__mmask16 mask_mines[16];
	for (size_t i = 0; i < 16; i++) {
		if ((_all_masks & (1 << i)) != 0) {
			if (sieve_get_min(masks[i],
				values[i],
				_mines[i],
				mask_mines[i]))
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
		_all_masks &= ~_mm512_cmpeq_epu32_mask(_masks, zero);
		_mm256_storeu_epi16(masks, _mm512_cvtepi32_epi16(_masks));
	}
	return count;
}


bool sieve_sort_16(uint32_t* a, size_t n, uint32_t* result) {
#ifdef USE_STD_SORT
	std::sort(a, a + n);
	memcpy(result, a, n * sizeof(uint32_t));
	return true;
#else
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
		result[0] = a[0] = std::min(a0, a1);
		result[1] = a[1] = std::max(a0, a1);
		return true;
	}
	else { //2=<n<=16
		uint32_t b[_16];
		memcpy(b, a, sizeof(uint32_t) * n);
		memset(b + n, 0xff, sizeof(uint32_t) * (_16 - n));
		sieve_sort16_32_loop(_mm512_loadu_epi32(b), b);
		memcpy(a, b, sizeof(uint32_t) * n);
		memcpy(result, b, sizeof(uint32_t) * n);
		return true;
	}
#endif
}
bool sieve_sort_256(uint32_t* a, size_t n, uint32_t* result) {
#ifdef USE_STD_SORT
	std::sort(a, a + n);
	memcpy(result, a, n * sizeof(uint32_t));
	return true;
#else
	if (n <= _16)
		return sieve_sort_16(a, n, result);
	else if (n > _256)
		return false;
	else { //16<n<=256
		uint32_t b[_256];
		memcpy(b, a, sizeof(uint32_t) * n);
		memset(b + n, 0xff, sizeof(uint32_t) * (_256 - n));
		__m512i values[16] = {
			_mm512_loadu_epi32(b + (0 << 4)),
			_mm512_loadu_epi32(b + (1 << 4)),
			_mm512_loadu_epi32(b + (2 << 4)),
			_mm512_loadu_epi32(b + (3 << 4)),
			_mm512_loadu_epi32(b + (4 << 4)),
			_mm512_loadu_epi32(b + (5 << 4)),
			_mm512_loadu_epi32(b + (6 << 4)),
			_mm512_loadu_epi32(b + (7 << 4)),
			_mm512_loadu_epi32(b + (8 << 4)),
			_mm512_loadu_epi32(b + (9 << 4)),
			_mm512_loadu_epi32(b + (10 << 4)),
			_mm512_loadu_epi32(b + (11 << 4)),
			_mm512_loadu_epi32(b + (12 << 4)),
			_mm512_loadu_epi32(b + (13 << 4)),
			_mm512_loadu_epi32(b + (14 << 4)),
			_mm512_loadu_epi32(b + (15 << 4)),
		};
		__mmask16 masks[16];
		memset(masks, 0xff, sizeof(masks));
		__mmask16 all_masks = 0xffff;

		int p = 0, count = 0;
		uint32_t _min = 0;
		while (all_masks != 0 && p < n) {
			if (0 == (count = seive_get_min(_min, all_masks, masks, values)))
				break;
			for (int i = 0; i < count; i++)
				result[p + i] = a[p + i] = _min;
			p += count;
		}
		return true;
	}
#endif
}
static bool sieve_collect(size_t n, size_t loops, size_t stride, size_t reminder, __mmask16 mask,
	uint32_t* source, uint32_t* destination) {
	if (n == 0 || loops == 0 || loops > 16 || mask == 0 || source == nullptr || destination == nullptr)
		return false;
	const size_t large_stride_threshold = _16M; //(1ULL << 24); //(1ULL << 12))
	const size_t extreme_large_stride_threshold = _16P; //(1ULL << 56);
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
		return true;
	}
	else if (stride <= extreme_large_stride_threshold) {
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
		return true;
	}
	return false;
}

static bool sieve_sort_core(uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth, int omp_depth);
static bool sieve_sort_omp(uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth, int omp_depth) {
	size_t loops = 0, stride = 0, reminder = 0;
	__mmask16 mask = 0;
	if (!get_config(n, loops, stride, reminder, mask, 8, 4)) return false;
	if (omp_depth > 0) {
#pragma omp parallel for
		for (int i = 0; i < loops; i++) {
			sieve_sort_core(a + i * stride,
				(i == loops - 1 && reminder > 0) ? reminder : stride,
				result + i * stride,
				max_depth, depth - 1, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < loops; i++) {
			sieve_sort_core(a + i * stride,
				(i == loops - 1 && reminder > 0) ? reminder : stride,
				result + i * stride,
				max_depth, depth - 1, omp_depth - 1);
		}
	}
	int delta_depth = max_depth - depth;

	if ((delta_depth & 1) == 1) {
		std::swap(result, a);
	}

	return sieve_collect(n, loops, stride, reminder, mask, result, a);
}
static bool sieve_sort_core(uint32_t* a, size_t n, uint32_t* result, int max_depth, int depth, int omp_depth) {
	return (n <= _256)
		? sieve_sort_256(a, n, result)
		: sieve_sort_omp(a, n, result, max_depth, depth, omp_depth)
		;
}


bool sieve_sort_avx512(uint32_t* a, size_t n, int omp_depth)
{
	bool done = false;
	//max(n)==256P (2^60)
	if (a == nullptr || n > _256P)
		return false;
	else if (n == 0)
		return true;
	else if (n == 1) {
		return true;
	}
	else if (n == 2) {
		uint32_t a0 = *(a + 0);
		uint32_t a1 = *(a + 1);
		*(a + 0) = std::min(a0, a1);
		*(a + 1) = std::max(a0, a1);
		return true;
	}
	else {
		uint32_t* result = new uint32_t[n];
		if (result != nullptr) {
			int max_depth = get_depth(n, 4);
			done = sieve_sort_core(a, n, result, max_depth, max_depth, omp_depth);
			delete[] result;
		}
	}
	return done;
}

