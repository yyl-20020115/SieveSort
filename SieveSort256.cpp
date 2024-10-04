#include "SieveSort.h"

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

static const __m128i __zero = _mm_setzero_si128();
static const __m256i _zero = _mm256_setzero_si256();
static const __m256i ones = _mm256_set1_epi32(1);
static const __m256i mones = _mm256_set1_epi32(-1);
static const __m256i ones64 = _mm256_set1_epi64x(1);


static __forceinline short _mm256_cmpge_epu32_popcnt(__m256i a, __m256i b) {
	__m256i ah = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a, 1));
	__m256i al = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a, 0));

	__m256i bh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(b, 1));
	__m256i bl = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(b, 0));

	__m256i rh = _mm256_or_si256(_mm256_cmpgt_epi64(ah, bh), _mm256_cmpeq_epi64(ah,bh));
	__m256i rl = _mm256_or_si256(_mm256_cmpgt_epi64(al, bl), _mm256_cmpeq_epi64(al,bl));

	short c = (
		+ ((rh.m256i_u64[0]) != 0)
		+ ((rh.m256i_u64[1]) != 0)
		+ ((rh.m256i_u64[2]) != 0)
		+ ((rh.m256i_u64[3]) != 0)
		+ ((rl.m256i_u64[0]) != 0)
		+ ((rl.m256i_u64[1]) != 0)
		+ ((rl.m256i_u64[2]) != 0)
		+ ((rl.m256i_u64[3]) != 0)
		);
	return c;
}
static __forceinline short _mm256_cmple_epu32_popcnt(__m256i a, __m256i b) {
	__m256i ah = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a, 1));
	__m256i al = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a, 0));

	__m256i bh = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(b, 1));
	__m256i bl = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(b, 0));

	__m256i rh = _mm256_cmpgt_epi64(ah, bh);
	__m256i rl = _mm256_cmpgt_epi64(al, bl);

	short c = 8 - (
		+ ((rh.m256i_u64[0]) != 0)
		+ ((rh.m256i_u64[1]) != 0)
		+ ((rh.m256i_u64[2]) != 0)
		+ ((rh.m256i_u64[3]) != 0)
		+ ((rl.m256i_u64[0]) != 0)
		+ ((rl.m256i_u64[1]) != 0)
		+ ((rl.m256i_u64[2]) != 0)
		+ ((rl.m256i_u64[3]) != 0)
		);
	return c;
}
static __forceinline __mmask8 _mm256_mask_cmpeq_epi32_mask_(__mmask8 mask, __m256i a, __m256i b) {
	__mmask8 t = 0;
	__m256i r = _mm256_cmpeq_epi32(a, b);
	t |= ((r.m256i_u32[0] != 0) << 0);
	t |= ((r.m256i_u32[1] != 0) << 1);
	t |= ((r.m256i_u32[2] != 0) << 2);
	t |= ((r.m256i_u32[3] != 0) << 3);
	t |= ((r.m256i_u32[4] != 0) << 4);
	t |= ((r.m256i_u32[5] != 0) << 5);
	t |= ((r.m256i_u32[6] != 0) << 6);
	t |= ((r.m256i_u32[7] != 0) << 7);
	return t & mask;
}
static __forceinline bool sieve_get_min(__mmask8 mask, __m256i a, uint32_t& _min, __mmask8& _mask_min) {
	if (mask == 0) return false;
	__m128i counts = _mm_setr_epi16(
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[0]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[1]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[2]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[3]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[4]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[5]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[6]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[7]), a)
	);
	counts.m128i_u16[0] = ((mask & (1 << 0)) != 0) ? counts.m128i_u16[0] : 0xffff;
	counts.m128i_u16[1] = ((mask & (1 << 1)) != 0) ? counts.m128i_u16[1] : 0xffff;
	counts.m128i_u16[2] = ((mask & (1 << 2)) != 0) ? counts.m128i_u16[2] : 0xffff;
	counts.m128i_u16[3] = ((mask & (1 << 3)) != 0) ? counts.m128i_u16[3] : 0xffff;
	counts.m128i_u16[4] = ((mask & (1 << 4)) != 0) ? counts.m128i_u16[4] : 0xffff;
	counts.m128i_u16[5] = ((mask & (1 << 5)) != 0) ? counts.m128i_u16[5] : 0xffff;
	counts.m128i_u16[6] = ((mask & (1 << 6)) != 0) ? counts.m128i_u16[6] : 0xffff;
	counts.m128i_u16[7] = ((mask & (1 << 7)) != 0) ? counts.m128i_u16[7] : 0xffff;

	__m128i result = _mm_minpos_epu16(counts);
	unsigned int index = result.m128i_i16[1] & 0x7;
	_mask_min = _mm256_mask_cmpeq_epi32_mask_(mask, a, _mm256_set1_epi32(_min = a.m256i_u32[index]));
	return true;
}
static __forceinline bool sieve_get_min_max(__mmask8 mask, __m256i a, uint32_t& _min, uint32_t& _max, __mmask8& _mask_min, __mmask8& _mask_max) {
	if (mask == 0) return false;
	__m128i counts = _mm_setr_epi16(
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[0]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[1]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[2]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[3]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[4]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[5]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[6]), a),
		_mm256_cmpge_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[7]), a)
	);

	counts.m128i_u16[0] = (mask & (1 << 0)) ? counts.m128i_u16[0] : 0xffff;
	counts.m128i_u16[1] = (mask & (1 << 1)) ? counts.m128i_u16[1] : 0xffff;
	counts.m128i_u16[2] = (mask & (1 << 2)) ? counts.m128i_u16[2] : 0xffff;
	counts.m128i_u16[3] = (mask & (1 << 3)) ? counts.m128i_u16[3] : 0xffff;
	counts.m128i_u16[4] = (mask & (1 << 4)) ? counts.m128i_u16[4] : 0xffff;
	counts.m128i_u16[5] = (mask & (1 << 5)) ? counts.m128i_u16[5] : 0xffff;
	counts.m128i_u16[6] = (mask & (1 << 6)) ? counts.m128i_u16[6] : 0xffff;
	counts.m128i_u16[7] = (mask & (1 << 7)) ? counts.m128i_u16[7] : 0xffff;

	__m128i result = _mm_minpos_epu16(counts);
	unsigned short value = result.m128i_i16[0];
	unsigned short index = result.m128i_i16[1] & 0x7;
	_mask_min = _mm256_mask_cmpeq_epi32_mask_(mask, a, _mm256_set1_epi32(_min = a.m256i_u32[index]));

	mask &= ~(_mask_min);
	counts = _mm_setr_epi16(
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[0]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[1]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[2]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[3]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[4]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[5]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[6]), a),
		_mm256_cmple_epu32_popcnt(_mm256_set1_epi32(a.m256i_u32[7]), a)
	);

	counts.m128i_u16[0] = (mask & (1 << 0)) ? counts.m128i_u16[0] : 0xffff;
	counts.m128i_u16[1] = (mask & (1 << 1)) ? counts.m128i_u16[1] : 0xffff;
	counts.m128i_u16[2] = (mask & (1 << 2)) ? counts.m128i_u16[2] : 0xffff;
	counts.m128i_u16[3] = (mask & (1 << 3)) ? counts.m128i_u16[3] : 0xffff;
	counts.m128i_u16[4] = (mask & (1 << 4)) ? counts.m128i_u16[4] : 0xffff;
	counts.m128i_u16[5] = (mask & (1 << 5)) ? counts.m128i_u16[5] : 0xffff;
	counts.m128i_u16[6] = (mask & (1 << 6)) ? counts.m128i_u16[6] : 0xffff;
	counts.m128i_u16[7] = (mask & (1 << 7)) ? counts.m128i_u16[7] : 0xffff;

	result = _mm_minpos_epu16(counts);
	value = result.m128i_u16[0];
	index = result.m128i_u16[1] & 0x7;
	_mask_max = _mm256_mask_cmpeq_epi32_mask_(mask, a, _mm256_set1_epi32(_max = a.m256i_u32[index]));
	return true;
}
static __forceinline bool sieve_get_min(__mmask8 mask, uint32_t a[8], uint32_t& _min, __mmask8& _mask_min) {
	return sieve_get_min(mask, _mm256_loadu_epi32(a), _min, _mask_min);
}
static __forceinline int get_mask(__mmask8 _mask, int i) {
	return (_mask & (1 << i)) != 0 ? -1 : 0;
}
static __forceinline __m256i _mm256_mask_blend_epi32_(__mmask8 mask, __m256i a, __m256i b) {
	__m256i __mask = _mm256_setr_epi32(
		~get_mask(mask, 0),
		~get_mask(mask, 1),
		~get_mask(mask, 2),
		~get_mask(mask, 3),
		~get_mask(mask, 4),
		~get_mask(mask, 5),
		~get_mask(mask, 6),
		~get_mask(mask, 7)
	);
	return _mm256_or_si256(
		_mm256_and_si256(a, __mask),
		_mm256_andnot_si256(__mask, b));
}
__m256i sieve_sort8_32_loop(__m256i a, uint32_t* result /*[8]*/) {
	__m256i target = _zero;
	__mmask8 mask = 0xff;
	__mmask8 _min_mask = 0, _max_mask = 0;
	uint32_t i = 0, j = 8;
	uint32_t _min = 0, _max = 0;
	uint32_t c_min = 0, c_max = 0;
	while (sieve_get_min_max(mask, a, _min, _max, _min_mask, _max_mask)) {
		c_min = __popcnt16(_min_mask);
		c_max = __popcnt16(_max_mask);
		target = _mm256_mask_blend_epi32_((-(!!c_min)) & ((~((~0U) << c_min)) << i),
			target, _mm256_set1_epi32(_min));

		target = _mm256_mask_blend_epi32_((-(!!c_max)) & ((~((~0U) << c_max)) << (j - c_max)),
			target, _mm256_set1_epi32(_max));

		i += c_min;
		j -= c_max;

		mask &= ~(_min_mask | _max_mask);
	}
	_mm256_storeu_epi32(result, target);
	return target;
}
static __forceinline int _mm128_reduce_add_maskz_popcnt_epu16_low_half(__mmask8 mask, __m128i masks) {
	return
		+(int)((mask & (1 << 0)) ? __popcnt16(masks.m128i_u8[0]) : 0)
		+ (int)((mask & (1 << 1)) ? __popcnt16(masks.m128i_u8[1]) : 0)
		+ (int)((mask & (1 << 2)) ? __popcnt16(masks.m128i_u8[2]) : 0)
		+ (int)((mask & (1 << 3)) ? __popcnt16(masks.m128i_u8[3]) : 0)
		+ (int)((mask & (1 << 4)) ? __popcnt16(masks.m128i_u8[4]) : 0)
		+ (int)((mask & (1 << 5)) ? __popcnt16(masks.m128i_u8[5]) : 0)
		+ (int)((mask & (1 << 6)) ? __popcnt16(masks.m128i_u8[6]) : 0)
		+ (int)((mask & (1 << 7)) ? __popcnt16(masks.m128i_u8[7]) : 0)
		;
}
static __forceinline __mmask8 _mm_cmpeq_epi16_mask_half(__m128i a, __m128i b)
{
	__mmask8 mask = 0;
	__m128i r = _mm_cmpeq_epi8(a, b);
	mask |= (r.m128i_u8[0] != 0 ? (1 << 0) : 0);
	mask |= (r.m128i_u8[1] != 0 ? (1 << 1) : 0);
	mask |= (r.m128i_u8[2] != 0 ? (1 << 2) : 0);
	mask |= (r.m128i_u8[3] != 0 ? (1 << 3) : 0);
	mask |= (r.m128i_u8[4] != 0 ? (1 << 4) : 0);
	mask |= (r.m128i_u8[5] != 0 ? (1 << 5) : 0);
	mask |= (r.m128i_u8[6] != 0 ? (1 << 6) : 0);
	mask |= (r.m128i_u8[7] != 0 ? (1 << 7) : 0);
	return mask;
}
static __forceinline __m128i _mm_mask_andnot_si128_half(__m128i src, __mmask8 mask, __m128i a, __m128i b)
{
	__m128i r = _mm_andnot_si128(a, b);
	r.m128i_u8[0] = (mask & (1 << 0)) ? r.m128i_u8[0] : src.m128i_u8[0];
	r.m128i_u8[1] = (mask & (1 << 1)) ? r.m128i_u8[1] : src.m128i_u8[1];
	r.m128i_u8[2] = (mask & (1 << 2)) ? r.m128i_u8[2] : src.m128i_u8[2];
	r.m128i_u8[3] = (mask & (1 << 3)) ? r.m128i_u8[3] : src.m128i_u8[3];
	r.m128i_u8[4] = (mask & (1 << 4)) ? r.m128i_u8[4] : src.m128i_u8[4];
	r.m128i_u8[5] = (mask & (1 << 5)) ? r.m128i_u8[5] : src.m128i_u8[5];
	r.m128i_u8[6] = (mask & (1 << 6)) ? r.m128i_u8[6] : src.m128i_u8[6];
	r.m128i_u8[7] = (mask & (1 << 7)) ? r.m128i_u8[7] : src.m128i_u8[7];
	return _mm_insert_epi64(r, 0, 1);
}
static __forceinline int seive_get_min(uint32_t& p_min, __mmask8& _all_masks, __mmask8 masks[8], __m256i values[8]) {
	if (_all_masks == 0) return 0;
	int count = 0;
	uint32_t  _mines[8];
	__mmask8 mask_mines[8];
	for (size_t i = 0; i < 8; i++) {
		if (masks[i] == 0) {
			_mines[i] = 0xffff;
			mask_mines[i] = 0;
		}
		else if (sieve_get_min(
			masks[i],
			values[i],
			_mines[i],
			mask_mines[i]))
		{
			//OK
		}
	}
	__mmask8 found_mask = 0;
	if (sieve_get_min(_all_masks, _mm256_loadu_epi32(_mines), p_min, found_mask)) {
		__m128i _mask_mines_m128 = _mm_loadu_si64(mask_mines);
		__m128i _masks_m128 = _mm_loadu_si64(masks);
		count = _mm128_reduce_add_maskz_popcnt_epu16_low_half(found_mask, _mask_mines_m128);
		_masks_m128 = _mm_mask_andnot_si128_half(_masks_m128, found_mask, _mask_mines_m128, _masks_m128);
		_all_masks &= ~(_mm_cmpeq_epi16_mask_half(_mask_mines_m128, __zero));
		_mm_storeu_si64(masks, _masks_m128);
	}
	return count;
}


bool sieve_sort_8(uint32_t* a /*[8]*/, size_t n, uint32_t* result/*[8]*/) {
#ifdef USE_STD_SORT
	std::sort(a, a + n);
	memcpy(result, a, n * sizeof(uint32_t));
	return true;
#else
	if (n > 8)
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
		uint32_t b[8];
		memcpy(b, a, sizeof(uint32_t) * n);
		memset(b + n, 0xff, sizeof(uint32_t) * (8 - n));
		sieve_sort8_32_loop(_mm256_loadu_epi32(b), b);
		memcpy(a, b, sizeof(uint32_t) * n);
		memcpy(result, b, sizeof(uint32_t) * n);
		return true;
	}
#endif
}
bool sieve_sort_64(uint32_t* a/*[64]*/, size_t n, uint32_t* result) {
#ifdef USE_STD_SORT
	std::sort(a, a + n);
	memcpy(result, a, n * sizeof(uint32_t));
	return true;
#else
	if (n <= 8)
		return sieve_sort_8(a, n, result);
	else if (n > 64)
		return false;
	else { //8<n<=64
		uint32_t b[64];
		memcpy(b, a, sizeof(uint32_t) * n);
		memset(b + n, 0xff, sizeof(uint32_t) * (64 - n));
		__m256i values[8] = {
			_mm256_loadu_epi32(b + (0 << 3)),
			_mm256_loadu_epi32(b + (1 << 3)),
			_mm256_loadu_epi32(b + (2 << 3)),
			_mm256_loadu_epi32(b + (3 << 3)),
			_mm256_loadu_epi32(b + (4 << 3)),
			_mm256_loadu_epi32(b + (5 << 3)),
			_mm256_loadu_epi32(b + (6 << 3)),
			_mm256_loadu_epi32(b + (7 << 3)),
		};
		__mmask8 masks[8];
		memset(masks, 0xff, sizeof(masks));
		__mmask8 all_masks = 0xff;

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
// xxx/xxx/xxx 
static __forceinline int get_depth(size_t n) {
	int top_bits = get_top_bit_index(n);
	int all_bits = __popcnt64(n) == 1
		? (top_bits - 1)
		: top_bits
		;
	return ((all_bits / 3) + ((all_bits % 3) != 0));
}
static __forceinline bool get_config(size_t n, size_t& loops, size_t& stride, size_t& reminder, __mmask8& mask, int min_bits = 6) {
	if (n < ((1ULL) << min_bits)) return false;
	int depths = get_depth(n);
	int max_bits = depths * 3;
	stride = (1ULL) << (max_bits - 3);
	reminder = n & (~((~0ULL) << (max_bits - 3)));
	loops = (n - reminder) / stride + (reminder > 0);
	mask = ~((~0U) << (loops));
	return true;
}
static __forceinline __m256i _mm256_mask_inc(__mmask8 k, __m256i a) {
	return _mm256_add_epi32(a, _mm256_set_epi32(
		(k & (1 << 7)) ? 1 : 0,
		(k & (1 << 6)) ? 1 : 0,
		(k & (1 << 5)) ? 1 : 0,
		(k & (1 << 4)) ? 1 : 0,
		(k & (1 << 3)) ? 1 : 0,
		(k & (1 << 2)) ? 1 : 0,
		(k & (1 << 1)) ? 1 : 0,
		(k & (1 << 0)) ? 1 : 0
	));
}
static bool sieve_collect(size_t n, size_t loops, size_t stride, size_t reminder, __mmask8 mask,
	uint32_t* source, uint32_t* destination) {
	if (n == 0 || loops == 0 || loops > 8 || mask == 0 || source == nullptr || destination == nullptr)
		return false;
	const size_t large_stride_threshold = _16M; //(1ULL << 24); //(1ULL << 12))
	const size_t extreme_large_stride_threshold = _16P; //(1ULL << 56);
	if (stride <= large_stride_threshold) {
		__m256i idx = _zero;
		__m256i top = _zero;
		uint32_t p = 0;
		for (int i = 0; i < loops; i++) {
			idx.m256i_u32[i] = p;
			top.m256i_u32[i] = p + (uint32_t)((i == loops - 1 && reminder > 0) ? reminder : stride);
			p += (uint32_t)stride;
		}
		int pc = 0, i = 0;
		uint32_t _min = 0;
		__mmask8 _mask_min = 0;
		while (mask != 0 && i < n) {
			__m256i vmask = _mm256_setr_epi32(
				(((mask & (1 << 0)) != 0) ? -1 : 0),
				(((mask & (1 << 1)) != 0) ? -1 : 0),
				(((mask & (1 << 2)) != 0) ? -1 : 0),
				(((mask & (1 << 3)) != 0) ? -1 : 0),
				(((mask & (1 << 4)) != 0) ? -1 : 0),
				(((mask & (1 << 5)) != 0) ? -1 : 0),
				(((mask & (1 << 6)) != 0) ? -1 : 0),
				(((mask & (1 << 7)) != 0) ? -1 : 0)
			);
			__m256i values = _mm256_mask_i32gather_epi32(
				_zero, (int*)source, idx, vmask, sizeof(uint32_t));
			if (!sieve_get_min(mask, values, _min, _mask_min)) break;
			idx = _mm256_mask_inc(_mask_min, idx);
			mask &= (~_mm256_mask_cmpeq_epi32_mask_(_mask_min, idx, top));
			pc = __popcnt16(_mask_min);
			for (int j = 0; j < pc; j++)
				destination[i++] = _min;
		}
		return true;
	}
	else if (stride <= extreme_large_stride_threshold) {
		__m256i _idx_low_ = _zero;
		__m256i top_low_ = _zero;
		size_t loops_low_ = loops >= 8 ? 8 : loops;
		size_t p = 0;
		for (int i = 0; i < loops_low_; i++) {
			_idx_low_.m256i_u64[i] = p;
			top_low_.m256i_u64[i] = p + ((i == loops - 1 && reminder > 0) ? reminder : stride);
			p += stride;
		}

		int pc = 0, i = 0;
		uint32_t _min = 0;
		__mmask8 _mask_low_ = 0;
		__mmask8 _mask_min_low_ = 0;
		__mmask8 _mask_min = 0;

		while (mask != 0 && i < n) {
			//__m128i vmask = __zero;//_mask_low_
			__m128i vmask = _mm_setr_epi32(
				(((mask & (1 << 0)) != 0) ? -1 : 0),
				(((mask & (1 << 1)) != 0) ? -1 : 0),
				(((mask & (1 << 2)) != 0) ? -1 : 0),
				(((mask & (1 << 3)) != 0) ? -1 : 0)
				//,
				//(((mask & (1 << 4)) != 0) ? -1 : 0),
				//(((mask & (1 << 5)) != 0) ? -1 : 0),
				//(((mask & (1 << 6)) != 0) ? -1 : 0),
				//(((mask & (1 << 7)) != 0) ? -1 : 0)
			);

			_mask_low_ = (__mmask8)(mask & 0xff);
			__m128i values_low_ = _mm256_mask_i64gather_epi32(
				__zero, (int*)source, _idx_low_, vmask, sizeof(uint32_t));

			if (!sieve_get_min(mask,
				_mm256_castsi128_si256(values_low_), _min, _mask_min)) break;

			_mask_min_low_ = _mask_min;

			_idx_low_ = _mm256_mask_add_epi64(_idx_low_, _mask_min_low_, _idx_low_, ones64);
			_mask_low_ &= _mm256_mask_cmpeq_epu64_mask(_mask_min_low_, _idx_low_, top_low_);

			mask &= ~((__mmask16)_mask_low_);
			pc = __popcnt16(_mask_min);
			for (int j = 0; j < pc; j++)
				destination[i++] = _min;
		}
		return true;
	}
	return false;
}

static bool sieve_sort_core(uint32_t* a, size_t n, uint32_t* result, int depth, int omp_depth);
static bool sieve_sort_omp(uint32_t* a, size_t n, uint32_t* result, int depth, int omp_depth) {
	size_t loops = 0, stride = 0, reminder = 0;
	__mmask8 mask = 0;
	if (!get_config(n, loops, stride, reminder, mask)) return false;
	if (omp_depth > 0 && depth >= 2) {
#pragma omp parallel for
		for (int i = 0; i < loops; i++) {
			sieve_sort_core(a + i * stride,
				(i == loops - 1 && reminder > 0) ? reminder : stride,
				result + i * stride,
				depth - 1, omp_depth - 1);
		}
	}
	else {
		for (int i = 0; i < loops; i++) {
			sieve_sort_core(a + i * stride,
				(i == loops - 1 && reminder > 0) ? reminder : stride,
				result + i * stride,
				depth - 1, omp_depth - 1);
		}
	}
	if (depth >= 4 && ((depth - 3) & 1) == 1) {
		std::swap(result, a);
	}
	return sieve_collect(n, loops, stride, reminder, mask, result, a);
}
static bool sieve_sort_core(uint32_t* a, size_t n, uint32_t* result, int depth, int omp_depth) {
	return (n <= 64)
		? sieve_sort_64(a, n, result)
		: sieve_sort_omp(a, n, result, depth, omp_depth)
		;
}

bool sieve_sort_avx2(uint32_t** pa, size_t n, int omp_depth)
{
	bool done = false;
	//max(n)==256P (2^60)
	if (pa == nullptr || *pa == nullptr)
		return false;
	else if (n <= 1)
		return true;
	else if (n == 2) {
		uint32_t a0 = *(*pa + 0);
		uint32_t a1 = *(*pa + 1);
		*(*pa + 0) = std::min(a0, a1);
		*(*pa + 1) = std::max(a0, a1);
		return true;
	}
	else {
		uint32_t* result = new uint32_t[n];
		if (result != nullptr) {
			int max_depth = get_depth(n);
			done = sieve_sort_core(*pa, n, result, max_depth, omp_depth);
			if (max_depth >= 4 && ((max_depth & 1) == 0)) {
				std::swap(*pa, result);
			}
			delete[] result;
		}
	}
	return done;
}
