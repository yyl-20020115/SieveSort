#pragma once
#include <iostream>
#include <immintrin.h>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <memory.h>
#include <omp.h>

#ifdef __GNUC__
#define __forceinline 
#endif

uint32_t generate_random_32(size_t range = 0ULL);
uint64_t generate_random_64(uint64_t range = 0ULL);

int get_top_bit_index(size_t n);

__m256i sieve_sort8_32_loop(__m256i a, uint32_t* result);
bool sieve_sort_8(uint32_t* a, size_t n, uint32_t* result);
bool sieve_sort_64(uint32_t* a, size_t n, uint32_t* result);
bool sieve_sort_avx2(uint32_t** pa, size_t n, int omp_depth = 64);

#ifdef ENABLE_AVX_512
bool sieve_sort_16(uint32_t* a, size_t n, uint32_t* result);
bool sieve_sort_256(uint32_t* a, size_t n, uint32_t* result);
__m512i sieve_sort16_32_loop(__m512i a, uint32_t* result);
__m512i sieve_sort8_64_loop(__m512i a, uint64_t* result);
__m512i sieve_sort16_32_direct(__m512i a, uint32_t* result);
__m512i sieve_sort8_64_direct(__m512i a, uint64_t* result);
bool sieve_sort_avx512(uint32_t** pa, size_t n, int omp_depth = 32);
#endif