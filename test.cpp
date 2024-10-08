#include "SieveSort.h"
#include <intrin.h>

static void short_tests_avx512() {
	const int max_retries = 100;
	const int count = 8;
	uint64_t result64x8[count] = { 0 };
	uint64_t compare64x8[count] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < count; i++) {
			compare64x8[i] = result64x8[i] = generate_random_64();
		}

		sieve_sort8_64_direct(_mm512_loadu_epi64(result64x8), result64x8);
		//sieve_sort64x8_loop(_mm512_loadu_epi64(result64x8), result64x8);
		std::sort(compare64x8, compare64x8 + count);
		bool ex = std::equal(compare64x8, compare64x8 + count, result64x8);
		if (!ex)
		{
			std::cout << "failed" << std::endl;
		}
	}
	std::cout << "64 pass" << std::endl;

	const int count2 = 16;
	uint32_t original32x16[count2] = { 0 };
	uint32_t result32x16[count2] = { 0 };
	uint32_t compare32x16[count2] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < count2; i++) {
			original32x16[i] = compare32x16[i] = result32x16[i] = generate_random_32();
		}
		__m512i t = { 0 };
		t = sieve_sort16_32_direct(_mm512_loadu_epi32(result32x16), result32x16);
		__m512i r = { 0 };
		r = sieve_sort16_32_loop(_mm512_loadu_epi32(result32x16), result32x16);
		//_mm512_storeu_epi32(result32x16, r);
		std::sort(compare32x16, compare32x16 + count2);
		bool ex = std::equal(result32x16, result32x16 + count2, compare32x16);
		if (!ex)
		{
			std::cout << "failed" << std::endl;
		}
	}
	std::cout << "32 pass" << std::endl;
	std::cout << "all pass" << std::endl;
}
static void long_test_avx512(const size_t count = 256, const int max_repeats = 1, const int use_omp = 0) {
	uint32_t** results_sieve = new uint32_t * [max_repeats];
	uint32_t** results_stdst = new uint32_t * [max_repeats];
#pragma omp parallel for
	for (int c = 0; c < max_repeats; c++) {
		results_sieve[c] = new uint32_t[count];
		results_stdst[c] = new uint32_t[count];
		for (size_t i = 0; i < count; i++) {
			results_stdst[c][i]
				= results_sieve[c][i]
				= generate_random_32();
		}
	}

	//ok for 16x
	auto start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		sieve_sort_avx512(results_sieve[c], count, (use_omp ? 32 : -1));
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
static void long_tests_avx512(size_t start = 16, size_t end = 20) {
	for (size_t i = start; i <= end; i++) {
		std::cout << std::endl;
		std::cout << "i=" << i << std::endl;
		long_test_avx512((1ULL<<i), 1, 1);
	}
}
static void short_tests_avx2() {
	const int count = 64;
	const int max_retries = 1000;
	uint32_t original[count] = { 0 };
	uint32_t result[count] = { 0 };
	uint32_t compare[count] = { 0 };
	for (int c = 0; c < max_retries; c++) {
		for (int i = 0; i < count; i++) {
			original[i] = compare[i] = result[i] = generate_random_32();
		}
		sieve_sort_64(original, count, result);
		//__m256i r = sieve_sort8_32_loop(_mm256_loadu_epi32(result), result);
		std::sort(compare, compare + count);
		bool ex = std::equal(result, result + count, compare);
		if (!ex) {
			std::cout << "failed" << std::endl;
		}
	}
	std::cout << "32 pass" << std::endl;
}
static void long_test_avx2(size_t count = 64, int max_repeats = 1, int use_omp = 1) {
	uint32_t** results_sieve = new uint32_t * [max_repeats];
	uint32_t** results_stdst = new uint32_t * [max_repeats];
#pragma omp parallel for
	for (int c = 0; c < max_repeats; c++) {
		results_sieve[c] = new uint32_t[count];
		results_stdst[c] = new uint32_t[count];
		for (size_t i = 0; i < count; i++) {
			results_stdst[c][i]
				= results_sieve[c][i]
				= generate_random_32();
		}
	}
	//ok for 16x
	auto start = std::chrono::high_resolution_clock::now();
	for (int c = 0; c < max_repeats; c++) {
		sieve_sort_avx2(results_sieve[c], count, (use_omp ? 32 : -1));
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
		bool beq = std::equal(results_sieve[c], results_sieve[c] + count, results_stdst[c]);
		if (!beq) {
			for (int d = 0; d < count; d++) {
				if (results_sieve[c][d] != results_stdst[c][d]) {
					std::cout << "found bad value at repeat " << c << " index " << d << std::endl;
				}
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
static void long_tests_avx2(size_t start = 12, size_t end = 28) {
	for (size_t i = start; i <= end; i++) {
		std::cout << std::endl;
		std::cout << "i=" << i << std::endl;
		long_test_avx2((1ULL << i), 1, 1);
	}
}

static void test256() {
	short_tests_avx2();
	long_tests_avx2();
}

static void test512()
{
	short_tests_avx512();
	long_tests_avx512();
}

static bool has_avx512() {

	int regs[4] = { 16,0,0,0 };

	__cpuid(regs, 0x7); // Call CPUID with feature identifier

	return (regs[1] & (1 << 16)) != 0;

}

int main(int argc, char* argv[])
{
	if (has_avx512()) {
		test512();
	}
	test256();
	return 0;
}

