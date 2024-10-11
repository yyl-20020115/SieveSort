#include <iostream>
#include <chrono>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SieveSortCUDA.cuh"
#include <omp.h>

static void long_test_cuda(const size_t count = 256, const int max_repeats = 1) {
	uint32_t** results_sieve = new uint32_t * [max_repeats];
	uint32_t** results_stdst = new uint32_t * [max_repeats];
//#pragma omp parallel for
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
		sieve_sort_cuda(results_sieve[c], count);
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
					std::cout << "found bad value at repeat " << c << " index " << d 
						<<":"<<std::hex<< results_sieve[c][d]<<", "<< results_stdst[c][d]<<std::dec<< std::endl;
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
static void long_tests_cuda(size_t start = 12, size_t end = 16) {
	for (size_t i = start; i <= end; i++) {
		std::cout << std::endl;
		std::cout << "i=" << i << std::endl;
		long_test_cuda((1ULL << i), 1);
	}
}

int main()
{
	long_tests_cuda();
	return 0;
}