#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <iomanip>
#include "Benchmark.hpp"
#include "sum.hpp"
#include <random>

void print_m128(__m128 &a) {
    std::cout << std::setw(9) << a[0] << " ";
    std::cout << std::setw(9) << a[1] << " ";
    std::cout << std::setw(9) << a[2] << " ";
    std::cout << std::setw(9) << a[3] << std::endl;
}

void print_m512d(__m512d &a) {
    for (int i = 0; i < 7; ++i) {
        std::cout << std::setw(9) << a[i] << " ";
    }
    std::cout << std::setw(9) << a[7] << std::endl;
}

void print_m512i_8x64(__m512i &a) {
    uint64_t bytes[8];
    _mm512_storeu_epi64(bytes, a);
    for (int i = 0; i < 7; ++i) {
        std::cout << std::setw(9) << bytes[i] << " ";
    }
    std::cout << std::setw(9) << bytes[7] << std::endl;
}

void instrinsics_examples() {
    // SIMD = single instruction multiple data
    // MMX = multimedia extension
    // SSE = streaming simd extension
    // SSSE = supplemental streaming simd extension
    // AVX = advanced vector extension

    // extensions in total:
    // MMX, SSE, SSE2, SSE3, SSSE3, SSE4, AVX, AVX2, AVX512
    // Question: what is usually supported by a cpu from 2021 and after ??

    // data must be 16 byte aligned or else segfault
    float __attribute__((aligned(16))) raw_a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float __attribute__((aligned(16))) raw_b[] = {17.0f, 16.0f, 15.0f, 14.0f};

    // documentation for intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ps
    // this is a very helpfull website

    // how does the intrinsics naming scheme work ??

    // sse instructions
    // load packed single precision
    __m128 a = _mm_load_ps(raw_a);
    __m128 b = _mm_load_ps(raw_b);
    __m128 c = _mm_add_ps(a, b);
    __m128 d = _mm_mul_ps(c, c);
    __m128 e = _mm_hadd_ps(a, b); // horizontal add

    print_m128(a);
    print_m128(b);
    print_m128(c);
    print_m128(d);
    print_m128(e);
    std::cout << std::endl;

    double __attribute__((aligned(64))) raw_da[8] = {3.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 14.0};
    double __attribute__((aligned(64))) raw_db[8] = {63.0, 75.0, 87.0, 98.0, 129.0, 1311.0, 1413.0, 1514.0};

    // avx512f instructions
    __m512d da = _mm512_load_pd(raw_da);
    __m512d db = _mm512_load_pd(raw_db);
    __m512d dc = _mm512_div_pd(db, da);
    __m512d dd = _mm512_sqrt_pd(dc);
    __m512d de = _mm512_rsqrt14_pd(db); // 1 / sqrt(..)
    __m512d dz = _mm512_set1_pd(0); // set all values to 0

    print_m512d(da);
    print_m512d(db);
    print_m512d(dc);
    print_m512d(dd);
    print_m512d(de);
    print_m512d(dz);
    std::cout << std::endl;

    // highly specialized instruction for example
    // __m512i _mm512_maskz_gf2p8affineinv_epi64_epi8(__mmask64 k, __m512i x, __m512i A, int b)

    // Compute an inverse affine transformation in the Galois Field 2^8.
    // An affine transformation is defined by A * x + b, where A represents an 8 by 8 bit matrix, x represents an 8-bit vector,
    // and b is a constant immediate byte. The inverse of the 8-bit values in x is defined with respect
    // to the reduction polynomial x^8 + x^4 + x^3 + x + 1. Store the packed 8-bit results in dst using
    // zeromask k (elements are zeroed out when the corresponding mask bit is not set).

    // permutate with index
    uint64_t idx_array[] = {0, 1, 2, 3, 8, 9, 10, 11};
    __m512i idx = _mm512_loadu_epi64(idx_array); // load unaligned
    __m512d pa = _mm512_permutex2var_pd(da, idx, db);

    print_m512i_8x64(idx);
    print_m512d(da);
    print_m512d(db);
    print_m512d(pa);
    std::cout << std::endl;
}

void generate_random_data(int wanted_len, uint32_t **dest, int *len) {
    *len = (wanted_len / 64) * 64;
    *dest = static_cast<uint32_t *>(_mm_malloc(*len * sizeof(uint32_t), 64));

    //std::random_device dev;
    std::mt19937 rng(0x44);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 2);

    std::cout << "Generating " << *len << " random numbers..." << std::endl;
    for (int i = 0; i < *len; ++i) {
        (*dest)[i] = dist(rng);
    }

}

void free_random_data(uint32_t **ptr) {
    _mm_free(*ptr);
    *ptr = nullptr;
}

void simple_sum_benchmark() {
    // summing up data benchmarks
    uint32_t *data = nullptr;
    int data_len = 0;
    generate_random_data(130'000'000, &data, &data_len);
    const int ITERATIONS = 300;

    uint32_t ret = 0;
    auto naive = Benchmark("naive-sum", ITERATIONS, [&]() {
        ret = naive_sum(data, data_len);
    });
    naive.run();
    std::cout << "naive: " << ret << "\n";

    auto simd = Benchmark("simd-sum", ITERATIONS, [&]() {
        ret = simd_sum(data, data_len);
    });
    simd.run();
    std::cout << "simd: " << ret << "\n";

    auto simd_256 = Benchmark("simd-256-sum", ITERATIONS, [&]() {
        ret = simd_256_sum(data, data_len);
    });
    simd_256.run();
    std::cout << "simd-256: " << ret << "\n";

    auto naive_omp = Benchmark("naive+omp-sum", ITERATIONS, [&]() {
        ret = naive_omp_sum(data, data_len);
    });
    naive_omp.run();
    std::cout << "naive+omp: " << ret << "\n";

    auto simd_omp = Benchmark("simd+omp-sum", ITERATIONS, [&]() {
        ret = simd_omp_sum(data, data_len);
    });
    simd_omp.run();
    std::cout << "simd+omp: " << ret << "\n";

    naive.print_stats();
    simd.print_stats();
    simd_256.print_stats();
    naive_omp.print_stats();
    simd_omp.print_stats();

    free_random_data(&data);
}

void selected_sum_benchmark() {
    // summing up selected data benchmarks
    uint32_t *data = nullptr;
    int data_len = 0;
    generate_random_data(130'000'000, &data, &data_len);
    const int ITERATIONS = 300;

    uint32_t ret = 0;
    auto naive = Benchmark("selected-naive-sum", ITERATIONS, [&]() {
        ret = selected_naive_sum(data, data_len);
    });
    naive.run();
    std::cout << "selected-naive: " << ret << "\n";

    auto simd = Benchmark("selected-simd-sum", ITERATIONS, [&]() {
        ret = selected_simd_sum(data, data_len);
    });
    simd.run();
    std::cout << "selected-simd: " << ret << "\n";

    auto simd_256 = Benchmark("selected-simd-256-sum", ITERATIONS, [&]() {
        ret = selected_simd_256_sum(data, data_len);
    });
    simd_256.run();
    std::cout << "selected-256-simd: " << ret << "\n";

    auto naive_omp = Benchmark("selected-naive+omp-sum", ITERATIONS, [&]() {
        ret = selected_naive_omp_sum(data, data_len);
    });
    naive_omp.run();
    std::cout << "selected-naive+omp: " << ret << "\n";

    auto simd_omp = Benchmark("selected-simd+omp-sum", ITERATIONS, [&]() {
        ret = selected_simd_512_omp_sum(data, data_len);
    });
    simd_omp.run();
    std::cout << "selected-simd+omp: " << ret << "\n";

    naive.print_stats();
    simd.print_stats();
    simd_256.print_stats();
    naive_omp.print_stats();
    simd_omp.print_stats();

    free_random_data(&data);
}

int main() {
    uint32_t *data = nullptr;
    int data_len = 0;
    generate_random_data(50'000'000, &data, &data_len);
    const int ITERATIONS = 300;

    uint32_t ret = 0;
    auto naive = Benchmark("selected-naive-sum", ITERATIONS, [&]() {
        ret = selected_naive_omp_sum(data, data_len);
    });
    naive.run();
    std::cout << "selected-naive-sum: " << ret << "\n";

    ret = 0;
    auto simd256 = Benchmark("selected-simd-256-sum", ITERATIONS, [&]() {
        ret = selected_simd_256_omp_sum(data, data_len);
    });
    simd256.run();
    std::cout << "selected-simd-256-sum: " << ret << "\n";

    ret = 0;
    auto simd512 = Benchmark("selected-simd-512-sum", ITERATIONS, [&]() {
        ret = selected_simd_512_omp_sum(data, data_len);
    });
    simd512.run();
    std::cout << "selected-simd-512-sum: " << ret << "\n";

    naive.print_stats();
    simd256.print_stats();
    simd512.print_stats();

    return 0;
}
