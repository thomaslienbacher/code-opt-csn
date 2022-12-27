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

int main() {
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


    // summing up data benchmarks
    const uint32_t N = (200'000'000 / 16) * 16;
    auto *data1 = new uint32_t[N];
    auto *data2 = new uint32_t[N];
    auto *data3 = new uint32_t[N];
    //std::random_device dev;
    std::mt19937 rng(0x44);
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 2);

    std::cout << "Generating " << N << " random numbers..." << std::endl;
    for (int i = 0; i < N; ++i) {
        data1[i] = dist(rng);
        data2[i] = data1[i] + 1;
        data3[i] = data1[i] + 2;
    }

    uint32_t ret = 0;
    auto naive = Benchmark("naive-sum", 500, [&]() {
        ret = naive_sum(data1, N) + naive_sum(data2, N) + naive_sum(data3, N);
    });
    naive.run();
    std::cout << "naive: " << ret << "\n";

    auto simd = Benchmark("simd-sum", 500, [&]() {
        ret = simd_sum(data1, N) + simd_sum(data2, N) + simd_sum(data3, N);
    });
    simd.run();
    std::cout << "simd: " << ret << "\n";

    auto naive_omp = Benchmark("naive+omp-sum", 500, [&]() {
        ret = naive_omp_sum(data1, N) + naive_omp_sum(data2, N) + naive_omp_sum(data3, N);
    });
    naive_omp.run();
    std::cout << "naive+omp: " << ret << "\n";

    auto simd_omp = Benchmark("simd+omp-sum", 500, [&]() {
        ret = simd_omp_sum(data1, N) + simd_omp_sum(data2, N) + simd_omp_sum(data3, N);
    });
    simd_omp.run();
    std::cout << "simd+omp: " << ret << "\n";

    naive.print_stats();
    simd.print_stats();
    naive_omp.print_stats();
    simd_omp.print_stats();

    delete[] data1;
    delete[] data2;
    delete[] data3;

    return 0;
}
