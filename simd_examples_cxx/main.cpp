#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <iomanip>

void print_m128(__m128 &a) {
    std::cout << std::setw(11) << a[0] << " ";
    std::cout << std::setw(11) << a[1] << " ";
    std::cout << std::setw(11) << a[2] << " ";
    std::cout << std::setw(11) << a[3] << std::endl;
}

void print_m512d(__m512d &a) {
    std::cout << std::setw(11) << a[0] << " ";
    std::cout << std::setw(11) << a[1] << " ";
    std::cout << std::setw(11) << a[2] << " ";
    std::cout << std::setw(11) << a[3] << " ";
    std::cout << std::setw(11) << a[4] << " ";
    std::cout << std::setw(11) << a[5] << " ";
    std::cout << std::setw(11) << a[6] << " ";
    std::cout << std::setw(11) << a[7] << std::endl;
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

    print_m128(a);
    print_m128(b);
    print_m128(c);
    print_m128(d);

    double __attribute__((aligned(64))) raw_da[8] = {3.0, 5.0, 7.0, 8.0, 9.0, 11.0, 13.0, 14.0};
    double __attribute__((aligned(64))) raw_db[8] = {63.0, 75.0, 87.0, 98.0, 129.0, 1311.0, 1413.0, 1514.0};

    // avx512f instructions
    __m512d da = _mm512_load_pd(raw_da);
    __m512d db = _mm512_load_pd(raw_db);
    __m512d dc = _mm512_div_pd(db, da);
    __m512d dd = _mm512_sqrt_pd(dc);
    __m512d de = _mm512_rsqrt14_pd(db); // 1 / sqrt(..)
    __m512d dz = _mm512_set1_pd(0); // set all values to 1.0

    print_m512d(da);
    print_m512d(db);
    print_m512d(dc);
    print_m512d(dd);
    print_m512d(de);
    print_m512d(dz);

    return 0;
}
