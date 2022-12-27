//
// Created by thomas on 27.12.22.
//

#include "sum.hpp"
#include <immintrin.h>
#include <cassert>
#include <omp.h>

uint32_t naive_sum(const uint32_t *data, const int len) {
    uint32_t sum = 0;

    for (int i = 0; i < len; ++i) {
        sum += data[i];
    }

    return sum;
}

uint32_t simd_sum(uint32_t *data, const int len) {
    assert(len % 16 == 0);
    __m512i sum = _mm512_set1_epi32(0);

    for (int i = 0; i < len; i += 16) {
        sum = _mm512_add_epi32(sum, _mm512_loadu_epi32(data + i));
    }

    uint32_t last = _mm512_reduce_add_epi32(sum);
    return last;
}

uint32_t naive_omp_sum(const uint32_t *data, const int len) {
    uint32_t sum = 0;

#pragma omp parallel for default(none) shared(len, data) reduction(+:sum)
    for (int i = 0; i < len; ++i) {
        sum += data[i];
    }

    return sum;
}

uint32_t simd_omp_sum(uint32_t *data, const int len) {
    assert(len % 16 == 0);
    __m512i sum = _mm512_set1_epi32(0);

#pragma omp declare reduction(_mm512_set1_epi32: __m512i: \
omp_out=_mm512_add_epi32(omp_out, omp_in)) initializer( \
omp_priv=_mm512_set1_epi32(0))

#pragma omp parallel for default(none) shared(len, data) reduction(_mm512_set1_epi32:sum)
    for (int i = 0; i < len; i += 16) {
        sum = _mm512_add_epi32(sum, _mm512_loadu_epi32(data + i));
    }

    uint32_t last = _mm512_reduce_add_epi32(sum);
    return last;
}