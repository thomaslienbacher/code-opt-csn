//
// Created by thomas on 27.12.22.
//

#include "sum.hpp"
#include <immintrin.h>
#include <cassert>
#include <omp.h>

uint32_t naive_sum(const uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    uint32_t sum = 0;

    for (int i = 0; i < len; ++i) {
        sum += data[i];
    }

    return sum;
}

uint32_t simd_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    __m512i sum = _mm512_set1_epi32(0);

    for (int i = 0; i < len; i += 16) {
        sum = _mm512_add_epi32(sum, _mm512_loadu_epi32(data + i));
    }

    uint32_t last = _mm512_reduce_add_epi32(sum);
    return last;
}

uint32_t simd_256_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    __m256i sum = _mm256_set1_epi32(0);

    for (int i = 0; i < len; i += 8) {
        sum = _mm256_add_epi32(sum, _mm256_loadu_epi32(data + i));
    }

    uint32_t last_row[8];
    _mm256_storeu_epi32(last_row, sum);

    uint32_t last = last_row[0] + last_row[1] + last_row[2] + last_row[3] +
                    last_row[4] + last_row[5] + last_row[6] + last_row[7];
    return last;
}

uint32_t naive_omp_sum(const uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    uint32_t sum = 0;

#pragma omp parallel for default(none) shared(len, data) reduction(+:sum)
    for (int i = 0; i < len; ++i) {
        sum += data[i];
    }

    return sum;
}

uint32_t simd_omp_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
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

uint32_t selected_naive_sum(const uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    uint32_t sum = 0;

    for (int i = 0; i < len; ++i) {
        if (data[i] == 1) {
            sum += data[i];
        }
    }

    return sum;
}

uint32_t selected_simd_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    __m512i sum = _mm512_set1_epi32(0);
    const __m512i ones = _mm512_set1_epi32(1);

    for (int i = 0; i < len; i += 64) {
        __m512i elem = _mm512_load_epi32(data + i);
        __mmask16 mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm512_load_epi32(data + i + 16);
        mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm512_load_epi32(data + i + 32);
        mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm512_load_epi32(data + i + 48);
        mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);
    }

    uint32_t last = _mm512_reduce_add_epi32(sum);
    return last;
}

uint32_t selected_simd_256_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    __m256i sum = _mm256_set1_epi32(0);
    const __m256i ones = _mm256_set1_epi32(1);

    for (int i = 0; i < len; i += 64) {
        __m256i elem = _mm256_load_epi32(data + i);
        __mmask8 mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 8);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 16);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 24);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 32);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 40);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 48);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 56);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);
    }

    uint32_t last_row[8];
    _mm256_storeu_epi32(last_row, sum);

    uint32_t last = last_row[0] + last_row[1] + last_row[2] + last_row[3] +
                    last_row[4] + last_row[5] + last_row[6] + last_row[7];
    return last;
}

uint32_t selected_naive_omp_sum(const uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    uint32_t sum = 0;

#pragma omp parallel for default(none) shared(len, data) reduction(+:sum)
    for (int i = 0; i < len; ++i) {
        if (data[i] == 1) {
            sum += data[i];
        }
    }

    return sum;
}

uint32_t selected_simd_256_omp_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    __m256i sum = _mm256_set1_epi32(0);
    const __m256i ones = _mm256_set1_epi32(1);

    for (int i = 0; i < len; i += 64) {
        __m256i elem = _mm256_load_epi32(data + i);
        __mmask8 mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 8);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 16);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 24);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 32);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 40);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 48);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm256_load_epi32(data + i + 56);
        mask = _mm256_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm256_mask_add_epi32(sum, mask, sum, elem);
    }

    uint32_t last_row[8];
    _mm256_storeu_epi32(last_row, sum);

    uint32_t last = last_row[0] + last_row[1] + last_row[2] + last_row[3] +
                    last_row[4] + last_row[5] + last_row[6] + last_row[7];
    return last;
}

uint32_t selected_simd_512_omp_sum(uint32_t *data, const int len) {
    assert(len % 64 == 0);
    assert((size_t) data % 64 == 0);
    __m512i sum = _mm512_set1_epi32(0);
    const __m512i ones = _mm512_set1_epi32(1);

#pragma omp declare reduction(_mm512_set1_epi32: __m512i: \
omp_out=_mm512_add_epi32(omp_out, omp_in)) initializer( \
omp_priv=_mm512_set1_epi32(0))

    // simd might be faster because of loop unrolling
#pragma omp parallel for default(none) shared(len, data, ones) reduction(_mm512_set1_epi32:sum)
    for (int i = 0; i < len; i += 64) {
        __m512i elem = _mm512_load_epi32(data + i);
        __mmask16 mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm512_load_epi32(data + i + 16);
        mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm512_load_epi32(data + i + 32);
        mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);

        elem = _mm512_load_epi32(data + i + 48);
        mask = _mm512_cmp_epu32_mask(elem, ones, _MM_CMPINT_EQ);
        sum = _mm512_mask_add_epi32(sum, mask, sum, elem);
    }

    uint32_t last = _mm512_reduce_add_epi32(sum);
    return last;
}