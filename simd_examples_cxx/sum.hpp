//
// Created by thomas on 27.12.22.
//

#ifndef SIMD_EXAMPLES_CXX_SUM_HPP
#define SIMD_EXAMPLES_CXX_SUM_HPP

#include <cstdint>

uint32_t naive_sum(const uint32_t *data, int len);

uint32_t simd_sum(uint32_t *data, int len);

uint32_t naive_omp_sum(const uint32_t *data, int len);

uint32_t simd_omp_sum(uint32_t *data, int len);


#endif //SIMD_EXAMPLES_CXX_SUM_HPP
