#include <cmath>

// copy these examples into godbolt compiler explorer to see the results

void apply_mult(double *__restrict__ a, const double *__restrict__ b) {
    for (int i = 0; i < 4; ++i) {
        a[i] += b[i];
    }
}

// 4 avx instructions
void copy_arr_4(double *__restrict__ a, const double *__restrict__ b) {
    for (int i = 0; i < 4; ++i) {
        a[i] = b[i];
    }
}

// lots of avx instructions
void copy_arr_16(double *__restrict__ a, const double *__restrict__ b) {
    for (int i = 0; i < 16; ++i) {
        a[i] = b[i];
    }
}

// lots lots of avx instructions
void copy_arr_32(double *__restrict__ a, const double *__restrict__ b) {
    for (int i = 0; i < 32; ++i) {
        a[i] = b[i];
    }
}

// use rep instruction
void copy_arr_64(double *__restrict__ a, const double *__restrict__ b) {
    for (int i = 0; i < 1024; ++i) {
        a[i] = b[i];
    }
}

// jump to memcpy
void copy_arr_memcpy(double *__restrict__ a, const double *__restrict__ b) {
    for (int i = 0; i < 1025; ++i) {
        a[i] = b[i];
    }
}

