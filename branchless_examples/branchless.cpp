#include <iostream>
#include <cstring>
#include <random>
#include "Benchmark.hpp"

// this is going to compile into a cmov instruction
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }

    // return a * (a > b) + b * (a <= b);
}

int func(int a, int b) {
    if (a + b > 10) {
        return 4;
    } else {
        return 5;
    }

    // return 4 + (a + b) > 10;
}

void uppercase(char *str, int n) {
    for (int i = 0; i < n; i++) {
        if (str[i] >= 'a' && str[i] <= 'z') {
            str[i] -= 32;
        }
    }
}

void uppercase_mult(char *str, int n) {
    for (int i = 0; i < n; i++) {
        str[i] -= 32 * (str[i] >= 'a' && str[i] <= 'z');
    }
}

void uppercase_shift(char *str, int n) {
    for (int i = 0; i < n; i++) {
        str[i] &= ~((str[i] >= 'a' && str[i] <= 'z') << 5);
    }
}

void uppercase_shift_sub(char *str, int n) {
    for (int i = 0; i < n; i++) {
        str[i] -= (str[i] >= 'a' && str[i] <= 'z') << 5;
    }
}

int main() {
    {
        char s[] = "To Upper test ][().-#+123456789";
        printf("%s => ", s);
        uppercase(s, strlen(s));
        printf("%s\n", s);
    }
    {
        char s[] = "To Upper test ][().-#+123456789";
        printf("%s => ", s);
        uppercase_mult(s, strlen(s));
        printf("%s\n", s);
    }
    {
        char s[] = "To Upper test ][().-#+123456789";
        printf("%s => ", s);
        uppercase_shift(s, strlen(s));
        printf("%s\n", s);
    }

    // performance differs when compiling with and without vector extensions
    // benchmark different uppercase functions
    const int LEN = 20'000'000;
    const int ITERATIONS = 1000;
    char *data = new char[LEN];
    std::mt19937 rng(0x44);
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, 127);

    std::cout << "Generating " << LEN << " random numbers..." << std::endl;
    for (int i = 0; i < LEN; ++i) {
        data[i] = dist(rng) & 0xff;
    }

    auto upp = Benchmark("uppercase", ITERATIONS, [&]() {
        uppercase(data, LEN);
    });
    upp.run();

    auto upp_mult = Benchmark("uppercase-mult", ITERATIONS, [&]() {
        uppercase_mult(data, LEN);
    });
    upp_mult.run();

    auto upp_shift = Benchmark("uppercase-shift", ITERATIONS, [&]() {
        uppercase_shift(data, LEN);
    });
    upp_shift.run();

    auto upp_shift_sub = Benchmark("uppercase-shift-sub", ITERATIONS, [&]() {
        uppercase_shift_sub(data, LEN);
    });
    upp_shift_sub.run();

    upp.print_stats();
    upp_mult.print_stats();
    upp_shift.print_stats();
    upp_shift_sub.print_stats();

    delete[] data;

    return 0;
}
