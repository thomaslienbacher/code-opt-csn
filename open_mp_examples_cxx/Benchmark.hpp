//
// Created by thomas on 26.12.22.
//

#ifndef OPEN_MP_EXAMPLES_CXX_BENCHMARK_HPP
#define OPEN_MP_EXAMPLES_CXX_BENCHMARK_HPP

#include <cstdint>
#include <functional>
#include <string>

class Benchmark {
    std::string name;
    int iterations = 0;
    double avg = 0;
    int iterations_in_avg = 0;
    uint64_t min = 0xffffffffffffffff;
    uint64_t max = 0;
    std::function<void()> test;
    uint64_t timer_start = 0;

    void start_timer();

    uint64_t stop_timer() const;

public:
    Benchmark(std::string name, int iterations, const std::function<void()> &test);

    void run();

    void print_stats() const;
};


#endif //OPEN_MP_EXAMPLES_CXX_BENCHMARK_HPP
