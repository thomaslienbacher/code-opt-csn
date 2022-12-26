//
// Created by thomas on 26.12.22.
//

#include "Benchmark.hpp"
#include <iostream>

Benchmark::Benchmark(std::string name, int iterations, const std::function<void()> &test) :
        name(std::move(name)), iterations(iterations), test(test) {}

void Benchmark::run() {
    std::cout << "Running bench " << name << " ";
    int progress = 0;
    for (int i = 0; i < iterations; ++i) {
        start_timer();
        test();
        uint64_t duration = stop_timer();
        min = std::min(duration, min);
        max = std::max(duration, max);

        double dur_ms = (double) duration / (1000 * 1000);
        avg = (iterations_in_avg * avg) + dur_ms;
        iterations_in_avg++;
        avg /= iterations_in_avg;

        if ((i * 40) / iterations > progress) {
            progress++;
            std::cout << ".";
            std::cout.flush();
        }
    }

    std::cout << "\n";
}

void Benchmark::print_stats() const {
    double min_ms = (double) min / (1000 * 1000);
    double max_ms = (double) max / (1000 * 1000);

    std::cout << "Benchmark " << name << " with " << iterations << " iters: \n";
    std::cout << "  min: " << min_ms << " ms\n";
    std::cout << "  avg: " << avg << " ms\n";
    std::cout << "  max: " << max_ms << " ms\n";
    std::cout << std::endl;
}

void Benchmark::start_timer() {
    struct timespec timer{};
    clock_gettime(CLOCK_MONOTONIC, &timer);
    timer_start = timer.tv_sec * (1000 * 1000 * 1000) + timer.tv_nsec;
}

uint64_t Benchmark::stop_timer() const {
    struct timespec timer{};
    clock_gettime(CLOCK_MONOTONIC, &timer);
    uint64_t stop_time = timer.tv_sec * (1000 * 1000 * 1000) + timer.tv_nsec;
    return stop_time - timer_start;
}
