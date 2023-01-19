#include <iostream>
#include <cstdint>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <sstream>
#include "Benchmark.hpp"

#define N (20'000'000)

uint64_t data[N];

int main() {
    std::cout << "omp num procs: " << omp_get_num_procs() << "\n";

    int priv = 0;
    std::atomic<int> shared = 0;
#pragma omp parallel default(none) private(priv) shared(stdout, shared)
    {
        int id = omp_get_thread_num();
        priv++;
        fprintf(stdout, "thread: %d  private: %d  shared: %d\n", id, priv, shared.fetch_add(1));
    }

    auto b1 = Benchmark("simple-for", 500, [&]() {
        for (int i = 0; i < N; ++i) {
            data[i] = i * 63;
            double r = sqrt((double) i);
            auto *ptr = (uint64_t *) (&r);
            data[i] += *ptr;
        }
    });

    auto b2 = Benchmark("for-omp-parallel-all", 500, [&]() {
#pragma omp parallel for shared(data) default(none)
        for (int i = 0; i < N; ++i) {
            data[i] = i * 63;
            double r = sqrt((double) i);
            auto *ptr = (uint64_t *) (&r);
            data[i] += *ptr;
        }
    });

    auto b3 = Benchmark("for-omp-parallel-3", 500, [&]() {
#pragma omp parallel for shared(data) default(none) num_threads(3)
        for (int i = 0; i < N; ++i) {
            data[i] = i * 63;
            double r = sqrt((double) i);
            auto *ptr = (uint64_t *) (&r);
            data[i] += *ptr;
        }
    });

    b1.run();
    b2.run();
    b3.run();
    b1.print_stats();
    b2.print_stats();
    b3.print_stats();

    return 0;
}
