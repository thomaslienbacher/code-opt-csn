#include <iostream>
#include <cstdint>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <sstream>
#include "Benchmark.hpp"

#define N (15'000'000)

uint64_t data[N];

uint64_t f(int i) {
    double r = i * 63;
    r *= sqrt(r / 1.4);
    auto a = (uint64_t) r;
    return a + 1;
}

int main() {
    std::cout << "available threads: " << omp_get_num_procs() << "\n";

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
            data[i] = f(i);
        }
    });

    auto b2 = Benchmark("for-omp-parallel-all", 500, [&]() {
#pragma omp parallel for shared(data) default(none)
        for (int i = 0; i < N; ++i) {
            data[i] = f(i);
        }
    });

    auto b3 = Benchmark("for-omp-parallel-3", 500, [&]() {
#pragma omp parallel for shared(data) default(none) num_threads(3)
        for (int i = 0; i < N; ++i) {
            data[i] = f(i);
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
