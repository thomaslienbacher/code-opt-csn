cmake_minimum_required(VERSION 3.24)
project(simd_examples_cxx)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)

set(CMAKE_CXX_FLAGS -march=native)

add_executable(simd_examples_cxx main.cpp sum.cpp sum.hpp Benchmark.cpp Benchmark.hpp)

target_link_libraries(simd_examples_cxx PUBLIC OpenMP::OpenMP_CXX)
