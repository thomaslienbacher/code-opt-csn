cmake_minimum_required(VERSION 3.24)
project(branchless_examples)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS)

add_executable(branchless_examples branchless.cpp Benchmark.cpp)
