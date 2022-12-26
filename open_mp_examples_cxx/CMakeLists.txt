cmake_minimum_required(VERSION 3.24)
project(open_mp_examples_cxx)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)

add_executable(open_mp_examples_cxx main.cpp Benchmark.cpp)

target_link_libraries(open_mp_examples_cxx PUBLIC OpenMP::OpenMP_CXX)

