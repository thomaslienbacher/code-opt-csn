cmake_minimum_required(VERSION 3.24)
project(simd_examples_cxx)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS -march=native)

add_executable(simd_examples_cxx main.cpp)
