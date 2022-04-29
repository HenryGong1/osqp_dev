cmake_minimum_required(VERSION 3.12)
project(osqp)


set(CMAKE_CXX_STANDARD 14)

include(macros.h)
add_executable(osqp main.cpp macros.h)