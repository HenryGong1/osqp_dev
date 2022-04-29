cmake_minimum_required(VERSION 3.17)
project(osqp)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)
enable_language(CUDA)
include(functions.h)
add_executable(osqp main.cu)
target_link_libraries (osqp cublas cusparse cusolver)
