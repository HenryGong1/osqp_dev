#ifndef OSQP_MACROS_H
#define OSQP_MACROS_H
#pragma once

#define MATRIX_SIZE (100)
#define MAX_ITER (10000)
#define MAX_TOL (2e-4)
#define RHO (0.1)
#define ALPHA (1.6)
#define COMPUTE_OBJ (0)
#define ELEMENTS_PER_THREAD (8)
#define THREADS_PER_BLOCK   (1024)
#include <chrono>
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lf \n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

#define CUDA OFF

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                            \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
        }                                                                                          \
    }while(0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusparse error");                                            \
        }                                                                                          \
    } while (0)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


#endif //OSQP_MACROS_H
