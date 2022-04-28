#ifndef OSQP_MACROS_H
#define OSQP_MACROS_H
#pragma once

#define MATRIX_SIZE (100)
#define MAX_ITER (300)
#define MAX_TOL (1e-4)
#define RHO (0.1)
#define ALPHA (1.6)
#define COMPUTE_OBJ (0)

#include <chrono>
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s: %lfs\n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

#define CUDA OFF
#endif //OSQP_MACROS_H
