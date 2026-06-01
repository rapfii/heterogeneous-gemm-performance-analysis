#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-4
#define WARMUP_RUNS  1
#define MEASURE_RUNS 3

static inline float *matrix_alloc(int N)
{
    return (float *)malloc((size_t)N * N * sizeof(float));
}

static inline void matrix_init(float *M, int N, unsigned int seed)
{
    for (int i = 0; i < N * N; i++) {
        seed = seed * 1103515245u + 12345u;
        M[i] = (float)(seed >> 16) / 65536.0f;
    }
}

static inline void matrix_zero(float *M, int N)
{
    memset(M, 0, (size_t)N * N * sizeof(float));
}

static inline void matrix_free(float *M)
{
    free(M);
}

static inline double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline int validate_result(const float *P, const float *S,
                                  int N, int verbose)
{
    double error = 0.0;
    for (int i = 0; i < N * N; i++) {
        error += fabs((double)P[i] - (double)S[i]);
    }
    double avg_error = error / ((double)N * N);

    if (verbose) {
        printf("  Validation: total_error=%.6e  avg_error=%.6e  %s\n",
               error, avg_error,
               avg_error < EPSILON ? "VALID" : "INVALID");
    }
    return avg_error < EPSILON ? 1 : 0;
}

void gemm_sequential(const float *A, const float *B, float *C, int N);

void gemm_openmp(const float *A, const float *B, float *C,
                 int N, int num_threads);

int gemm_opencl(const float *A, const float *B, float *C, int N,
                const char *kernel_path,
                double *t_h2d, double *t_kernel, double *t_d2h);

#endif
