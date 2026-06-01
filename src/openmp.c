#include "gemm_common.h"
#include <omp.h>

void gemm_openmp(const float *A, const float *B, float *C,
                 int N, int num_threads)
{
    if (num_threads > 0)
        omp_set_num_threads(num_threads);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
