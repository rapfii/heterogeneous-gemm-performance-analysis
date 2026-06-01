#include "gemm_common.h"
#include <string.h>

enum Mode { MODE_SEQ = 1, MODE_OMP = 2, MODE_OPENCL = 4, MODE_ALL = 7 };

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --mode <seq|omp|opencl|all> --size <N> [--threads <T>]\n"
        "\n"
        "Options:\n"
        "  --mode, -m     seq | omp | opencl | all\n"
        "  --size, -s     Matrix dimension N (e.g. 256, 512, 1024, 2048)\n"
        "  --threads, -t  Number of OpenMP threads (default: 6)\n"
        "  --csv, -c      Output only CSV lines (no verbose info)\n"
        "  --help, -h     Show this help\n", prog);
}

static double benchmark_sequential(const float *A, const float *B,
                                   float *C, int N, int csv_only)
{
    if (!csv_only) printf("[SEQ] Warmup...\n");
    matrix_zero(C, N);
    gemm_sequential(A, B, C, N);

    double total = 0.0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        matrix_zero(C, N);
        double t0 = get_time();
        gemm_sequential(A, B, C, N);
        double t1 = get_time();
        total += (t1 - t0);
        if (!csv_only) printf("  Run %d: %.6f s\n", r + 1, t1 - t0);
    }
    return total / MEASURE_RUNS;
}

static double benchmark_openmp(const float *A, const float *B,
                               float *C, int N, int threads, int csv_only)
{
    if (!csv_only) printf("[OMP] Warmup (threads=%d)...\n", threads);
    matrix_zero(C, N);
    gemm_openmp(A, B, C, N, threads);

    double total = 0.0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        matrix_zero(C, N);
        double t0 = get_time();
        gemm_openmp(A, B, C, N, threads);
        double t1 = get_time();
        total += (t1 - t0);
        if (!csv_only) printf("  Run %d: %.6f s\n", r + 1, t1 - t0);
    }
    return total / MEASURE_RUNS;
}

static double benchmark_opencl(const float *A, const float *B,
                               float *C, int N, const char *kernel_path,
                               int csv_only)
{
    double t_h2d, t_kernel, t_d2h;

    if (!csv_only) printf("[OpenCL] Warmup...\n");
    matrix_zero(C, N);
    if (gemm_opencl(A, B, C, N, kernel_path, &t_h2d, &t_kernel, &t_d2h) != 0) {
        fprintf(stderr, "[OpenCL] Warmup FAILED\n");
        return -1.0;
    }

    double total = 0.0;
    double total_h2d = 0.0, total_kern = 0.0, total_d2h = 0.0;
    for (int r = 0; r < MEASURE_RUNS; r++) {
        matrix_zero(C, N);
        double t0 = get_time();
        if (gemm_opencl(A, B, C, N, kernel_path,
                        &t_h2d, &t_kernel, &t_d2h) != 0) {
            fprintf(stderr, "[OpenCL] Run %d FAILED\n", r + 1);
            return -1.0;
        }
        double t1 = get_time();
        double wall = t1 - t0;
        total += wall;
        total_h2d  += t_h2d;
        total_kern += t_kernel;
        total_d2h  += t_d2h;
        if (!csv_only)
            printf("  Run %d: %.6f s  (H2D=%.6f  Kernel=%.6f  D2H=%.6f)\n",
                   r + 1, wall, t_h2d, t_kernel, t_d2h);
    }
    if (!csv_only)
        printf("  Avg breakdown: H2D=%.6f  Kernel=%.6f  D2H=%.6f\n",
               total_h2d / MEASURE_RUNS,
               total_kern / MEASURE_RUNS,
               total_d2h / MEASURE_RUNS);

    return total / MEASURE_RUNS;
}

int main(int argc, char **argv)
{
    int N        = 0;
    int threads  = 6;
    int mode     = 0;
    int csv_only = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                i++;
                if      (strcmp(argv[i], "seq")    == 0) mode = MODE_SEQ;
                else if (strcmp(argv[i], "omp")    == 0) mode = MODE_OMP;
                else if (strcmp(argv[i], "opencl") == 0) mode = MODE_OPENCL;
                else if (strcmp(argv[i], "all")    == 0) mode = MODE_ALL;
                else { fprintf(stderr, "Unknown mode: %s\n", argv[i]); return 1; }
            } else {
                fprintf(stderr, "Option %s requires an argument.\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) {
                i++;
                N = atoi(argv[i]);
            } else {
                fprintf(stderr, "Option %s requires an argument.\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                i++;
                threads = atoi(argv[i]);
            } else {
                fprintf(stderr, "Option %s requires an argument.\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--csv") == 0 || strcmp(argv[i], "-c") == 0) {
            csv_only = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (N <= 0 || mode == 0) {
        print_usage(argv[0]);
        return 1;
    }

    float *A      = matrix_alloc(N);
    float *B      = matrix_alloc(N);
    float *C_seq  = matrix_alloc(N);
    float *C_work = matrix_alloc(N);

    if (!A || !B || !C_seq || !C_work) {
        fprintf(stderr, "Memory allocation failed for N=%d\n", N);
        return 1;
    }

    matrix_init(A, N, 42);
    matrix_init(B, N, 137);

    if (!csv_only)
        printf("===================================================\n"
               " GEMM Benchmark - N=%d\n"
               "===================================================\n\n", N);

    double t_seq = 0.0;
    if (!csv_only) printf("-- Sequential (baseline) --\n");
    t_seq = benchmark_sequential(A, B, C_seq, N, csv_only);
    if (!csv_only) printf("  Average: %.6f s\n\n", t_seq);

    if (mode & MODE_SEQ) {
        printf("seq,%d,%.6f,1\n", N, t_seq);
    }

    if (mode & MODE_OMP) {
        if (!csv_only) printf("-- OpenMP (threads=%d) --\n", threads);
        double t_omp = benchmark_openmp(A, B, C_work, N, threads, csv_only);
        int valid = validate_result(C_work, C_seq, N, !csv_only);
        if (!csv_only) {
            printf("  Average: %.6f s  Speedup: %.2fx\n\n",
                   t_omp, t_seq / t_omp);
        }
        printf("omp,%d,%.6f,%d\n", N, t_omp, valid);
    }

    if (mode & MODE_OPENCL) {
        const char *kernel_path = "src/opencl/gemm_kernel.cl";

        if (!csv_only) printf("-- OpenCL (GPU) --\n");
        double t_ocl = benchmark_opencl(A, B, C_work, N, kernel_path,
                                        csv_only);
        if (t_ocl < 0) {
            printf("opencl,%d,-1,0\n", N);
        } else {
            int valid = validate_result(C_work, C_seq, N, !csv_only);
            if (!csv_only) {
                printf("  Average: %.6f s  Speedup: %.2fx\n\n",
                       t_ocl, t_seq / t_ocl);
            }
            printf("opencl,%d,%.6f,%d\n", N, t_ocl, valid);
        }
    }

    matrix_free(A);
    matrix_free(B);
    matrix_free(C_seq);
    matrix_free(C_work);

    return 0;
}
