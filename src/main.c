#include "gemm_common.h"
#include <string.h>
#include <getopt.h>

enum Mode { MODE_SEQ = 1, MODE_OMP = 2, MODE_OPENCL = 4, MODE_ALL = 7 };

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --mode <seq|omp|opencl|all> --size <N> [--threads <T>]\n"
        "\n"
        "Options:\n"
        "  --mode     seq | omp | opencl | all\n"
        "  --size     Matrix dimension N (e.g. 256, 512, 1024, 2048)\n"
        "  --threads  Number of OpenMP threads (default: 6)\n"
        "  --csv      Output only CSV lines (no verbose info)\n"
        "  --help     Show this help\n", prog);
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

    static struct option long_opts[] = {
        { "mode",    required_argument, NULL, 'm' },
        { "size",    required_argument, NULL, 's' },
        { "threads", required_argument, NULL, 't' },
        { "csv",     no_argument,       NULL, 'c' },
        { "help",    no_argument,       NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:s:t:ch", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'm':
            if      (strcmp(optarg, "seq")    == 0) mode = MODE_SEQ;
            else if (strcmp(optarg, "omp")    == 0) mode = MODE_OMP;
            else if (strcmp(optarg, "opencl") == 0) mode = MODE_OPENCL;
            else if (strcmp(optarg, "all")    == 0) mode = MODE_ALL;
            else { fprintf(stderr, "Unknown mode: %s\n", optarg); return 1; }
            break;
        case 's': N        = atoi(optarg); break;
        case 't': threads  = atoi(optarg); break;
        case 'c': csv_only = 1;            break;
        case 'h': print_usage(argv[0]);    return 0;
        default:  print_usage(argv[0]);    return 1;
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
