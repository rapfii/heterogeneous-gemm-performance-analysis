#include "../gemm_common.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

static char *load_kernel_source(const char *path, size_t *out_len)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[OpenCL] Cannot open kernel file: %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    size_t len = (size_t)ftell(fp);
    rewind(fp);
    char *src = (char *)malloc(len + 1);
    if (!src) { fclose(fp); return NULL; }
    size_t read = fread(src, 1, len, fp);
    src[read] = '\0';
    fclose(fp);
    if (out_len) *out_len = read;
    return src;
}

#define CL_CHECK(err, msg) do {                                       \
    if ((err) != CL_SUCCESS) {                                        \
        fprintf(stderr, "[OpenCL] %s  (error %d)\n", (msg), (err));   \
        ret = -1; goto cleanup;                                       \
    }                                                                 \
} while (0)

int gemm_opencl(const float *A, const float *B, float *C, int N,
                const char *kernel_path,
                double *t_h2d, double *t_kernel, double *t_d2h)
{
    int ret = 0;
    cl_int err;

    cl_platform_id   platform   = NULL;
    cl_device_id     device     = NULL;
    cl_context       context    = NULL;
    cl_command_queue  queue     = NULL;
    cl_program       program    = NULL;
    cl_kernel        kernel     = NULL;
    cl_mem           d_A = NULL, d_B = NULL, d_C = NULL;
    char            *src        = NULL;

    size_t mat_bytes = (size_t)N * N * sizeof(float);

    err = clGetPlatformIDs(1, &platform, NULL);
    CL_CHECK(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CL_CHECK(err, "clGetDeviceIDs (GPU)");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    fprintf(stderr, "  OpenCL device: %s\n", dev_name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");

    queue = clCreateCommandQueue(context, device, 0, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    size_t src_len;
    src = load_kernel_source(kernel_path, &src_len);
    if (!src) { ret = -1; goto cleanup; }

    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&src, &src_len, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-mad-enable", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_len);
        char *log = (char *)malloc(log_len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_len, log, NULL);
        fprintf(stderr, "[OpenCL] Build log:\n%s\n", log);
        free(log);
        ret = -1; goto cleanup;
    }

    kernel = clCreateKernel(program, "gemm_tiled", &err);
    CL_CHECK(err, "clCreateKernel");

    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,  mat_bytes, NULL, &err);
    CL_CHECK(err, "clCreateBuffer A");
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,  mat_bytes, NULL, &err);
    CL_CHECK(err, "clCreateBuffer B");
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mat_bytes, NULL, &err);
    CL_CHECK(err, "clCreateBuffer C");

    double t0 = get_time();
    err  = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, mat_bytes, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, mat_bytes, B, 0, NULL, NULL);
    clFinish(queue);
    double t1 = get_time();
    *t_h2d = t1 - t0;

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int),    &N);
    CL_CHECK(err, "clSetKernelArg");

    size_t tile = 16;
    size_t global[2] = {
        ((N + tile - 1) / tile) * tile,
        ((N + tile - 1) / tile) * tile
    };
    size_t local[2] = { tile, tile };

    t0 = get_time();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    CL_CHECK(err, "clEnqueueNDRangeKernel");
    clFinish(queue);
    t1 = get_time();
    *t_kernel = t1 - t0;

    t0 = get_time();
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, mat_bytes, C,
                               0, NULL, NULL);
    clFinish(queue);
    t1 = get_time();
    *t_d2h = t1 - t0;

cleanup:
    if (d_A)     clReleaseMemObject(d_A);
    if (d_B)     clReleaseMemObject(d_B);
    if (d_C)     clReleaseMemObject(d_C);
    if (kernel)  clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue)   clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    free(src);

    return ret;
}
