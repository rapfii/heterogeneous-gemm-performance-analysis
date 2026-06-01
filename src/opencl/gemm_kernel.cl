#define TILE_SIZE 16

__kernel void gemm_tiled(
    __global const float *A,
    __global const float *B,
    __global       float *C,
    const int N)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + local_col;
        if (row < N && a_col < N)
            tileA[local_row][local_col] = A[row * N + a_col];
        else
            tileA[local_row][local_col] = 0.0f;

        int b_row = t * TILE_SIZE + local_row;
        if (b_row < N && col < N)
            tileB[local_row][local_col] = B[b_row * N + col];
        else
            tileB[local_row][local_col] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_row][k] * tileB[k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
