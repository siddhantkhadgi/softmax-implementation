#include <stdio.h>
#include <stdlib.h>

#define N 32
#define warp_size 32
#define max_threads_per_block 1024
#define warps_per_block max_threads_per_block / warp_size

// Helper function for error checking
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        printf("\nError in %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void randInts(int *x, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = rand() % 100;
    }
}

__global__ void cudaFn(int *A, float *softmax)
{
    __shared__ float smem[1024];

    // tid = threadIndex
    int tid = threadIdx.x;

    // printf("A:%d", A[0]);
    int local_max = -INFINITY;
    float local_norm = 0.0f;

    for (int i = tid; i < N; i += blockDim.x)
    {
        float x = A[i];
        if (x > local_max)
        {
            local_norm += expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
    {
        int compare_val = __shfl_down_sync(0xffffffff, val, offset);
        // printf("comparing %d and %d\n", val, compare_val);
        val = fmaxf(val, compare_val);
        // printf("after")
    }

    // printf("%d, ", val);
    if (blockDim.x > warp_size)
    {
        if (tid % warp_size == 0)
        {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        if (tid < warp_size)
        {
            val = (tid < ((blockDim.x + warp_size - 1) / (warp_size))) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2)
            {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0)
                smem[0] = val;
        }
    }
    else
    {
        if (tid == 0)
            smem[0] = val;
    }
    __syncthreads();

    int max = smem[0];

    int max_diff = local_max - max;
    float exp_max_diff = expf(max_diff);
    val = local_norm * expf(local_max - max);

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __syncthreads();

    // printf("val: %f\n", val);
    if (blockDim.x > warp_size)
    {
        if (tid % warp_size == 0)
        {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        if (tid < warp_size)
        {
            val = (tid < ((blockDim.x + warp_size - 1) / (warp_size))) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2)
            {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0)
                smem[0] = val;
        }
    }
    else
    {
        if (tid == 0)
            smem[0] = val;
    }
    __syncthreads();

    float norm = smem[0];
    __syncthreads();

    // printf("norm: %f\n", norm);

    for (int i = tid; i < N; i += blockDim.x)
    {
        softmax[i] = expf(A[i] - max) / norm;
    }
    __syncthreads();
}

void run_cudaFn(int *__restrict__ A, float *__restrict__ B)
{
    int total_warps = (N + warp_size - 1) / (warp_size);
    int total_blocks = (total_warps + warps_per_block - 1) / (warps_per_block);

    cudaFn<<<total_blocks, warps_per_block>>>(A, B);
    cudaDeviceSynchronize(); // Wait for kernel to complete
    printf("cuda complete\n");
}

int main()
{
    int *A;
    float *B;
    int *d_A;
    float *d_B;
    cudaError_t err;

    A = (int *)malloc(N * sizeof(int));
    B = (float *)malloc(N * sizeof(float));

    err = cudaMalloc((void **)&d_A, N * sizeof(int));
    checkCudaError(err, "cuda malloc A");

    err = cudaMalloc((void **)&d_B, N * sizeof(float));

    randInts(A, N);

    printf("input vector\n");
    for (int i = 0; i < N; i++)
    {
        printf("%d, ", A[i]);
    }
    printf("\n");

    err = cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err, "cuda mem cpy");

    run_cudaFn(d_A, d_B);

    cudaMemcpy(B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("softmax result:\n");
    float sum = 0.0f;

    for (int i = 0; i < N; i++)
    {
        sum += B[i];
        printf("%f\n", sum);
    }

    printf("Sum: %f", sum);

    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}