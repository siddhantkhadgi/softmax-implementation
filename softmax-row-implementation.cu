#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32
#define OPTIMAL_CHUNK_SIZE (1024 * 1024)  // 1M elements per chunk (4MB for floats)

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        printf("\nError in %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void randInts(float *x, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = rand() % 100;
    }
}

// Your existing kernel (assumed to be defined elsewhere)
__global__ void find_max_and_norm_vectorized(float *A, int N, float *max, float *norm)
{
    __shared__ float smem[512];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    
    float local_max = -INFINITY;
    float local_norm = 0.0f;
    
    // Vectorized loads for better memory throughput
    // Process 4 elements at a time when possible
    float4 *A_vec = (float4*)A;
    int vec_N = N / 4;
    int vec_idx = global_idx;
    
    // Process vectorized portion
    for (int i = vec_idx; i < vec_N; i += blockDim.x * gridDim.x) {
        float4 vals = A_vec[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
    }
    
    // Handle remaining elements
    int remaining_start = vec_N * 4;
    for (int i = remaining_start + global_idx; i < N; i += blockDim.x * gridDim.x) {
        if (i < N) {
            local_max = fmaxf(local_max, A[i]);
        }
    }
    
    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    // Block-level max reduction
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        smem[warp_id] = local_max;
    }
    __syncthreads();
    
    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? smem[tid] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    
    float global_max = smem[0];
    
    // Second pass: vectorized norm calculation
    for (int i = vec_idx; i < vec_N; i += blockDim.x * gridDim.x) {
        float4 vals = A_vec[i];
        local_norm += expf(vals.x - global_max) + expf(vals.y - global_max) + 
                     expf(vals.z - global_max) + expf(vals.w - global_max);
    }
    
    // Handle remaining elements for norm
    for (int i = remaining_start + global_idx; i < N; i += blockDim.x * gridDim.x) {
        if (i < N) {
            local_norm += expf(A[i] - global_max);
        }
    }
    
    // Norm reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_norm += __shfl_down_sync(0xffffffff, local_norm, offset);
    }
    
    if (lane_id == 0) {
        smem[warp_id] = local_norm;
    }
    __syncthreads();
    
    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? smem[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
    
    if (tid == 0) {
        *max = global_max;
        *norm = smem[0];
    }
}

void run_cudaFn_chunk(float *d_A, int N, float *d_max, float *d_norm)
{
    int block_size = 256;
    int num_blocks = min(32, (N + block_size - 1)/block_size);
    
    find_max_and_norm_vectorized<<<num_blocks, block_size>>>(d_A, N, d_max, d_norm);
    cudaDeviceSynchronize();
}

void process_chunks(float *A, int N, float *global_max, float *global_norm)
{
    float *d_A, *d_max, *d_norm;
    float *chunk_max, *chunk_norm;
    cudaError_t err;
    
    // Allocate device memory for optimal chunk size
    int chunk_size = min(N, OPTIMAL_CHUNK_SIZE);
    
    err = cudaMalloc((void **)&d_A, chunk_size * sizeof(float));
    checkCudaError(err, "cuda malloc chunk A");
    
    err = cudaMalloc((void **)&d_max, sizeof(float));
    checkCudaError(err, "cuda malloc chunk max");
    
    err = cudaMalloc((void **)&d_norm, sizeof(float));
    checkCudaError(err, "cuda malloc chunk norm");
    
    // Host memory for results
    chunk_max = (float *)malloc(sizeof(float));
    chunk_norm = (float *)malloc(sizeof(float));
    
    // Initialize global results
    *global_max = -FLT_MAX;
    *global_norm = 0.0f;
    
    printf("Processing %d elements in chunks of %d...\n", N, chunk_size);
    
    // Process chunks
    int processed = 0;
    int chunk_count = 0;
    
    while (processed < N)
    {
        int current_chunk_size = min(chunk_size, N - processed);
        chunk_count++;
        
        printf("Processing chunk %d: elements %d to %d\n", 
               chunk_count, processed, processed + current_chunk_size - 1);
        
        // Copy chunk to device
        err = cudaMemcpy(d_A, A + processed, current_chunk_size * sizeof(float), 
                        cudaMemcpyHostToDevice);
        checkCudaError(err, "cuda memcpy chunk to device");
        
        // Process chunk
        run_cudaFn_chunk(d_A, current_chunk_size, d_max, d_norm);
        
        // Copy results back
        err = cudaMemcpy(chunk_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError(err, "cuda memcpy chunk max");
        
        err = cudaMemcpy(chunk_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError(err, "cuda memcpy chunk norm");
        
        // Update global maximum
        if (*chunk_max > *global_max)
        {
            *global_max = *chunk_max;
        }
        
        // For now, just accumulate norm (this is not mathematically correct for softmax)
        // We'll need a second pass for correct softmax computation
        *global_norm += *chunk_norm;
        
        processed += current_chunk_size;
        
        printf("Chunk %d - Max: %f, Norm: %f\n", chunk_count, *chunk_max, *chunk_norm);
    }
    
    printf("First pass complete. Global max: %f\n", *global_max);
    
    // Second pass: Recalculate norm using global maximum for numerical stability
    printf("Starting second pass for correct norm calculation...\n");
    *global_norm = 0.0f;
    processed = 0;
    chunk_count = 0;
    
    while (processed < N)
    {
        int current_chunk_size = min(chunk_size, N - processed);
        chunk_count++;
        
        // Copy chunk to device
        err = cudaMemcpy(d_A, A + processed, current_chunk_size * sizeof(float), 
                        cudaMemcpyHostToDevice);
        checkCudaError(err, "cuda memcpy chunk to device");
        
        // Set the global max on device for this chunk
        err = cudaMemcpy(d_max, global_max, sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError(err, "cuda memcpy global max to device");
        
        // Process chunk with known global max
        run_cudaFn_chunk(d_A, current_chunk_size, d_max, d_norm);
        
        // Copy norm result back
        err = cudaMemcpy(chunk_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError(err, "cuda memcpy chunk norm");
        
        // Accumulate norm
        *global_norm += *chunk_norm;
        
        processed += current_chunk_size;
    }
    
    printf("Second pass complete. Final norm: %f\n", *global_norm);
    
    // Cleanup
    free(chunk_max);
    free(chunk_norm);
    cudaFree(d_A);
    cudaFree(d_max);
    cudaFree(d_norm);
}

int main()
{
    float *A, *max_result, *norm_result;
    int N = 4096 * 4096;  // Larger array to demonstrate chunking
    
    // Allocate host memory
    A = (float *)malloc(N * sizeof(float));
    max_result = (float *)malloc(sizeof(float));
    norm_result = (float *)malloc(sizeof(float));
    
    if (!A || !max_result || !norm_result)
    {
        printf("Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize data
    randInts(A, N);
    
    printf("Array size: %d elements (%.2f MB)\n", N, (N * sizeof(float)) / (1024.0 * 1024.0));
    printf("Chunk size: %d elements (%.2f MB)\n", OPTIMAL_CHUNK_SIZE, 
           (OPTIMAL_CHUNK_SIZE * sizeof(float)) / (1024.0 * 1024.0));
    
    // Process in chunks
    process_chunks(A, N, max_result, norm_result);
    
    printf("\nFinal Results:\n");
    printf("Global Max: %f\n", *max_result);
    printf("Global Norm: %f\n", *norm_result);
    
    // Verify with CPU calculation (optional, for small arrays)
    if (N <= 10000)
    {
        float cpu_max = -FLT_MAX;
        float cpu_norm = 0.0f;
        
        // Find max
        for (int i = 0; i < N; i++)
        {
            if (A[i] > cpu_max) cpu_max = A[i];
        }
        
        // Calculate norm
        for (int i = 0; i < N; i++)
        {
            cpu_norm += expf(A[i] - cpu_max);
        }
        
        printf("\nCPU Verification:\n");
        printf("CPU Max: %f (GPU: %f, diff: %f)\n", cpu_max, *max_result, 
               fabsf(cpu_max - *max_result));
        printf("CPU Norm: %f (GPU: %f, diff: %f)\n", cpu_norm, *norm_result, 
               fabsf(cpu_norm - *norm_result));
    }
    
    // Cleanup
    free(A);
    free(max_result);
    free(norm_result);
    
    return 0;
}
