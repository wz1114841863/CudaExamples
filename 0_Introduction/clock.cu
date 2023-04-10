#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_functions.h"

#define NUM_BLOCKS 64
#define NUM_THREADS 256

__global__ static void timeReduction(const float *input, float *output, clock_t *timer) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // if tid == 0 in per block
    if (tid == 0) { timer[bid] = clock(); }

    shared[tid] = input[tid];  // 0 - 256
    shared[tid + blockDim.x] = input[tid + blockDim.x];  // 256 - 511

    for (int d = blockDim.x; d > 0; d /= 2) {  // d = 256,128,64,32,16,8,4,2,1,0
        __syncthreads();

        if (tid < d) {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0) {
                shared[tid] = f1;
            }
        }
    }

    // write result
    if (tid == 0) { output[bid] = shared[0]; }
    __syncthreads();

    if (tid == 0) { timer[bid + gridDim.x] = clock(); }
}

int clockMain(int argc, char **argv) {
    int devID = findCudaDevice(argc, (const char **)argv);

    float *d_input = nullptr;
    float *d_output = nullptr;
    clock_t *d_timer = nullptr;

    clock_t h_timer[NUM_BLOCKS * 2];
    float h_input[NUM_THREADS * 2];

    for (int i = 0; i < NUM_THREADS * 2; i++) {
        h_input[i] = (float)i;
    }

    checkCudaErrors(cudaMalloc((void **)&d_input, sizeof(float) * NUM_THREADS * 2));
    checkCudaErrors(cudaMalloc((void **)&d_output, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cudaMalloc((void **)&d_timer, sizeof(clock_t) * NUM_BLOCKS * 2));

    checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice));
    timeReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS, 0>>>(d_input, d_output, d_timer);
    checkCudaErrors(cudaMemcpy(h_timer, d_timer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_timer));

    long double avgElapsedClocks = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += (long double)(h_timer[i + NUM_BLOCKS] - h_timer[i]);
    }

    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    return EXIT_SUCCESS;
}