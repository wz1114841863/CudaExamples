#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

// cuda kernel function
__global__ void vectorAddAddUniMem(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

// host main function
void vectorAddUniMemMain() {
    // cuda error code
    cudaError_t err = cudaSuccess;

    //  print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(numElements);
    printf("[Vector addition of %d elements] \n", numElements);

    // allocate the unified memory
    float *A = nullptr;
    err = cudaMallocManaged((void **)&A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *B = nullptr;
    err = cudaMallocManaged((void **)&B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *C = nullptr;
    err = cudaMallocManaged((void **)&C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // initialize the host memory
    for (int i = 0; i < numElements; ++i) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    // execute kernel function
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);

    vectorAddAddUniMem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // cuda synchronize
    cudaDeviceSynchronize();

    // free memory
    err = cudaFree(A);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
}