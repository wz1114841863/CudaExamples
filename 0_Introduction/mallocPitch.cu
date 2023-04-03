#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"


__global__ void mallocPitchKernel(float *devPtr, size_t pitch,
                                  int width, int height) {
    for (int r = 0; r < height; ++r) {
        float *row = (float *)((char *)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}

void mallocPitchMain() {
    int width = 64, height = 64;
    float *devPtr = nullptr;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height));
    mallocPitchKernel<<<100, 512>>>(devPtr, pitch, width, height);
    printf("finish.");
}