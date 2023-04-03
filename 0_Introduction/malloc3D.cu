#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"


__global__ void malloc3DKernel(cudaPitchedPtr devPitchedPtr,
                               int width, int height, int depth) {
    char *devPtr = (char *)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char *slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float *row = (float *)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}

void malloc3DMain() {
    int width = 64, height = 64, depth = 64;
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
    cudaPitchedPtr devPitchedPtr;
    checkCudaErrors(cudaMalloc3D(&devPitchedPtr, extent));
    malloc3DKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);
    printf("finish.");
}