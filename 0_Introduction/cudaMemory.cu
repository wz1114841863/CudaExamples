#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__constant__ float constData[256];

__device__ float devData;

__device__ float *devPtr;



__global__ void cudaMemPrint() {
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData);
    // alter the value
    devData += 2.0f;
}

void cudaMemoryMain() {
    float data[256] = {0};
    checkCudaErrors(cudaMemcpyToSymbol(constData, data, sizeof(data)));
    checkCudaErrors(cudaMemcpyFromSymbol(data, constData, sizeof(data)));

    float value = 3.14f;
    checkCudaErrors(cudaMemcpyToSymbol(devData, &value, sizeof(float)));

    float* ptr;
    checkCudaErrors(cudaMalloc(&ptr, 256 * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(devPtr, &ptr, sizeof(ptr)));

    float *dp = nullptr;
    checkCudaErrors(cudaGetSymbolAddress((void **)&dp,devData));
    checkCudaErrors(cudaMemcpy(dp, &value, sizeof(float), cudaMemcpyHostToDevice));
    cudaMemPrint<<<1, 1>>>();
    checkCudaErrors(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);
    size_t sz = 0;
    checkCudaErrors(cudaGetSymbolSize(&sz, constData));
    printf("Host:   the sizeof devData: %zu\n", sz);
}