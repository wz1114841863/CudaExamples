#include <stdio.h>
#include <iostream>

__global__ void helloWorld() {
    int gridX = blockIdx.x;
    int gridY = blockIdx.y;
    int blockX = threadIdx.x;
    int blockY = threadIdx.y;
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int blockSize = blockDim.y * blockDim.x;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int thread = blockId * blockSize + threadId;
    printf("Hello World print by gird<%d, %d>, block<%d, %d> thread = %d. \n",
           gridX, gridY, blockX, blockY, thread);
}

void helloWorldMain() {
    std::cout << "hello world by cpu" << std::endl;
    dim3 block(3, 4, 1);
    dim3 grid(2, 3, 1);
    helloWorld<<<grid, block>>>();
    cudaDeviceSynchronize();
}