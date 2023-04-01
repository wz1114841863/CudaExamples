#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_functions.h"

cudaAccessPolicyWindow initAccessPolicyWindow() {
    cudaAccessPolicyWindow accessPolicyWindow{0};
    accessPolicyWindow.base_ptr = nullptr;
    accessPolicyWindow.num_bytes = 0;
    accessPolicyWindow.hitRatio = 0.f;
    accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    return accessPolicyWindow;
}

__global__ static void kernelCacheSegmentTest(int *data, int dataSize, int *trash,
                                              int bigDataSize, int hitCount) {
    __shared__ unsigned int hit;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tID = row * blockDim.y + col;

    uint32_t psRand = tID;

    atomicExch(&hit, 0);
    __syncthreads();

    while (hit < hitCount) {
        psRand ^= psRand << 13;
        psRand ^= psRand >> 17;
        psRand ^= psRand << 5;

        int idx = tID - psRand;
        if (idx < 0) {
            idx = -idx;
        }

        if ((tID % 2) == 0) {
            data[psRand % dataSize] = data[psRand % dataSize] + data[idx % dataSize];
        } else {
            trash[psRand % bigDataSize] = trash[psRand % bigDataSize] + trash[idx % bigDataSize];
        }

        atomicAdd(&hit, 1);
    }
}

void simpleAttributesMain(int argc, char **argv) {
    bool bTestResult = true;
    cudaAccessPolicyWindow accessPolicyWindow;
    cudaDeviceProp deviceProp;
    cudaStream_t stream;
    cudaStreamAttrValue streamAttrValue;
    cudaStreamAttrID streamAttrID;

    int *dataDevicePointer;
    int *dataHostPointer;
    int dataSize;
    int *bigDataDevicePointer;
    int *bigDataHostPointer;
    int bigDataSize;

    // set a timer
    StopWatchInterface *timer = 0;

    printf("%s Starting...\n\n", argv[0]);

    int devID = findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    dim3 threads(32, 32);
    dim3 blocks(deviceProp.maxGridSize[1], 1);

    // Make sure device the l2 optimization
    if (deviceProp.persistingL2CacheMaxSize == 0) {
        printf("Waiving execution as device %d does not support persisting L2 "
                "Caching\n",
                devID);
        exit(EXIT_WAIVED);
    }

    // create stream to assiocate with window
    checkCudaErrors(cudaStreamCreate(&stream));

    // Set the amount of l2 cache that will be persisting to maximum the device can support
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                       deviceProp.persistingL2CacheMaxSize));

    // set stream attribute
    streamAttrID = cudaStreamAttributeAccessPolicyWindow;

    // alloctate data
    streamAttrValue.accessPolicyWindow = initAccessPolicyWindow();
    accessPolicyWindow = initAccessPolicyWindow();

    // allocate size of both buffers
    bigDataSize = (deviceProp.l2CacheSize * 4) / sizeof(int);
    dataSize = (deviceProp.l2CacheSize / 4) / sizeof(int);

    // allocate host data
    checkCudaErrors(cudaMallocHost(&dataHostPointer, dataSize * sizeof(int)));
    checkCudaErrors(cudaMallocHost(&bigDataHostPointer, bigDataSize * sizeof(int)));

    for (int i = 0; i < bigDataSize; ++i) {
        if (i < dataSize) {
            dataHostPointer[i] = i;
        }
        bigDataHostPointer[bigDataSize - i - 1] = i;
    }

    checkCudaErrors(cudaMalloc((void **)&dataDevicePointer, dataSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&bigDataDevicePointer, bigDataSize * sizeof(int)));

    checkCudaErrors(cudaMemcpyAsync(dataDevicePointer, dataHostPointer,
                                    dataSize * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(bigDataDevicePointer, bigDataHostPointer,
                                    bigDataSize * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));

    // make a window for the buffer of interest
    accessPolicyWindow.base_ptr = (void *)dataDevicePointer;
    accessPolicyWindow.num_bytes = dataSize * sizeof(int);
    accessPolicyWindow.hitRatio = 1.f;
    accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    streamAttrValue.accessPolicyWindow = accessPolicyWindow;

    // assign window to stream
    checkCudaErrors(cudaStreamSetAttribute(stream, streamAttrID, &streamAttrValue));

    // demote any privious persisting lines
    checkCudaErrors(cudaCtxResetPersistingL2Cache());

    checkCudaErrors(cudaStreamSynchronize(stream));

    kernelCacheSegmentTest<<<blocks, threads, 0, stream>>>(
            dataDevicePointer, dataSize, bigDataDevicePointer, bigDataSize, 0xAFFFF);

    checkCudaErrors(cudaStreamSynchronize(stream));
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // Free memory
    checkCudaErrors(cudaFreeHost(dataHostPointer));
    checkCudaErrors(cudaFreeHost(bigDataHostPointer));
    checkCudaErrors(cudaFree(dataDevicePointer));
    checkCudaErrors(cudaFree(bigDataDevicePointer));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
