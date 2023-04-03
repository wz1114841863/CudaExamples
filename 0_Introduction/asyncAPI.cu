#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"
#include "helper_functions.h"

__global__ void increment_kernel(int *g_data, int inc_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] += inc_value;
}

__host__ bool correct_output(int *data, const int n, const int x) {
    for (int i = 0; i < n; ++i) {
        if (data[i] != x) {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }
    }
    return true;
}

int asyncAPImain(int argc, char *argv[]) {
    int devID = 0;
    cudaDeviceProp devProp;
    printf("[%s] - Starting... \n", argv[0]);

    devID = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&devProp, devID));
    printf("CUDA device [%s]\n", devProp.name);

    int n = 16 * 1024 * 1024;
    int nbytes = n * (int)sizeof(int);
    int value = 26;

    int *h_a = nullptr;
    checkCudaErrors(cudaMallocHost((void **)&h_a, nbytes));
    memset(h_a, 0, nbytes);

    int *d_a= nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    // do some thing wait for stream
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
           counter);

    // check the output for correctness
    bool bFinalResults = correct_output(h_a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(h_a));
    checkCudaErrors(cudaFree(d_a));

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}