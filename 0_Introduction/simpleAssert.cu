#include <windows.h>
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h"

const char *sampleName = "simpleAssert";
bool testResult = true;

__global__ void testKernel(int N) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gtid < N);
}


void runTest(int argc, char **argv) {
    int Nblocks = 2;
    int Nthreads = 32;
    cudaError_t error;

    int devID = findCudaDevice(argc, (const char **)argv);

    dim3 dimGrid(Nblocks);
    dim3 dimBlock(Nthreads);
    printf("Launch kernel to generate assertion failures\n");
    testKernel<<<dimGrid, dimBlock>>>(60);

    // Synchronize (flushes assert output).
    printf("\n-- Begin assert output\n\n");
    error = cudaDeviceSynchronize();
    printf("\n-- End assert output\n\n");

    // Check for errors and failed asserts in asynchronous kernel launch.
    if (error == cudaErrorAssert) {
        printf("Device assert failed as expected, "
                "CUDA error message is: %s\n\n",
                cudaGetErrorString(error));
    }

    testResult = error == cudaErrorAssert;
}

int simpleAssertMain(int argc, char **argv) {
    printf("starting... \n");

    runTest(argc, argv);

    printf("%s completed, returned %s\n", sampleName,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
