#include <cooperative_groups.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "helper_functions.h"

namespace cg = cooperative_groups;

// this is a kelnel that does no real work
// but runs at least for a specified number of work
__global__ void clock_block(clock_t *d_o, clock_t clock_count) {
    unsigned int start_clock = (unsigned int)clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        unsigned int end_clock = (unsigned int)clock();
        clock_offset = (clock_t)(end_clock - start_clock);
    }
    d_o[0] = clock_offset;
}

// single warp reduction kernnel
__global__ void sum(clock_t *d_clocks, int N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ clock_t s_clocks[32];

    clock_t my_sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    cg::sync(cta);

    for (int i = 16; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }

        cg::sync(cta);
    }

    d_clocks[0] = s_clocks[0];
}

int conCurKernelsMain(int argc, char **argv) {
    int nkernels = 8;                           // number of concurrent kernels
    int nstreams = nkernels + 1;                // use one more stream than concurrnet kernel
    int nbytes = nkernels * sizeof(clock_t);    // number of data types
    float kernel_time = 10;                     // time the kernel should run in ms
    float elapsed_time = 0;                         // timing varibales
    int cuda_device = 0;

    printf("[%s] - strating... \n", argv[0]);

    // get number of kernels if overridden on the command line
    if (checkCmdLineFlag(argc, (const char **)argv, "nkernels")) {
        nkernels = getCmdLineArgumentInt(argc, (const char **)argv, "nkernels");
        nstreams = nkernels + 1;
    }

    cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if ((deviceProp.concurrentKernels == 0)) {
        printf("> GPU does not support concurrent kernel execution\n");
        printf("  CUDA kernel runs will be serialized\n");
    }

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // allocate host memory
    clock_t *h_a = 0;
    checkCudaErrors(cudaMallocHost((void **)&h_a, nbytes));

    // allocate device memory
    clock_t *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(
                cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    // the events are used for synchronization only and hence do not need to
    // record timings this also makes events not introduce global sync points when
    // recorded which is critical to get overlap
    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *)malloc(nkernels * sizeof(cudaEvent_t));
    for (int i = 0; i < nkernels; ++i) {
        checkCudaErrors(
                cudaEventCreateWithFlags(&kernelEvent[i], cudaEventDisableTiming));
    }

    // time execution with nkernels streams
    clock_t total_clocks = 0;
    clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);

    cudaEventRecord(start_event, 0);

    // use different stream to execute kerel
    for (int i = 0; i < nkernels; ++i) {
        clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
        total_clocks += time_clocks;
        checkCudaErrors(cudaEventRecord(kernelEvent[i], streams[i]));
        // make the last stream wait for the kernel event to be recorded
        checkCudaErrors(
                cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0));
    }

    // queue a sum kernel and a copy back to host in the last stream
    // the commands in this stream get dispatched as soon as all the kernel events
    // have been recorded
    sum<<<1, 32, 0, streams[nstreams - 1]>>>(d_a, nkernels);
    checkCudaErrors(cudaMemcpyAsync(h_a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams - 1]));

    // at this point the CPU has dispatched all work for the GPU and can continue
    // processing other tasks in parallel

    // in this sample we just wait until the GPU is done
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    printf("Expected time for serial execution of %d kernels = %.3fs\n", nkernels,
           nkernels * kernel_time / 1000.0f);
    printf("Expected time for concurrent execution of %d kernels = %.3fs\n",
           nkernels, kernel_time / 1000.0f);
    printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

    bool bTestResult = (h_a[0] > total_clocks);

    // release resources
    for (int i = 0; i < nkernels; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(kernelEvent[i]);
    }

    free(streams);
    free(kernelEvent);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(h_a);
    cudaFree(d_a);

    if (!bTestResult) {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}