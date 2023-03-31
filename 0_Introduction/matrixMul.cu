#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "helper_functions.h"
#include "helper_cuda.h"

// Matrix multiplication (CUDA Kernel) on the device: C = A * B
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int m, int n, int k) {
    // Each thread computes an element in the C matrix
    float Cvalue = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Take values in the for loop, taking each row of A and each column of B
    for (int i = 0; i < k; ++i) {
        Cvalue += A[row * k + i] * B[i * n + col];
    }

    C[row * n + col] = Cvalue;
}

// initialzie use constant value
void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

// Run a simple test of matrix multiplication using CUDA
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA, const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    // initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == nullptr) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // cuds stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host mem to device
    checkCudaErrors(
            cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(
            cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // create and start timer
    printf("Computing result using CUDA Kernel...\n");
    MatrixMulCUDA<<<grid, threads, 0, stream >>> (d_C, d_A, d_B, dimsA.y, dimsB.x, dimsA.x);

    printf("done \n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // check result
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsB.y * valB));
        double dot_length = dimsB.y;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsB.y * valB, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");


    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    return 1;
}

int matrixMulMain(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    // select device ID
    int dev = findCudaDevice(argc, (const char **)argv);
    int block_size = 32;

    dim3 dimsA(2 * block_size, 2 * block_size, 1);
    dim3 dimsB(4 * block_size, 2 * block_size, 1);


    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
           dimsB.x, dimsB.y);

    checkCudaErrors(cudaProfilerStart());
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(matrix_result);
}