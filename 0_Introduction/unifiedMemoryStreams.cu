// This sample implements a simple task consumer using threads and streams
// with all data in Unified Memory, and tasks consumed by both host and device
#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#ifdef USE_PTHREADS     // Supplementary definition of USE_PTHREADS when using pthreads
#include <pthread.h>
#else
#include <omp.h>        // openMP
#endif

#include <cublas_v2.h>  // CUDA Basic Linear Algebra Subroutine
#include "helper_cuda.h"

// #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// // SRAND48 and DRAND48 don't exist on windows, but these are the equivalent
// // functions
// void srand48(long seed) { srand((unsigned int)seed); }
// double drand48() { return double(rand()) / RAND_MAX; }
// #endif
void srand48(long seed) { srand((unsigned int)seed); }
double drand48() { return double(rand()) / RAND_MAX; }

const char *sSDKname = "UnifiedMemoryStreams";

// simple task
template <typename T>
struct Task {
    unsigned int size, id;
    T *data;
    T *result;
    T *vector;

    Task():size(0), id(0), data(nullptr), result(nullptr), vector(nullptr) {};
    Task(unsigned  int s):size(s), id(0), data(nullptr), result(nullptr), vector(nullptr){
        // allocate unified memory -- the operation performed in this example will be a DGEMV
        checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
        checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
        checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~Task() {
        // ensure all memory is deallocated
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(data));
        checkCudaErrors(cudaFree(result));
        checkCudaErrors(cudaFree(vector));
    }

    void allocate (const unsigned int s, const unsigned int unique_id) {
        // allocate unified memory outsife of constructor
        id = unique_id;
        size = s;
        checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
        checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
        checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
        checkCudaErrors(cudaDeviceSynchronize());

        // populate data with random elements
        for (unsigned int i = 0; i < size * size; ++i) {
            data[i] = drand48();
        }

        for (unsigned int i = 0; i < size; ++i) {
            result[i] = 0;
            vector[i] = drand48();
        }
    }
};

#ifdef USE_PTHREADS
typedef struct threadData_t {
    int tid;
    Task<double> *TaskListPtr;
    cudaStream_t *streams;
    cublasHandle_t *handles;
    int taskSize;
}threadData;
#endif

// simple host dgemv: assume data is in row-major format and square
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result) {
    // rows
    for (int i = 0; i < n; ++i) {
        result[i] *= beta;
        for (int j = 0; j < n; ++j) {
            result[i] += A[i * n + j] * x[j];
        }
    }
}

// execute a single task on either host or device depending on size
#ifdef USE_PTHREADS
void *execute(void *inArgs) {
    threadData *dataPtr = (threadData *)inpArgs;
    cudaStream_t
}
#else
#endif