#include <cooperative_groups.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "helper_functions.h"

namespace cg = cooperative_groups;

int conCurKernelsMain(int argc, char **argv) {
    int nkernels = 8;
    int nstreams = nkernels + 1;
    int nbytes = nkernels * sizeof(clock_t);
    float kernel_time = 10;
    float elapsed_time;
    int cuda_device = 0;
}