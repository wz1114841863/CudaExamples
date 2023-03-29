#ifndef CUDA_SAMPLES_HELPER_FUNCTIONS_H
#define CUDA_SAMPLES_HELPER_FUNCTIONS_H

#ifdef WIN32
#pragma warning(disable : 4996)
#endif

// includes, project
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// includes, timer, string parsing, image helpers
#include "exception.h"
#include "helper_image.h"   // helper functions for image compare, dump, data comparisons
#include "helper_string.h"  // helper functions for string parsing
#include "helper_timer.h"   // helper functions for timers

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif //CUDA_SAMPLES_HELPER_FUNCTIONS_H
