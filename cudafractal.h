#pragma once
#define CUDAFRACTAL_EXPORTS // only activate this when building .dll
#include "cuda_runtime.h"
#include "helpers/helper_cuda.h"
#include "helpers/helper_functions.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#ifdef CUDAFRACTAL_EXPORTS
#define CUDAFRACTAL_API __declspec(dllexport)
#else
#define CUDAFRACTAL_API __declspec(dllimport)
#endif

inline __device__ cuDoubleComplex exponentialForm(const cuDoubleComplex zn);
inline __device__ cuDoubleComplex computeNext(cuDoubleComplex zn_ex, const double exponent, const cuDoubleComplex constant);
inline __device__ float computeIterations(const cuDoubleComplex z0, const double exponent, const cuDoubleComplex constant, const size_t max_iteration);
inline __device__ float getColor(float iteration, int maxIteration);
static __global__ void calcPixelValue(
    float* pixels, 
    const size_t width, 
    const size_t height, 
    const double exponent, 
    const cuDoubleComplex constant, 
    const size_t max_iteration, 
    const double x_st, 
    const double y_st, 
    const double scale, 
    bool log_expression);

#ifdef __cplusplus
extern "C" {
#endif

namespace myCUDA
{
    CUDAFRACTAL_API void printCudaDevice(int argc, char **argv);

    CUDAFRACTAL_API __host__ void allocateHostPinnedMemory(float*& host_texture, const size_t width, const size_t height);
    CUDAFRACTAL_API __host__ void allocateDeviceMemory(float*& dev_texture, const size_t width, const size_t height);
    
    CUDAFRACTAL_API __host__ void deleteHostPinnedMemory(float*& host_texture);
    CUDAFRACTAL_API __host__ void deleteDeviceMemory(float*& dev_texture);

    CUDAFRACTAL_API __host__ void generateTexture(
        float* host_texture, 
        float* dev_texture, 
        const size_t width, 
        const size_t height, 
        const double exponent, 
        const cuDoubleComplex constant, 
        const size_t max_iteration, 
        const double x_st, 
        const double y_st, 
        const double scale, 
        const bool log_expression);
}

#ifdef __cplusplus
}
#endif