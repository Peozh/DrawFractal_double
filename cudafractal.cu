#include "cudafractal.h"

inline __device__ cuDoubleComplex exponentialForm(const cuDoubleComplex zn)
{
    double r = std::sqrt(zn.x*zn.x + zn.y*zn.y);
    double w = std::atan(zn.y / zn.x);
    return { r, w };
}

inline __device__ cuDoubleComplex computeNext(cuDoubleComplex zn_ex, const double exponent, const cuDoubleComplex constant)
{
    // Zn^2 + C
    zn_ex.x = pow(zn_ex.x, exponent);
    zn_ex.y *= exponent;
    cuDoubleComplex zn { zn_ex.x*cos(zn_ex.y), zn_ex.x*sin(zn_ex.y) };
    zn.x += constant.x;
    zn.y += constant.y;

    return exponentialForm(zn);
}

inline __device__ float computeIterations(const cuDoubleComplex z0, const double exponent, const cuDoubleComplex constant, const size_t max_iteration)
{
    cuDoubleComplex zn_ex = exponentialForm(z0);
    int iteration = 0;
    while ((zn_ex.x < 1000.0) && (iteration < max_iteration))
    {
        zn_ex = computeNext(zn_ex, exponent, constant);
        ++iteration;
    }
    const double norm = zn_ex.x;
    const float floatIteration = double(iteration) - log2(max(1.0, log2(norm)));
    return floatIteration;
}

inline __device__ float getColor(float iteration, int maxIteration)
{
    return iteration/maxIteration;
}

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
    bool log_expression)
{
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int xIdx = idx % width;
    const unsigned int yIdx = idx / width;
    const float x = xIdx;
    const float y = height - yIdx;
    const double x_polar = (x - x_st) * scale;
    const double y_polar = (y - y_st) * scale;

    cuDoubleComplex z0 {x_polar, y_polar};
    int iteration = computeIterations(z0, exponent, constant, max_iteration);

    if (log_expression)
    {
        float intensity = getColor(iteration, max_iteration) + 0.0001f;
        float logIntensity = max(0.0f, -log2(intensity));
        logIntensity *= 1000;
        logIntensity -= int(logIntensity);
        pixels[idx] = logIntensity;
    }
    else
    {
        float intensity = getColor(iteration, max_iteration);
        pixels[idx] = intensity;
    }
}

void myCUDA::printCudaDevice(int argc, char **argv)
{
    printf("Printing CUDA Device... [%s]\n", __FILE__);
    findCudaDevice(argc, (const char **)argv);
}

void myCUDA::allocateHostPinnedMemory(float*& host_texture, const size_t width, const size_t height)
{
    if (width == 0) return;
    if (height == 0) return;
    if (host_texture != nullptr) return;
    constexpr size_t threadsPerBlock = 32;
    const size_t pixelCount = width*height;
    checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&host_texture), sizeof(float)*(pixelCount + threadsPerBlock), cudaHostAllocPortable));
}
void myCUDA::allocateDeviceMemory(float*& dev_texture, const size_t width, const size_t height)
{
    if (width == 0) return;
    if (height == 0) return;
    if (dev_texture != nullptr) return;
    constexpr size_t threadsPerBlock = 32;
    const size_t pixelCount = width*height;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dev_texture), sizeof(float)*(pixelCount + threadsPerBlock)));
}

void myCUDA::deleteHostPinnedMemory(float*& host_texture)
{
    if (host_texture == nullptr) return;
    checkCudaErrors(cudaFreeHost(reinterpret_cast<void *>(host_texture)));
    host_texture = nullptr;
}
void myCUDA::deleteDeviceMemory(float*& dev_texture)
{
    if (dev_texture == nullptr) return;
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(dev_texture)));
    dev_texture = nullptr;
}

void myCUDA::generateTexture(
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
    const bool log_expression)
{
    if (host_texture == nullptr) return;
    if (dev_texture == nullptr) return;
    if (width == 0) return;
    if (height == 0) return;

    constexpr size_t threadsPerBlock = 32;
    size_t pixelCount = width*height;
    size_t blockCount = (pixelCount + threadsPerBlock)/threadsPerBlock;
    dim3 gridDim(blockCount, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);
    calcPixelValue<<<gridDim, blockDim>>>(dev_texture, width, height, exponent, constant, max_iteration, x_st, y_st, scale, log_expression);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(host_texture, dev_texture, sizeof(float)*(pixelCount), cudaMemcpyDeviceToHost));
}