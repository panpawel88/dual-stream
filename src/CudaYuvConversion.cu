#include "CudaYuvConversion.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to convert NV12 to RGBA
__global__ void nv12ToRgbaKernel(
    const uint8_t* __restrict__ yPlane,     // Y plane (luminance)
    const uint8_t* __restrict__ uvPlane,    // UV plane (chrominance)
    uint8_t* __restrict__ rgbaOut,          // RGBA output
    int width, int height,
    int yPitch, int uvPitch, int rgbaPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Sample Y (luminance) - full resolution
    int yIndex = y * yPitch + x;
    float yVal = static_cast<float>(yPlane[yIndex]) / 255.0f;
    
    // Sample UV (chrominance) - half resolution for NV12
    int uvX = (x / 2) * 2;  // U and V are interleaved, so step by 2
    int uvY = y / 2;
    int uvIndex = uvY * uvPitch + uvX;
    
    float u = (static_cast<float>(uvPlane[uvIndex]) - 128.0f) / 255.0f;     // U component
    float v = (static_cast<float>(uvPlane[uvIndex + 1]) - 128.0f) / 255.0f; // V component
    
    // BT.709 YUV to RGB conversion (same as D3D11 shader)
    float r = yVal + 1.402f * v;
    float g = yVal - 0.344f * u - 0.714f * v;
    float b = yVal + 1.772f * u;
    
    // Clamp to [0, 1] range
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));
    
    // Convert to 8-bit and write RGBA (OpenGL expects RGBA order)
    int rgbaIndex = y * rgbaPitch + x * 4;
    rgbaOut[rgbaIndex + 0] = static_cast<uint8_t>(r * 255.0f); // R
    rgbaOut[rgbaIndex + 1] = static_cast<uint8_t>(g * 255.0f); // G
    rgbaOut[rgbaIndex + 2] = static_cast<uint8_t>(b * 255.0f); // B
    rgbaOut[rgbaIndex + 3] = 255;                               // A
}

extern "C" {

cudaError_t convertNv12ToRgba(
    const void* nv12Data,
    size_t nv12Pitch,
    void* rgbaData,
    size_t rgbaPitch,
    int width, int height,
    cudaStream_t stream)
{
    // NV12 format: Y plane followed by interleaved UV plane
    const uint8_t* yPlane = static_cast<const uint8_t*>(nv12Data);
    const uint8_t* uvPlane = yPlane + (nv12Pitch * height); // UV plane starts after Y plane
    
    // UV plane has half the height of Y plane
    size_t uvPitch = nv12Pitch; // Same pitch as Y plane
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    nv12ToRgbaKernel<<<gridSize, blockSize, 0, stream>>>(
        yPlane, uvPlane,
        static_cast<uint8_t*>(rgbaData),
        width, height,
        static_cast<int>(nv12Pitch),
        static_cast<int>(uvPitch),
        static_cast<int>(rgbaPitch)
    );
    
    // Check for kernel launch errors
    return cudaGetLastError();
}

} // extern "C"