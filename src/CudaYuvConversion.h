#pragma once

#if USE_OPENGL_RENDERER && HAVE_CUDA

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Convert NV12 format to RGBA using CUDA
cudaError_t convertNv12ToRgba(
    const void* nv12Data,        // Input NV12 data (Y plane followed by UV plane)
    size_t nv12Pitch,            // Pitch (bytes per row) of NV12 data
    void* rgbaData,              // Output RGBA data
    size_t rgbaPitch,            // Pitch (bytes per row) of RGBA data
    int width, int height,       // Frame dimensions
    cudaStream_t stream          // CUDA stream for async execution
);

#ifdef __cplusplus
}
#endif

#endif // USE_OPENGL_RENDERER && HAVE_CUDA