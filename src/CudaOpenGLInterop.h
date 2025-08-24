#pragma once

#if USE_OPENGL_RENDERER && HAVE_CUDA

#include <glad/gl.h>

// Use opaque handles for CUDA types to avoid header conflicts
// All CUDA types are represented as void* in headers

class CudaOpenGLInterop {
public:
    CudaOpenGLInterop();
    ~CudaOpenGLInterop();
    
    // Initialize CUDA/OpenGL interop
    bool Initialize();
    void Cleanup();
    
    // Register OpenGL texture with CUDA
    bool RegisterTexture(GLuint textureID, void** resource);
    bool UnregisterTexture(void* resource);
    
    // Map/unmap resources for access
    bool MapResources(void** resources, unsigned int count, void* stream = nullptr);
    bool UnmapResources(void** resources, unsigned int count, void* stream = nullptr);
    
    // Get CUDA array from mapped resource
    bool GetMappedArray(void* resource, void** array);
    
    // Copy CUDA device memory to OpenGL texture
    bool CopyDeviceToTexture(void* srcPtr, size_t srcPitch, 
                           void* dstResource, 
                           int width, int height, 
                           void* stream = nullptr);
    
    // Copy CUDA device memory to OpenGL texture with YUV to RGBA conversion
    bool CopyYuvToTexture(void* yuvPtr, size_t yuvPitch, 
                         void* dstResource, 
                         int width, int height, 
                         void* stream = nullptr);
    
    // Utility functions
    bool IsInitialized() const { return m_initialized; }
    
private:
    bool m_initialized;
    
    
    // Disable copy constructor and assignment
    CudaOpenGLInterop(const CudaOpenGLInterop&) = delete;
    CudaOpenGLInterop& operator=(const CudaOpenGLInterop&) = delete;
};

#endif // USE_OPENGL_RENDERER && HAVE_CUDA