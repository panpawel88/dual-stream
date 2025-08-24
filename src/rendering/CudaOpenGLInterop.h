#pragma once

#if USE_OPENGL_RENDERER && HAVE_CUDA

#include <glad/gl.h>
#include "core/Logger.h"

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
    
    // Test CUDA interop functionality with a given resource
    bool TestResourceMapping(void* resource, void* stream = nullptr);
    
    // Device capability and validation functions
    bool ValidateDeviceCapabilities();
    bool ValidateTextureSize(int width, int height);
    bool CheckMemoryAvailability(size_t requiredBytes);
    bool ValidateMemoryAlignment(void* ptr, size_t pitch, int width, int height);
    bool ValidateCudaArraySize(void* cudaArray, int expectedWidth, int expectedHeight);
    
private:
    bool m_initialized;
    
    // RAII wrapper for CUDA graphics resource mapping
    class CudaResourceMapper {
    public:
        CudaResourceMapper(CudaOpenGLInterop* interop, void** resources, unsigned int count, void* stream = nullptr)
            : m_interop(interop), m_resources(resources), m_count(count), m_stream(stream), m_mapped(false) {
            if (m_interop && m_resources && m_count > 0) {
                m_mapped = m_interop->MapResources(m_resources, m_count, m_stream);
            }
        }
        
        ~CudaResourceMapper() {
            if (m_mapped && m_interop && m_resources && m_count > 0) {
                if (!m_interop->UnmapResources(m_resources, m_count, m_stream)) {
                    LOG_WARNING("Failed to unmap CUDA graphics resource in destructor");
                }
            }
        }
        
        bool IsValid() const { return m_mapped; }
        
        // Non-copyable
        CudaResourceMapper(const CudaResourceMapper&) = delete;
        CudaResourceMapper& operator=(const CudaResourceMapper&) = delete;
        
        // Movable
        CudaResourceMapper(CudaResourceMapper&& other) noexcept
            : m_interop(other.m_interop), m_resources(other.m_resources), 
              m_count(other.m_count), m_stream(other.m_stream), m_mapped(other.m_mapped) {
            other.m_mapped = false; // Transfer ownership
        }
        
    private:
        CudaOpenGLInterop* m_interop;
        void** m_resources;
        unsigned int m_count;
        void* m_stream;
        bool m_mapped;
    };
    
    // RAII wrapper for CUDA memory allocation
    class CudaMemoryGuard {
    public:
        explicit CudaMemoryGuard(size_t size) : m_ptr(nullptr), m_allocated(false) {
            if (size > 0) {
                // We can't include cuda_runtime.h here, so we'll implement allocation in the cpp file
                m_allocated = AllocateCudaMemory(&m_ptr, size);
            }
        }
        
        ~CudaMemoryGuard() {
            if (m_allocated && m_ptr) {
                FreeCudaMemory(m_ptr);
            }
        }
        
        void* Get() const { return m_ptr; }
        bool IsValid() const { return m_allocated && m_ptr; }
        
        // Non-copyable
        CudaMemoryGuard(const CudaMemoryGuard&) = delete;
        CudaMemoryGuard& operator=(const CudaMemoryGuard&) = delete;
        
        // Movable
        CudaMemoryGuard(CudaMemoryGuard&& other) noexcept
            : m_ptr(other.m_ptr), m_allocated(other.m_allocated) {
            other.m_ptr = nullptr;
            other.m_allocated = false;
        }
        
    private:
        void* m_ptr;
        bool m_allocated;
        
        static bool AllocateCudaMemory(void** ptr, size_t size);
        static void FreeCudaMemory(void* ptr);
    };
    
    
    CudaOpenGLInterop(const CudaOpenGLInterop&) = delete;
    CudaOpenGLInterop& operator=(const CudaOpenGLInterop&) = delete;
};

#endif // USE_OPENGL_RENDERER && HAVE_CUDA