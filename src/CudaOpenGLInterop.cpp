#include "CudaOpenGLInterop.h"

#if USE_OPENGL_RENDERER && HAVE_CUDA

#include "Logger.h"
#include <iostream>

// Include CUDA headers and redefine types
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaGL.h>

// Redefine the forward declared types to actual CUDA types
#undef CUresult
#undef CUstream
#undef CUarray
#undef CUdeviceptr
#undef CUgraphicsResource
// The actual CUDA types will be used from the headers

CudaOpenGLInterop::CudaOpenGLInterop()
    : m_initialized(false) {
}

CudaOpenGLInterop::~CudaOpenGLInterop() {
    Cleanup();
}

bool CudaOpenGLInterop::Initialize() {
    if (m_initialized) {
        return true;
    }
    
    // Initialize CUDA runtime API
    cudaError_t result = cudaSetDevice(0);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaSetDevice: ", cudaGetErrorString(result));
        return false;
    }
    
    LOG_INFO("CUDA/OpenGL interop initialized successfully");
    m_initialized = true;
    return true;
}

void CudaOpenGLInterop::Cleanup() {
    m_initialized = false;
}

bool CudaOpenGLInterop::RegisterTexture(GLuint textureID, void** resource) {
    if (!m_initialized) {
        LOG_ERROR("CudaOpenGLInterop not initialized");
        return false;
    }
    
    // Use CUDA runtime API instead of driver API
    cudaGraphicsResource** cudaResource = reinterpret_cast<cudaGraphicsResource**>(resource);
    cudaError_t result = cudaGraphicsGLRegisterImage(cudaResource, textureID, GL_TEXTURE_2D, 
                                                     cudaGraphicsRegisterFlagsWriteDiscard);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaGraphicsGLRegisterImage: ", cudaGetErrorString(result));
        return false;
    }
    
    LOG_DEBUG("OpenGL texture ", textureID, " registered with CUDA");
    return true;
}

void CudaOpenGLInterop::UnregisterTexture(void* resource) {
    if (resource) {
        cudaGraphicsResource* cudaResource = static_cast<cudaGraphicsResource*>(resource);
        cudaError_t result = cudaGraphicsUnregisterResource(cudaResource);
        if (result != cudaSuccess) {
            LOG_ERROR("CUDA error in cudaGraphicsUnregisterResource: ", cudaGetErrorString(result));
        }
    }
}

bool CudaOpenGLInterop::MapResources(void** resources, unsigned int count, void* stream) {
    if (!m_initialized || !resources || count == 0) {
        return false;
    }
    
    cudaGraphicsResource** cudaResources = reinterpret_cast<cudaGraphicsResource**>(resources);
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    cudaError_t result = cudaGraphicsMapResources(count, cudaResources, cudaStream);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaGraphicsMapResources: ", cudaGetErrorString(result));
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::UnmapResources(void** resources, unsigned int count, void* stream) {
    if (!m_initialized || !resources || count == 0) {
        return false;
    }
    
    cudaGraphicsResource** cudaResources = reinterpret_cast<cudaGraphicsResource**>(resources);
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    cudaError_t result = cudaGraphicsUnmapResources(count, cudaResources, cudaStream);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaGraphicsUnmapResources: ", cudaGetErrorString(result));
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::GetMappedArray(void* resource, void** array) {
    if (!m_initialized || !array) {
        return false;
    }
    
    cudaGraphicsResource* cudaResource = static_cast<cudaGraphicsResource*>(resource);
    cudaArray_t* cudaArray = reinterpret_cast<cudaArray_t*>(array);
    cudaError_t result = cudaGraphicsSubResourceGetMappedArray(cudaArray, cudaResource, 0, 0);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaGraphicsSubResourceGetMappedArray: ", cudaGetErrorString(result));
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::CopyDeviceToTexture(void* srcPtr, size_t srcPitch, 
                                          void* dstResource, 
                                          int width, int height, 
                                          void* stream) {
    if (!m_initialized) {
        return false;
    }
    
    void* cudaSrcPtr = srcPtr;
    cudaGraphicsResource* cudaDstResource = static_cast<cudaGraphicsResource*>(dstResource);
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    
    // Map the resource
    void* resourcePtr = dstResource;
    if (!MapResources(&resourcePtr, 1, stream)) {
        return false;
    }
    
    // Get the CUDA array from the mapped resource
    void* arrayPtr;
    if (!GetMappedArray(dstResource, &arrayPtr)) {
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    cudaArray_t dstArray = static_cast<cudaArray_t>(arrayPtr);
    
    // CUDA texture copying is not properly implemented yet
    // Return false to indicate CUDA interop is not functional
    LOG_DEBUG("CUDA texture copying not implemented - CUDA interop unavailable");
    
    // Unmap the resource
    UnmapResources(&resourcePtr, 1, stream);
    
    return false; // Indicate CUDA interop is not working
}


#endif // USE_OPENGL_RENDERER && HAVE_CUDA