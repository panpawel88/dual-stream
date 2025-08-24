#include "CudaOpenGLInterop.h"

#if USE_OPENGL_RENDERER && HAVE_CUDA

#include "core/Logger.h"
#include "CudaYuvConversion.h"
#include <iostream>

// Include CUDA headers and redefine types
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaGL.h>


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
    cudaError_t result = cudaGraphicsGLRegisterImage(
        reinterpret_cast<cudaGraphicsResource**>(resource), 
        textureID, 
        GL_TEXTURE_2D, 
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaGraphicsGLRegisterImage: ", cudaGetErrorString(result));
        return false;
    }
    
    LOG_DEBUG("OpenGL texture ", textureID, " registered with CUDA");
    return true;
}

bool CudaOpenGLInterop::UnregisterTexture(void* resource) {
    if (!resource) {
        LOG_WARNING("Attempted to unregister null CUDA resource");
        return false;
    }
    
    cudaError_t result = cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(resource));
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaGraphicsUnregisterResource: ", cudaGetErrorString(result));
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::MapResources(void** resources, unsigned int count, void* stream) {
    if (!m_initialized || !resources || count == 0) {
        return false;
    }
    
    cudaError_t result = cudaGraphicsMapResources(
        count, 
        reinterpret_cast<cudaGraphicsResource**>(resources), 
        static_cast<cudaStream_t>(stream));
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
    
    cudaError_t result = cudaGraphicsUnmapResources(
        count, 
        reinterpret_cast<cudaGraphicsResource**>(resources), 
        static_cast<cudaStream_t>(stream));
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
    
    cudaError_t result = cudaGraphicsSubResourceGetMappedArray(
        reinterpret_cast<cudaArray_t*>(array), 
        static_cast<cudaGraphicsResource*>(resource), 
        0, 0);
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
        LOG_ERROR("CudaOpenGLInterop not initialized");
        return false;
    }
    
    if (!srcPtr || !dstResource) {
        LOG_ERROR("Invalid source pointer or destination resource");
        return false;
    }
    
    // Use RAII wrapper to automatically map/unmap the resource
    CudaResourceMapper resourceMapper(this, &dstResource, 1, stream);
    if (!resourceMapper.IsValid()) {
        LOG_ERROR("Failed to map CUDA graphics resource");
        return false;
    }
    
    // Get the CUDA array from the mapped resource
    void* arrayPtr;
    if (!GetMappedArray(dstResource, &arrayPtr)) {
        LOG_ERROR("Failed to get mapped CUDA array");
        return false;
    }
    
    // Copy from CUDA device memory to CUDA array (which is bound to OpenGL texture)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(srcPtr, srcPitch, width, height);
    copyParams.dstArray = static_cast<cudaArray_t>(arrayPtr);
    copyParams.extent = make_cudaExtent(width, height, 1);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    cudaError_t result;
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    if (cudaStream) {
        result = cudaMemcpy3DAsync(&copyParams, cudaStream);
    } else {
        result = cudaMemcpy3D(&copyParams);
    }
    
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaMemcpy3D: ", cudaGetErrorString(result));
        return false;
    }
    
    // Synchronize if using a stream to ensure copy is complete
    if (cudaStream) {
        result = cudaStreamSynchronize(cudaStream);
        if (result != cudaSuccess) {
            LOG_ERROR("CUDA error in cudaStreamSynchronize: ", cudaGetErrorString(result));
            return false;
        }
    }

    LOG_DEBUG("CUDA texture copy successful - Size: ", width, "x", height, ", pitch: ", srcPitch);
    return true;
}

bool CudaOpenGLInterop::CopyYuvToTexture(void* yuvPtr, size_t yuvPitch, 
                                        void* dstResource, 
                                        int width, int height, 
                                        void* stream) {
    if (!m_initialized) {
        LOG_ERROR("CudaOpenGLInterop not initialized");
        return false;
    }
    
    if (!yuvPtr || !dstResource) {
        LOG_ERROR("Invalid source pointer or destination resource");
        return false;
    }
    
    // Use RAII wrapper to automatically map/unmap the resource
    CudaResourceMapper resourceMapper(this, &dstResource, 1, stream);
    if (!resourceMapper.IsValid()) {
        LOG_ERROR("Failed to map CUDA graphics resource");
        return false;
    }
    
    // Get the CUDA array from the mapped resource
    void* arrayPtr;
    if (!GetMappedArray(dstResource, &arrayPtr)) {
        LOG_ERROR("Failed to get mapped CUDA array");
        return false;
    }
    
    cudaArray_t dstArray = static_cast<cudaArray_t>(arrayPtr);
    
    // Use RAII wrapper for temporary RGBA buffer allocation
    size_t rgbaSize = width * height * 4; // RGBA = 4 bytes per pixel
    CudaMemoryGuard tempRgbaBuffer(rgbaSize);
    if (!tempRgbaBuffer.IsValid()) {
        LOG_ERROR("CUDA error allocating temporary RGBA buffer");
        return false;
    }
    
    // Convert NV12 YUV to RGBA using CUDA kernel
    size_t rgbaPitch = width * 4; // 4 bytes per pixel (RGBA)
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    cudaError_t result = convertNv12ToRgba(yuvPtr, yuvPitch, tempRgbaBuffer.Get(), rgbaPitch, width, height, cudaStream);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in YUV to RGBA conversion: ", cudaGetErrorString(result));
        return false;
    }
    
    // Copy converted RGBA data to OpenGL texture array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(tempRgbaBuffer.Get(), rgbaPitch, width, height);
    copyParams.dstArray = dstArray;
    copyParams.extent = make_cudaExtent(width, height, 1);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    if (cudaStream) {
        result = cudaMemcpy3DAsync(&copyParams, cudaStream);
    } else {
        result = cudaMemcpy3D(&copyParams);
    }
    
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaMemcpy3D (RGBA to texture): ", cudaGetErrorString(result));
        return false;
    }
    
    // Synchronize if using a stream to ensure copy is complete
    if (cudaStream) {
        result = cudaStreamSynchronize(cudaStream);
        if (result != cudaSuccess) {
            LOG_ERROR("CUDA error in cudaStreamSynchronize: ", cudaGetErrorString(result));
            return false;
        }
    }

    LOG_DEBUG("CUDA YUV to RGBA texture copy successful - Size: ", width, "x", height, ", pitch: ", yuvPitch);
    return true;
}

bool CudaOpenGLInterop::TestResourceMapping(void* resource, void* stream) {
    if (!m_initialized) {
        LOG_ERROR("CudaOpenGLInterop not initialized");
        return false;
    }
    
    if (!resource) {
        LOG_ERROR("Invalid resource for testing");
        return false;
    }
    
    // Use RAII wrapper to test resource mapping/unmapping
    CudaResourceMapper resourceMapper(this, &resource, 1, stream);
    if (!resourceMapper.IsValid()) {
        LOG_ERROR("Failed to map CUDA graphics resource - interop not functional");
        return false;
    }

    LOG_INFO("CUDA interop test successful - texture mapping/unmapping works");
    return true;
}

// CudaMemoryGuard helper functions implementation
bool CudaOpenGLInterop::CudaMemoryGuard::AllocateCudaMemory(void** ptr, size_t size) {
    cudaError_t result = cudaMalloc(ptr, size);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA memory allocation failed: ", cudaGetErrorString(result));
        return false;
    }
    return true;
}

void CudaOpenGLInterop::CudaMemoryGuard::FreeCudaMemory(void* ptr) {
    if (ptr) {
        cudaError_t result = cudaFree(ptr);
        if (result != cudaSuccess) {
            LOG_WARNING("CUDA memory deallocation failed: ", cudaGetErrorString(result));
        }
    }
}

#endif // USE_OPENGL_RENDERER && HAVE_CUDA