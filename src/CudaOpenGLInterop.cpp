#include "CudaOpenGLInterop.h"

#if USE_OPENGL_RENDERER && HAVE_CUDA

#include "Logger.h"
#include "CudaYuvConversion.h"
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
        LOG_ERROR("CudaOpenGLInterop not initialized");
        return false;
    }
    
    if (!srcPtr || !dstResource) {
        LOG_ERROR("Invalid source pointer or destination resource");
        return false;
    }
    
    void* cudaSrcPtr = srcPtr;
    cudaGraphicsResource* cudaDstResource = static_cast<cudaGraphicsResource*>(dstResource);
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    
    // Map the resource
    void* resourcePtr = dstResource;
    if (!MapResources(&dstResource, 1, stream)) {
        LOG_ERROR("Failed to map CUDA graphics resource");
        return false;
    }
    
    // Get the CUDA array from the mapped resource
    void* arrayPtr;
    if (!GetMappedArray(dstResource, &arrayPtr)) {
        LOG_ERROR("Failed to get mapped CUDA array");
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    cudaArray_t dstArray = static_cast<cudaArray_t>(arrayPtr);
    
    // Copy from CUDA device memory to CUDA array (which is bound to OpenGL texture)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(cudaSrcPtr, srcPitch, width, height);
    copyParams.dstArray = dstArray;
    copyParams.extent = make_cudaExtent(width, height, 1);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    
    cudaError_t result;
    if (cudaStream) {
        result = cudaMemcpy3DAsync(&copyParams, cudaStream);
    } else {
        result = cudaMemcpy3D(&copyParams);
    }
    
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in cudaMemcpy3D: ", cudaGetErrorString(result));
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    // Synchronize if using a stream to ensure copy is complete
    if (cudaStream) {
        result = cudaStreamSynchronize(cudaStream);
        if (result != cudaSuccess) {
            LOG_ERROR("CUDA error in cudaStreamSynchronize: ", cudaGetErrorString(result));
            UnmapResources(&resourcePtr, 1, stream);
            return false;
        }
    }
    
    // Unmap the resource
    if (!UnmapResources(&resourcePtr, 1, stream)) {
        LOG_ERROR("Failed to unmap CUDA graphics resource");
        return false;
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
    
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    
    // Map the resource
    void* resourcePtr = dstResource;
    if (!MapResources(&resourcePtr, 1, stream)) {
        LOG_ERROR("Failed to map CUDA graphics resource");
        return false;
    }
    
    // Get the CUDA array from the mapped resource
    void* arrayPtr;
    if (!GetMappedArray(dstResource, &arrayPtr)) {
        LOG_ERROR("Failed to get mapped CUDA array");
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    cudaArray_t dstArray = static_cast<cudaArray_t>(arrayPtr);
    
    // Allocate temporary RGBA buffer for conversion
    size_t rgbaSize = width * height * 4; // RGBA = 4 bytes per pixel
    void* tempRgbaBuffer;
    cudaError_t result = cudaMalloc(&tempRgbaBuffer, rgbaSize);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error allocating temporary RGBA buffer: ", cudaGetErrorString(result));
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    // Convert NV12 YUV to RGBA using CUDA kernel
    size_t rgbaPitch = width * 4; // 4 bytes per pixel (RGBA)
    result = convertNv12ToRgba(yuvPtr, yuvPitch, tempRgbaBuffer, rgbaPitch, width, height, cudaStream);
    if (result != cudaSuccess) {
        LOG_ERROR("CUDA error in YUV to RGBA conversion: ", cudaGetErrorString(result));
        cudaFree(tempRgbaBuffer);
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    // Copy converted RGBA data to OpenGL texture array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(tempRgbaBuffer, rgbaPitch, width, height);
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
        cudaFree(tempRgbaBuffer);
        UnmapResources(&resourcePtr, 1, stream);
        return false;
    }
    
    // Synchronize if using a stream to ensure copy is complete
    if (cudaStream) {
        result = cudaStreamSynchronize(cudaStream);
        if (result != cudaSuccess) {
            LOG_ERROR("CUDA error in cudaStreamSynchronize: ", cudaGetErrorString(result));
            cudaFree(tempRgbaBuffer);
            UnmapResources(&resourcePtr, 1, stream);
            return false;
        }
    }
    
    // Clean up temporary buffer
    cudaFree(tempRgbaBuffer);
    
    // Unmap the resource
    if (!UnmapResources(&resourcePtr, 1, stream)) {
        LOG_ERROR("Failed to unmap CUDA graphics resource");
        return false;
    }
    
    LOG_DEBUG("CUDA YUV to RGBA texture copy successful - Size: ", width, "x", height, ", pitch: ", yuvPitch);
    return true;
}


#endif // USE_OPENGL_RENDERER && HAVE_CUDA