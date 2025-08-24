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
    
    if (!ValidateDeviceCapabilities()) {
        LOG_ERROR("CUDA device does not meet requirements for OpenGL interop");
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
    
    if (!ValidateTextureSize(width, height)) {
        LOG_ERROR("Texture size validation failed for ", width, "x", height);
        return false;
    }
    
    size_t rgbaSize = static_cast<size_t>(width) * height * 4; // RGBA = 4 bytes per pixel
    if (!CheckMemoryAvailability(rgbaSize)) {
        LOG_ERROR("Insufficient GPU memory for RGBA buffer allocation");
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
    
    if (!ValidateCudaArraySize(dstArray, width, height)) {
        LOG_ERROR("CUDA array size validation failed - OpenGL texture may have wrong dimensions");
        return false;
    }
    
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
    
    if (!ValidateMemoryAlignment(tempRgbaBuffer.Get(), rgbaPitch, width, height)) {
        LOG_ERROR("Memory alignment validation failed before cudaMemcpy3D");
        return false;
    }
    
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


bool CudaOpenGLInterop::ValidateDeviceCapabilities() {
    cudaDeviceProp prop;
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    
    cudaError_t result = cudaGetDeviceProperties(&prop, currentDevice);
    if (result != cudaSuccess) {
        LOG_ERROR("Failed to get device properties for validation: ", cudaGetErrorString(result));
        return false;
    }
    
    // Check minimum compute capability (3.0 for modern CUDA features)
    if (prop.major < 3) {
        LOG_ERROR("Device compute capability ", prop.major, ".", prop.minor, " is too old (minimum 3.0 required)");
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::ValidateTextureSize(int width, int height) {
    cudaDeviceProp prop;
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    
    cudaError_t result = cudaGetDeviceProperties(&prop, currentDevice);
    if (result != cudaSuccess) {
        LOG_ERROR("Failed to get device properties for texture size validation: ", cudaGetErrorString(result));
        return false;
    }
    
    if (width > prop.maxTexture2D[0] || height > prop.maxTexture2D[1]) {
        LOG_ERROR("Texture size ", width, "x", height, " exceeds device limits (", 
                  prop.maxTexture2D[0], "x", prop.maxTexture2D[1], ")");
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::CheckMemoryAvailability(size_t requiredBytes) {
    size_t freeMem = 0, totalMem = 0;
    cudaError_t result = cudaMemGetInfo(&freeMem, &totalMem);
    if (result != cudaSuccess) {
        LOG_ERROR("Failed to get memory info: ", cudaGetErrorString(result));
        return false;
    }
    
    // Add 10% safety margin for memory fragmentation
    size_t safetyMargin = requiredBytes / 10;
    size_t totalRequired = requiredBytes + safetyMargin;
    
    if (totalRequired > freeMem) {
        LOG_ERROR("Insufficient GPU memory - Required: ", totalRequired / (1024*1024), 
                  " MB, Available: ", freeMem / (1024*1024), " MB");
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::ValidateMemoryAlignment(void* ptr, size_t pitch, int width, int height) {
    if (!ptr) {
        LOG_ERROR("Invalid pointer for alignment validation");
        return false;
    }
    
    // Validate pitch is at least as wide as the row
    size_t minPitch = static_cast<size_t>(width) * 4; // 4 bytes per RGBA pixel
    if (pitch < minPitch) {
        LOG_ERROR("Pitch ", pitch, " is less than minimum required ", minPitch);
        return false;
    }
    
    return true;
}

bool CudaOpenGLInterop::ValidateCudaArraySize(void* cudaArray, int expectedWidth, int expectedHeight) {
    if (!cudaArray) {
        LOG_ERROR("Invalid CUDA array for size validation");
        return false;
    }
    
    cudaArray_t array = static_cast<cudaArray_t>(cudaArray);
    
    // Get array descriptor to check dimensions
    cudaChannelFormatDesc desc;
    cudaExtent extent;
    unsigned int flags;
    
    cudaError_t result = cudaArrayGetInfo(&desc, &extent, &flags, array);
    if (result != cudaSuccess) {
        LOG_ERROR("Failed to get CUDA array info: ", cudaGetErrorString(result));
        return false;
    }
    
    if (extent.width != static_cast<size_t>(expectedWidth) || 
        extent.height != static_cast<size_t>(expectedHeight)) {
        LOG_ERROR("CUDA array dimension mismatch - Expected: ", expectedWidth, "x", expectedHeight, 
                  ", Actual: ", extent.width, "x", extent.height);
        return false;
    }
    
    return true;
}

#endif // USE_OPENGL_RENDERER && HAVE_CUDA