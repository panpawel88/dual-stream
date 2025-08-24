# Video Decoding System

This directory implements the video decoding pipeline with support for hardware acceleration and multiple rendering backends.

## Architecture Overview

The decoding system consists of two main components:
1. **HardwareDecoder** - Detection and management of available hardware decoders
2. **VideoDecoder** - Unified decoder interface with hardware/software fallback

## Core Components

### HardwareDecoder
**File:** `HardwareDecoder.h/cpp`
**Purpose:** Hardware decoder detection and capability management

**Decoder Types:**
```cpp
enum class DecoderType {
    NONE,        // No decoder available
    NVDEC,       // NVIDIA hardware decoder
    SOFTWARE     // Software decoder fallback
};
```

**Key Features:**
- **Runtime Detection:** Automatically detects available hardware decoders
- **Capability Testing:** Validates codec support (H264/H265)
- **Best Decoder Selection:** Chooses optimal decoder for given codec
- **Graceful Fallback:** Falls back to software when hardware unavailable

**Detection Logic:**
```cpp
// Tests NVDEC availability by creating CUDA device context
bool TestNVDECAvailability() {
    // Try to create CUDA hardware device context
    // Check for h264_cuvid and hevc_cuvid decoders
    // Return availability status
}
```

### VideoDecoder
**File:** `VideoDecoder.h/cpp`  
**Purpose:** Unified video decoder with multi-backend support

**Supported Backends:**
- **D3D11VA:** DirectX 11 Video Acceleration (D3D11 renderer)
- **CUDA:** NVIDIA CUDA decoding (OpenGL renderer with CUDA interop)
- **Software:** CPU-based decoding with format conversion

## Frame Output Architecture

### DecodedFrame Structure
```cpp
struct DecodedFrame {
    // DirectX 11 texture (for D3D11 renderer)
    ComPtr<ID3D11Texture2D> texture;
    
    // Software frame data (for OpenGL renderer)
    uint8_t* data;
    int width, height, pitch;
    
    // CUDA hardware frame data (OpenGL + CUDA)
    void* cudaPtr;
    size_t cudaPitch;
    void* cudaResource;  // OpenGL interop resource
    bool isHardwareCuda;
    
    // Frame metadata
    double presentationTime;
    bool valid;
    bool isYUV;          // Requires YUV->RGB conversion
    bool keyframe;       // I-frame flag
    DXGI_FORMAT format;
};
```

### Multi-Backend Frame Processing

#### D3D11 Hardware Path
```cpp
bool ProcessHardwareFrame(DecodedFrame& outFrame) {
    // Extract D3D11 texture from AVFrame
    ID3D11Texture2D* hwTexture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    
    // Handle texture arrays (common with hardware decode)
    int arrayIndex = reinterpret_cast<intptr_t>(frame->data[1]);
    
    // Create single texture from array slice if needed
    // Set format flags (YUV vs RGB)
}
```

#### OpenGL + CUDA Path  
```cpp
bool ProcessCudaHardwareFrame(DecodedFrame& outFrame) {
    // Extract CUDA device pointer from AVFrame
    void* devicePtr = reinterpret_cast<void*>(frame->data[0]);
    size_t pitch = static_cast<size_t>(frame->linesize[0]);
    
    // Store CUDA pointers for OpenGL interop
    outFrame.cudaPtr = devicePtr;
    outFrame.isHardwareCuda = true;
}
```

#### Software Path
```cpp
bool ProcessSoftwareFrame(DecodedFrame& outFrame) {
    // Convert YUV to RGB using libswscale
    SwsContext* swsContext = sws_getContext(..., AV_PIX_FMT_BGRA, ...);
    sws_scale(swsContext, frame->data, frame->linesize, ...);
    
    // Store RGB data for renderer consumption
}
```

## Hardware Context Management

### D3D11VA Context Creation
```cpp
bool CreateHardwareDeviceContext() {
    // Create D3D11VA context using existing D3D11 device
    AVD3D11VADeviceContext* d3d11vaContext = ...;
    d3d11vaContext->device = m_d3dDevice.Get();
    d3d11vaContext->device_context = m_d3dContext.Get();
    
    // Initialize FFmpeg hardware context
    av_hwdevice_ctx_init(m_hwDeviceContext);
}
```

### CUDA Context Creation
```cpp
bool CreateCudaDeviceContext() {
    // Initialize CUDA Runtime API (for OpenGL interop compatibility)
    cudaSetDevice(0);
    
    // Get current context created by Runtime API
    CUcontext currentContext;
    cuCtxGetCurrent(&currentContext);
    
    // Use existing context for FFmpeg (shared ownership)
    AVCUDADeviceContext* cudaDeviceContext = ...;
    cudaDeviceContext->cuda_ctx = currentContext;
}
```

## Decoder Initialization Flow

### Hardware Decoder Path
1. **Capability Check:** `HardwareDecoder::GetBestDecoder(codecId)`
2. **Context Creation:** `CreateHardwareDeviceContext()` (D3D11VA or CUDA)
3. **FFmpeg Setup:** `SetupHardwareDecoding()` - Links hardware context to codec
4. **Codec Opening:** `avcodec_open2()` with hardware acceleration

### Software Decoder Fallback
1. **Codec Finding:** `avcodec_find_decoder(codecId)`
2. **Context Allocation:** `avcodec_alloc_context3()`
3. **Parameter Copying:** `avcodec_parameters_to_context()`
4. **Codec Opening:** `avcodec_open2()` without hardware context

## Format Support and Conversion

### Supported Input Formats
- **H.264** (AVC) - Hardware and software decoding
- **H.265** (HEVC) - Hardware and software decoding
- **Container:** MP4 containers with H264/H265 video streams

### Output Format Handling
- **Hardware Frames:** Native format preservation (NV12, P010, etc.)
- **Software Frames:** Automatic YUV->RGB conversion using libswscale
- **Format Detection:** Automatic YUV/RGB format flagging for shader processing

## Performance Characteristics

### Hardware Decoding Benefits
- **GPU Acceleration:** Dedicated video decode units (NVDEC)
- **Memory Efficiency:** Direct GPU memory allocation
- **Format Optimization:** Native YUV format handling
- **CPU Offload:** Reduced CPU usage for decode operations

### Fallback Scenarios
- **Hardware Unavailable:** Automatic software fallback
- **Driver Issues:** Graceful degradation to CPU decoding
- **Memory Pressure:** Fallback when GPU memory exhausted
- **Codec Unsupported:** Software path for unsupported hardware codecs

## Error Handling and Recovery

### Decoder Failure Handling
```cpp
bool Initialize(...) {
    // Try hardware decoding first
    if (decoderInfo.type == DecoderType::NVDEC && decoderInfo.available) {
        success = InitializeHardwareDecoder(codecParams);
        if (!success) {
            LOG_INFO("Hardware decoding failed, falling back to software");
            success = InitializeSoftwareDecoder(codecParams);
        }
    }
}
```

### Frame Processing Errors
- **Invalid Frames:** Graceful handling of decode failures
- **Format Mismatches:** Automatic format conversion
- **Memory Allocation:** Safe cleanup on allocation failures
- **Timeout Handling:** Prevents infinite decode loops (100 attempt limit)

## Integration Points

### With VideoManager
```cpp
// VideoManager initializes decoders with appropriate backend info
bool InitializeVideoStream(VideoStream& stream, const std::string& filePath, 
                          ID3D11Device* d3dDevice, bool cudaInteropAvailable) {
    DecoderInfo decoderInfo = HardwareDecoder::GetBestDecoder(codecId);
    stream.decoder.Initialize(codecParams, decoderInfo, d3dDevice, timebase, cudaInteropAvailable);
}
```

### With Renderers
- **D3D11 Renderer:** Receives `ID3D11Texture2D` directly from hardware decode
- **OpenGL Renderer:** Receives RGB data (software) or CUDA pointers (hardware)
- **Format Negotiation:** `isYUV` flag indicates required shader processing

## Memory Management

### RAII Pattern
```cpp
VideoDecoder::~VideoDecoder() {
    Cleanup();  // Automatic cleanup on destruction
}

void VideoDecoder::Reset() {
    // Clean up FFmpeg contexts
    if (m_codecContext) avcodec_free_context(&m_codecContext);
    if (m_hwDeviceContext) av_buffer_unref(&m_hwDeviceContext);
    if (m_frame) av_frame_free(&m_frame);
}
```

### Frame Data Management
```cpp
DecodedFrame::~DecodedFrame() {
    if (data) {
        delete[] data;
        data = nullptr;
    }
    if (cudaResource) {
        cuGraphicsUnregisterResource(static_cast<CUgraphicsResource>(cudaResource));
        cudaResource = nullptr;
    }
}
```

This decoding system provides a robust foundation for multi-backend video processing with automatic hardware acceleration and graceful software fallback.