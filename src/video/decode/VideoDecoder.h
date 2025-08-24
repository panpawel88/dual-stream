#pragma once

#include <memory>
#include <string>
#include "HardwareDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}

#include <d3d11.h>
#include <wrl/client.h>

// No CUDA types in header - use opaque handles

using Microsoft::WRL::ComPtr;

struct DecodedFrame {
    // DirectX 11 texture (for D3D11 renderer)
    ComPtr<ID3D11Texture2D> texture;
    
    // Software frame data (for OpenGL renderer)
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int pitch = 0;  // bytes per row
    
#if USE_OPENGL_RENDERER && HAVE_CUDA
    // CUDA hardware frame data (for OpenGL renderer with hardware decoding)
    // Use opaque handles to avoid CUDA type conflicts
    void* cudaPtr = nullptr;
    size_t cudaPitch = 0;
    void* cudaResource = nullptr;  // OpenGL interop resource
    bool isHardwareCuda = false;
#endif
    
    double presentationTime;
    bool valid;
    bool isYUV;  // True for hardware frames that need YUV->RGB conversion in shader
    bool keyframe;  // True if this frame is a keyframe (I-frame)
    DXGI_FORMAT format;
    
    DecodedFrame() : presentationTime(0.0), valid(false), isYUV(false), keyframe(false), format(DXGI_FORMAT_B8G8R8A8_UNORM) {}
    
    // Destructor to clean up software data and CUDA resources
    ~DecodedFrame();
    
    // Copy constructor
    DecodedFrame(const DecodedFrame& other) 
        : texture(other.texture), presentationTime(other.presentationTime), valid(other.valid)
        , isYUV(other.isYUV), keyframe(other.keyframe), format(other.format)
        , width(other.width), height(other.height), pitch(other.pitch) {
        if (other.data && width > 0 && height > 0 && pitch > 0) {
            size_t dataSize = pitch * height;
            data = new uint8_t[dataSize];
            memcpy(data, other.data, dataSize);
        } else {
            data = nullptr;
        }
#if USE_OPENGL_RENDERER && HAVE_CUDA
        // Note: CUDA resources are not copyable, only reference the same data
        cudaPtr = other.cudaPtr;
        cudaPitch = other.cudaPitch;
        cudaResource = 0; // Don't copy graphics resource, it's managed elsewhere
        isHardwareCuda = other.isHardwareCuda;
#endif
    }
    
    // Assignment operator
    DecodedFrame& operator=(const DecodedFrame& other);
};

class VideoDecoder {
public:
    VideoDecoder();
    ~VideoDecoder();
    
    bool Initialize(AVCodecParameters* codecParams, const DecoderInfo& decoderInfo, ID3D11Device* d3dDevice = nullptr, AVRational streamTimebase = {0, 1}, bool cudaInteropAvailable = false);
    void Cleanup();
    
    bool SendPacket(AVPacket* packet);
    bool ReceiveFrame(DecodedFrame& frame);
    void Flush();
    
    // Getters
    bool IsInitialized() const { return m_initialized; }
    bool IsHardwareAccelerated() const { return m_useHardwareDecoding; }
    const DecoderInfo& GetDecoderInfo() const { return m_decoderInfo; }
    
private:
    bool m_initialized;
    bool m_useHardwareDecoding;
    DecoderInfo m_decoderInfo;
    
    // FFmpeg components
    const AVCodec* m_codec;
    AVCodecContext* m_codecContext;
    AVBufferRef* m_hwDeviceContext;
    AVFrame* m_frame;
    AVFrame* m_hwFrame;
    AVRational m_streamTimebase;
    
    // DirectX 11 components
    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    
#if USE_OPENGL_RENDERER && HAVE_CUDA
    // CUDA components for OpenGL hardware decoding (opaque handles)
    void* m_cudaContext;
    bool m_cudaContextOwned;  // True if we created the context, false if shared
#endif
    
    // Initialization helpers
    bool InitializeHardwareDecoder(AVCodecParameters* codecParams);
    bool InitializeSoftwareDecoder(AVCodecParameters* codecParams);
    bool CreateHardwareDeviceContext();
    bool SetupHardwareDecoding();
    
    // Frame processing
    bool ProcessHardwareFrame(DecodedFrame& outFrame);
    bool ProcessSoftwareFrame(DecodedFrame& outFrame);
    bool CreateTextureFromFrame(AVFrame* frame, ComPtr<ID3D11Texture2D>& texture);
    bool CreateSoftwareFrameData(AVFrame* frame, DecodedFrame& outFrame);
    
    // Hardware-specific helpers
    bool IsHardwareFrame(AVFrame* frame) const;
    bool ExtractD3D11Texture(AVFrame* frame, ComPtr<ID3D11Texture2D>& texture);
    
#if USE_OPENGL_RENDERER && HAVE_CUDA
    // CUDA-specific helpers (using opaque handles)
    bool CreateCudaDeviceContext();
    bool SetupCudaDecoding();
    bool ExtractCudaDevicePtr(AVFrame* frame, void*& devicePtr, size_t& pitch);
    bool ProcessCudaHardwareFrame(DecodedFrame& outFrame);
#endif
    
    void Reset();
};