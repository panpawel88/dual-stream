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

using Microsoft::WRL::ComPtr;

struct DecodedFrame {
    ComPtr<ID3D11Texture2D> texture;
    double presentationTime;
    bool valid;
    
    DecodedFrame() : presentationTime(0.0), valid(false) {}
};

class VideoDecoder {
public:
    VideoDecoder();
    ~VideoDecoder();
    
    bool Initialize(AVCodecParameters* codecParams, const DecoderInfo& decoderInfo, ID3D11Device* d3dDevice);
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
    
    // DirectX 11 components
    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    
    // Initialization helpers
    bool InitializeHardwareDecoder(AVCodecParameters* codecParams);
    bool InitializeSoftwareDecoder(AVCodecParameters* codecParams);
    bool CreateHardwareDeviceContext();
    bool SetupHardwareDecoding();
    
    // Frame processing
    bool ProcessHardwareFrame(DecodedFrame& outFrame);
    bool ProcessSoftwareFrame(DecodedFrame& outFrame);
    bool CreateTextureFromFrame(AVFrame* frame, ComPtr<ID3D11Texture2D>& texture);
    
    // Hardware-specific helpers
    bool IsHardwareFrame(AVFrame* frame) const;
    bool TransferHardwareFrame();
    
    void Reset();
};