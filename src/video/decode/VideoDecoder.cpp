#include "VideoDecoder.h"
#include "core/Logger.h"
#include <iostream>
#include <iomanip>


extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libswscale/swscale.h>
#ifdef HAVE_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
}

#ifdef HAVE_CUDA
// Include CUDA headers in implementation
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

VideoDecoder::VideoDecoder()
    : m_initialized(false)
    , m_useHardwareDecoding(false)
    , m_codec(nullptr)
    , m_codecContext(nullptr)
    , m_hwDeviceContext(nullptr)
    , m_frame(nullptr)
    , m_hwFrame(nullptr)
#ifdef HAVE_CUDA
    , m_cudaContext(nullptr)
    , m_cudaContextOwned(false)
#endif
{
}

VideoDecoder::~VideoDecoder() {
    Cleanup();
}

bool VideoDecoder::Initialize(AVCodecParameters* codecParams, const DecoderInfo& decoderInfo, ID3D11Device* d3dDevice, AVRational streamTimebase, bool cudaInteropAvailable) {
    if (m_initialized) {
        Cleanup();
    }
    
    if (!codecParams) {
        LOG_ERROR("Invalid codec parameters for VideoDecoder initialization");
        return false;
    }
    
    // D3D device is optional now (required only for hardware decoding)
    if (d3dDevice) {
        m_d3dDevice = d3dDevice;
        m_d3dDevice->GetImmediateContext(&m_d3dContext);
    }
    m_decoderInfo = decoderInfo;
    m_streamTimebase = streamTimebase;
    
    LOG_INFO("Initializing video decoder with ", decoderInfo.name);
    
    // Allocate frames
    m_frame = av_frame_alloc();
    m_hwFrame = av_frame_alloc();
    if (!m_frame || !m_hwFrame) {
        LOG_ERROR("Failed to allocate AVFrame structures");
        Cleanup();
        return false;
    }
    
    // Initialize decoder based on renderer type and hardware availability
    bool success = false;
    if (decoderInfo.type == DecoderType::NVDEC && decoderInfo.available) {
        // For OpenGL renderer, try CUDA hardware decoding only if CUDA interop is available
        if (!d3dDevice && cudaInteropAvailable) {
#ifdef HAVE_CUDA
            LOG_INFO("Attempting CUDA hardware decoding for OpenGL renderer (CUDA interop available)");
            success = InitializeHardwareDecoder(codecParams);
            if (success) {
                m_useHardwareDecoding = true;
                LOG_INFO("CUDA hardware decoding enabled for OpenGL");
            } else {
                LOG_INFO("CUDA hardware decoding failed, falling back to software");
                success = InitializeSoftwareDecoder(codecParams);
                m_useHardwareDecoding = false;
            }
#else
            LOG_INFO("CUDA not available, using software decoding");
            success = InitializeSoftwareDecoder(codecParams);
            m_useHardwareDecoding = false;
#endif
        } else if (!d3dDevice && !cudaInteropAvailable) {
            LOG_INFO("CUDA interop not available for OpenGL renderer, using software decoding");
            success = InitializeSoftwareDecoder(codecParams);
            m_useHardwareDecoding = false;
        } else
        // For D3D11 renderer or when CUDA is not available
        if (d3dDevice) {
            LOG_INFO("Attempting D3D11VA hardware decoding for D3D11 renderer");
            success = InitializeHardwareDecoder(codecParams);
            if (success) {
                m_useHardwareDecoding = true;
                LOG_INFO("D3D11VA hardware decoding enabled");
            } else {
                LOG_INFO("D3D11VA hardware decoding failed, falling back to software");
                success = InitializeSoftwareDecoder(codecParams);
                m_useHardwareDecoding = false;
            }
        }
    }
    
    // Fall back to software decoding if hardware was not attempted or failed
    if (!success) {
        success = InitializeSoftwareDecoder(codecParams);
        m_useHardwareDecoding = false;
        LOG_INFO("Software decoding enabled");
    }
    
    if (!success) {
        LOG_ERROR("Failed to initialize video decoder");
        Cleanup();
        return false;
    }
    
    m_initialized = true;
    return true;
}

void VideoDecoder::Cleanup() {
    Reset();
}

bool VideoDecoder::SendPacket(AVPacket* packet) {
    if (!m_initialized || !m_codecContext) {
        LOG_DEBUG("SendPacket failed - decoder not initialized or no codec context");
        return false;
    }
    
    LOG_DEBUG("Sending packet to decoder - Size: ", (packet ? packet->size : 0),
              ", PTS: ", (packet && packet->pts != AV_NOPTS_VALUE ? packet->pts : -1),
              ", DTS: ", (packet && packet->dts != AV_NOPTS_VALUE ? packet->dts : -1));
    
    int ret = avcodec_send_packet(m_codecContext, packet);
    if (ret < 0) {
        if (ret == AVERROR_EOF) {
            LOG_DEBUG("Decoder reached end of stream");
            return true; // End of stream
        }
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_DEBUG("Error sending packet to decoder: ", errorBuf, " (ret=", ret, ")");
        return false;
    }
    
    LOG_DEBUG("Packet sent to decoder successfully");
    return true;
}

bool VideoDecoder::ReceiveFrame(DecodedFrame& frame) {
    if (!m_initialized || !m_codecContext) {
        LOG_DEBUG("ReceiveFrame failed - decoder not initialized or no codec context");
        return false;
    }
    
    frame.valid = false;
    
    int ret = avcodec_receive_frame(m_codecContext, m_frame);
    if (ret < 0) {
        if (ret == AVERROR(EAGAIN)) {
            LOG_DEBUG("No frame available yet (EAGAIN)");
            return true; // No frame available yet
        } else if (ret == AVERROR_EOF) {
            LOG_DEBUG("End of stream reached (EOF)");
            return true; // End of stream
        }
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_DEBUG("Error receiving frame from decoder: ", errorBuf, " (ret=", ret, ")");
        return false;
    }
    
    LOG_DEBUG("Received frame from decoder - Size: ", m_frame->width, "x", m_frame->height,
              ", Format: ", m_frame->format, ", PTS: ", m_frame->pts,
              ", Codec Timebase: ", m_codecContext->time_base.num, "/", m_codecContext->time_base.den,
              ", Stream Timebase: ", m_streamTimebase.num, "/", m_streamTimebase.den);
    
    // Process frame based on decoder type
    bool success = false;
    if (m_useHardwareDecoding) {
        LOG_DEBUG("Processing hardware frame");
        success = ProcessHardwareFrame(frame);
    } else {
        LOG_DEBUG("Processing software frame");
        success = ProcessSoftwareFrame(frame);
    }
    
    if (success) {
        // Set presentation time using stream timebase
        if (m_frame->pts != AV_NOPTS_VALUE) {
            if (m_streamTimebase.den != 0) {
                frame.presentationTime = static_cast<double>(m_frame->pts) * av_q2d(m_streamTimebase);
                LOG_DEBUG("Frame presentation time (using stream timebase): ", frame.presentationTime, " seconds");
            } else {
                // Fallback to codec timebase if stream timebase is invalid
                frame.presentationTime = static_cast<double>(m_frame->pts) * av_q2d(m_codecContext->time_base);
                LOG_DEBUG("Frame presentation time (using codec timebase): ", frame.presentationTime, " seconds");
            }
        } else {
            LOG_DEBUG("Frame has no PTS (AV_NOPTS_VALUE)");
        }
        
        // Set keyframe flag based on FFmpeg's frame information
        frame.keyframe = (m_frame->flags & AV_FRAME_FLAG_KEY) || (m_frame->pict_type == AV_PICTURE_TYPE_I);
        if (frame.keyframe) {
            LOG_DEBUG("Frame is a keyframe (I-frame) at time: ", frame.presentationTime);
        }
        
        frame.valid = true;
        LOG_DEBUG("Frame processed successfully");
    } else {
        LOG_DEBUG("Failed to process frame");
    }
    
    return success;
}

void VideoDecoder::Flush() {
    if (m_codecContext) {
        avcodec_flush_buffers(m_codecContext);
    }
}

bool VideoDecoder::InitializeHardwareDecoder(AVCodecParameters* codecParams) {
    // Find appropriate hardware decoder
    m_codec = avcodec_find_decoder(codecParams->codec_id);
    if (!m_codec) {
        LOG_ERROR("Decoder not found for codec");
        return false;
    }
    
    m_codecContext = avcodec_alloc_context3(m_codec);
    if (!m_codecContext) {
        LOG_ERROR("Failed to allocate codec context");
        return false;
    }
    
    // Copy codec parameters
    int ret = avcodec_parameters_to_context(m_codecContext, codecParams);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Failed to copy codec parameters: ", errorBuf);
        return false;
    }
    
    // Create hardware device context
    if (!CreateHardwareDeviceContext()) {
        return false;
    }
    
    // Setup hardware decoding
    if (!SetupHardwareDecoding()) {
        return false;
    }
    
    // Open codec
    ret = avcodec_open2(m_codecContext, m_codec, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Failed to open hardware codec: ", errorBuf);
        return false;
    }
    
    return true;
}

bool VideoDecoder::InitializeSoftwareDecoder(AVCodecParameters* codecParams) {
    m_codec = avcodec_find_decoder(codecParams->codec_id);
    if (!m_codec) {
        LOG_ERROR("Software decoder not found for codec");
        return false;
    }
    
    m_codecContext = avcodec_alloc_context3(m_codec);
    if (!m_codecContext) {
        LOG_ERROR("Failed to allocate codec context");
        return false;
    }
    
    // Copy codec parameters
    int ret = avcodec_parameters_to_context(m_codecContext, codecParams);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Failed to copy codec parameters: ", errorBuf);
        return false;
    }
    
    // Open codec
    ret = avcodec_open2(m_codecContext, m_codec, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Failed to open software codec: ", errorBuf);
        return false;
    }
    
    return true;
}

bool VideoDecoder::CreateHardwareDeviceContext() {
    // For OpenGL renderer, create CUDA device context
    if (!m_d3dDevice) {
#ifdef HAVE_CUDA
        return CreateCudaDeviceContext();
#else
        LOG_ERROR("CUDA not available for OpenGL renderer");
        return false;
#endif
    }
    
    // For D3D11 renderer, create D3D11VA device context using the existing D3D11 device
    AVHWDeviceContext* deviceContext;
    AVD3D11VADeviceContext* d3d11vaContext;
    
    m_hwDeviceContext = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
    if (!m_hwDeviceContext) {
        LOG_ERROR("Failed to allocate D3D11VA device context");
        return false;
    }
    
    deviceContext = reinterpret_cast<AVHWDeviceContext*>(m_hwDeviceContext->data);
    d3d11vaContext = reinterpret_cast<AVD3D11VADeviceContext*>(deviceContext->hwctx);
    
    // Use our existing D3D11 device
    d3d11vaContext->device = m_d3dDevice.Get();
    d3d11vaContext->device->AddRef(); // AddRef since FFmpeg will release it
    d3d11vaContext->device_context = m_d3dContext.Get();
    d3d11vaContext->device_context->AddRef();
    
    int ret = av_hwdevice_ctx_init(m_hwDeviceContext);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Failed to initialize D3D11VA device context: ", errorBuf);
        return false;
    }
    
    return true;
}

bool VideoDecoder::SetupHardwareDecoding() {
    if (!m_codecContext || !m_hwDeviceContext) {
        return false;
    }
    
    m_codecContext->hw_device_ctx = av_buffer_ref(m_hwDeviceContext);
    return true;
}

bool VideoDecoder::ProcessHardwareFrame(DecodedFrame& outFrame) {
    if (!IsHardwareFrame(m_frame)) {
        LOG_ERROR("Expected hardware frame but got software frame");
        return false;
    }
    
#ifdef HAVE_CUDA
    // For OpenGL renderer with CUDA, process CUDA hardware frames
    if (m_frame->format == AV_PIX_FMT_CUDA) {
        return ProcessCudaHardwareFrame(outFrame);
    }
#endif
    
    // For D3D11 renderer, extract D3D11 texture from hardware frame
    if (!ExtractD3D11Texture(m_frame, outFrame.texture)) {
        return false;
    }
    
    // Get actual format from the extracted texture
    if (outFrame.texture) {
        D3D11_TEXTURE2D_DESC desc;
        outFrame.texture->GetDesc(&desc);
        outFrame.format = desc.Format;
        LOG_DEBUG("Hardware frame format: ", desc.Format);
        
        // Set YUV flag based on actual format
        if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.Format == DXGI_FORMAT_B8G8R8X8_UNORM) {
            // RGB format
            outFrame.isYUV = false;
            LOG_DEBUG("Hardware texture is RGB format: ", desc.Format);
        } else {
            // YUV format (NV12, P010, 420_OPAQUE, etc.)
            outFrame.isYUV = true;
            LOG_DEBUG("Hardware texture is YUV format: ", desc.Format, ", enabling YUV processing");
        }
    } else {
        // Fallback
        outFrame.isYUV = false;
        outFrame.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    }
    
    return true;
}

bool VideoDecoder::ProcessSoftwareFrame(DecodedFrame& outFrame) {
    // Check renderer type and handle accordingly
    // For OpenGL renderer, create software frame data
    // For D3D11 renderer, create texture
    bool success = false;
    
    // Try OpenGL renderer first (software frame data)
    success = CreateSoftwareFrameData(m_frame, outFrame);
    if (!success) {
        // Fallback to D3D11 renderer (create texture)
        success = CreateTextureFromFrame(m_frame, outFrame.texture);
    }
    
    if (success) {
        outFrame.isYUV = false; // Software frame is converted to RGB
        outFrame.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    }
    return success;
}

bool VideoDecoder::CreateTextureFromFrame(AVFrame* frame, ComPtr<ID3D11Texture2D>& texture) {
    if (!frame || !m_d3dDevice) {
        LOG_DEBUG("CreateTextureFromFrame failed - no frame or D3D device");
        return false;
    }
    
    LOG_DEBUG("Creating texture from frame - Size: ", frame->width, "x", frame->height, 
              ", Format: ", frame->format);
    
    // Convert YUV frame to RGB using libswscale
    AVFrame* rgbFrame = av_frame_alloc();
    if (!rgbFrame) {
        LOG_DEBUG("Failed to allocate RGB frame");
        return false;
    }
    
    // Set up RGB frame properties
    rgbFrame->format = AV_PIX_FMT_BGRA;
    rgbFrame->width = frame->width;
    rgbFrame->height = frame->height;
    
    int ret = av_frame_get_buffer(rgbFrame, 32);
    if (ret < 0) {
        LOG_DEBUG("Failed to allocate RGB frame buffer");
        av_frame_free(&rgbFrame);
        return false;
    }
    
    // Create conversion context
    SwsContext* swsContext = sws_getContext(
        frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
        frame->width, frame->height, AV_PIX_FMT_BGRA,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    if (!swsContext) {
        LOG_DEBUG("Failed to create SWS context for format conversion");
        av_frame_free(&rgbFrame);
        return false;
    }
    
    // Convert YUV to RGB
    ret = sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height,
                    rgbFrame->data, rgbFrame->linesize);
    
    sws_freeContext(swsContext);
    
    if (ret < 0) {
        LOG_DEBUG("Failed to convert YUV to RGB");
        av_frame_free(&rgbFrame);
        return false;
    }
    
    LOG_DEBUG("Successfully converted YUV to RGB");
    
    // Create texture description
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = frame->width;
    textureDesc.Height = frame->height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    // Create initial data for texture
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = rgbFrame->data[0];
    initData.SysMemPitch = rgbFrame->linesize[0];
    initData.SysMemSlicePitch = 0;
    
    // Create texture with actual frame data
    HRESULT hr = m_d3dDevice->CreateTexture2D(&textureDesc, &initData, &texture);
    if (FAILED(hr)) {
        LOG_DEBUG("Failed to create D3D11 texture with data. HRESULT: 0x", std::hex, hr);
        av_frame_free(&rgbFrame);
        return false;
    }
    
    LOG_DEBUG("D3D11 texture created successfully with actual frame data!");
    
    av_frame_free(&rgbFrame);
    return true;
}

bool VideoDecoder::CreateSoftwareFrameData(AVFrame* frame, DecodedFrame& outFrame) {
    if (!frame) {
        LOG_DEBUG("CreateSoftwareFrameData failed - no frame");
        return false;
    }
    
    LOG_DEBUG("Creating software frame data - Size: ", frame->width, "x", frame->height, 
              ", Format: ", frame->format);
    
    // Convert YUV frame to RGB using libswscale
    AVFrame* rgbFrame = av_frame_alloc();
    if (!rgbFrame) {
        LOG_DEBUG("Failed to allocate RGB frame");
        return false;
    }
    
    // Set up RGB frame properties
    rgbFrame->format = AV_PIX_FMT_BGRA;
    rgbFrame->width = frame->width;
    rgbFrame->height = frame->height;
    
    int ret = av_frame_get_buffer(rgbFrame, 32);
    if (ret < 0) {
        LOG_DEBUG("Failed to allocate RGB frame buffer");
        av_frame_free(&rgbFrame);
        return false;
    }
    
    // Create conversion context
    SwsContext* swsContext = sws_getContext(
        frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
        frame->width, frame->height, AV_PIX_FMT_BGRA,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    if (!swsContext) {
        LOG_DEBUG("Failed to create SWS context for format conversion");
        av_frame_free(&rgbFrame);
        return false;
    }
    
    // Convert YUV to RGB
    ret = sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height,
                    rgbFrame->data, rgbFrame->linesize);
    
    sws_freeContext(swsContext);
    
    if (ret < 0) {
        LOG_DEBUG("Failed to convert YUV to RGB");
        av_frame_free(&rgbFrame);
        return false;
    }
    
    LOG_DEBUG("Successfully converted YUV to RGB");
    
    // Clean up existing data if any
    if (outFrame.data) {
        delete[] outFrame.data;
        outFrame.data = nullptr;
    }
    
    // Copy frame data
    outFrame.width = frame->width;
    outFrame.height = frame->height;
    outFrame.pitch = rgbFrame->linesize[0];
    
    size_t dataSize = outFrame.pitch * outFrame.height;
    outFrame.data = new uint8_t[dataSize];
    memcpy(outFrame.data, rgbFrame->data[0], dataSize);
    
    LOG_DEBUG("Software frame data created successfully - Size: ", dataSize, " bytes");
    
    av_frame_free(&rgbFrame);
    return true;
}

bool VideoDecoder::IsHardwareFrame(AVFrame* frame) const {
    if (!frame) {
        return false;
    }
    
    // Check if the frame format is a hardware pixel format
    return frame->format == AV_PIX_FMT_D3D11 ||
           frame->format == AV_PIX_FMT_DXVA2_VLD ||
#ifdef HAVE_CUDA
           frame->format == AV_PIX_FMT_CUDA ||
#endif
           frame->hw_frames_ctx != nullptr;
}

bool VideoDecoder::ExtractD3D11Texture(AVFrame* frame, ComPtr<ID3D11Texture2D>& texture) {
    if (!frame || frame->format != AV_PIX_FMT_D3D11) {
        LOG_DEBUG("Frame is not D3D11 format or is null");
        return false;
    }
    
    // Extract D3D11 texture directly from the hardware frame
    // For D3D11 frames, data[0] contains the ID3D11Texture2D pointer
    // and data[1] contains the texture array index
    ID3D11Texture2D* hwTexture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    if (!hwTexture) {
        LOG_DEBUG("No D3D11 texture found in hardware frame");
        return false;
    }
    
    // Get texture description to understand format and properties
    D3D11_TEXTURE2D_DESC desc;
    hwTexture->GetDesc(&desc);
    
    LOG_DEBUG("Hardware texture extracted - Size: ", desc.Width, "x", desc.Height, 
              ", Format: ", desc.Format, ", ArraySize: ", desc.ArraySize);
    
    // Update outFrame format information based on actual hardware texture format
    switch (desc.Format) {
        case DXGI_FORMAT_NV12:
            LOG_DEBUG("Hardware texture is NV12 format (87)");
            break;
        case DXGI_FORMAT_P010:
            LOG_DEBUG("Hardware texture is P010 format (104)");
            break;
        case DXGI_FORMAT_420_OPAQUE:
            LOG_DEBUG("Hardware texture is 420_OPAQUE format (189)");
            break;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            LOG_DEBUG("Hardware texture is B8G8R8A8_UNORM format (87)");
            break;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            LOG_DEBUG("Hardware texture is R8G8B8A8_UNORM format (28)");
            break;
        default:
            LOG_DEBUG("Hardware texture format: ", desc.Format, " (unknown)");
            break;
    }
    
    // If this is a texture array (common with hardware decode), we need to create a view of the specific slice
    int arrayIndex = reinterpret_cast<intptr_t>(frame->data[1]);
    if (desc.ArraySize > 1) {
        // Create a new texture as a copy of the specific array slice
        D3D11_TEXTURE2D_DESC newDesc = desc;
        newDesc.ArraySize = 1;
        newDesc.Usage = D3D11_USAGE_DEFAULT;
        newDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        newDesc.CPUAccessFlags = 0;
        newDesc.MiscFlags = 0;
        
        HRESULT hr = m_d3dDevice->CreateTexture2D(&newDesc, nullptr, &texture);
        if (FAILED(hr)) {
            LOG_DEBUG("Failed to create texture copy. HRESULT: 0x", std::hex, hr);
            return false;
        }
        
        // Copy the specific array slice to our new texture
        m_d3dContext->CopySubresourceRegion(
            texture.Get(), 0, 0, 0, 0,
            hwTexture, arrayIndex, nullptr
        );
    } else {
        // Single texture, use directly
        texture = hwTexture;
    }
    
    LOG_DEBUG("D3D11 texture extracted successfully from hardware frame");
    return true;
}

void VideoDecoder::Reset() {
    m_initialized = false;
    m_useHardwareDecoding = false;
    
    if (m_codecContext) {
        avcodec_free_context(&m_codecContext);
    }
    
    if (m_hwDeviceContext) {
        av_buffer_unref(&m_hwDeviceContext);
    }
    
    if (m_frame) {
        av_frame_free(&m_frame);
    }
    
    if (m_hwFrame) {
        av_frame_free(&m_hwFrame);
    }
    
    m_codec = nullptr;
    m_d3dDevice.Reset();
    m_d3dContext.Reset();
    
#ifdef HAVE_CUDA
    // No manual CUDA context cleanup needed - Runtime API handles it
    m_cudaContext = nullptr;
    m_cudaContextOwned = false;
#endif
}

#ifdef HAVE_CUDA

bool VideoDecoder::CreateCudaDeviceContext() {
    // Initialize CUDA Runtime API first (for OpenGL interop compatibility)
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        LOG_ERROR("Failed to set CUDA device: ", cudaGetErrorString(cudaResult));
        return false;
    }
    
    // Initialize CUDA Driver API
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        LOG_ERROR("Failed to initialize CUDA driver API");
        return false;
    }
    
    // Get the current CUDA context created by Runtime API
    CUcontext currentContext;
    result = cuCtxGetCurrent(&currentContext);
    if (result != CUDA_SUCCESS || !currentContext) {
        LOG_ERROR("Failed to get current CUDA context from Runtime API");
        return false;
    }
    
    // Use the existing Runtime context for FFmpeg (don't create a new one)
    m_cudaContext = currentContext;
    m_cudaContextOwned = false; // We don't own it, Runtime API does
    
    // Create CUDA hardware device context for FFmpeg
    m_hwDeviceContext = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!m_hwDeviceContext) {
        LOG_ERROR("Failed to allocate CUDA device context");
        return false;
    }
    
    AVHWDeviceContext* deviceContext = reinterpret_cast<AVHWDeviceContext*>(m_hwDeviceContext->data);
    AVCUDADeviceContext* cudaDeviceContext = reinterpret_cast<AVCUDADeviceContext*>(deviceContext->hwctx);
    
    // Use the current Runtime API context for FFmpeg
    cudaDeviceContext->cuda_ctx = currentContext;
    
    int ret = av_hwdevice_ctx_init(m_hwDeviceContext);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOG_ERROR("Failed to initialize CUDA device context: ", errorBuf);
        return false;
    }
    
    LOG_INFO("CUDA device context created successfully");
    return true;
}

bool VideoDecoder::SetupCudaDecoding() {
    if (!m_codecContext || !m_hwDeviceContext) {
        return false;
    }
    
    m_codecContext->hw_device_ctx = av_buffer_ref(m_hwDeviceContext);
    return true;
}

bool VideoDecoder::ExtractCudaDevicePtr(AVFrame* frame, void*& devicePtr, size_t& pitch) {
    if (!frame || frame->format != AV_PIX_FMT_CUDA) {
        LOG_DEBUG("Frame is not CUDA format or is null");
        return false;
    }
    
    // For CUDA frames, data[0] contains the CUdeviceptr
    // and linesize[0] contains the pitch
    devicePtr = reinterpret_cast<void*>(frame->data[0]);
    pitch = static_cast<size_t>(frame->linesize[0]);
    
    LOG_DEBUG("CUDA device pointer extracted: 0x", std::hex, devicePtr, std::dec, 
              ", pitch: ", pitch, ", frame size: ", frame->width, "x", frame->height);
    
    return devicePtr != 0;
}

bool VideoDecoder::ProcessCudaHardwareFrame(DecodedFrame& outFrame) {
    if (!IsHardwareFrame(m_frame) || m_frame->format != AV_PIX_FMT_CUDA) {
        LOG_ERROR("Expected CUDA hardware frame but got different format");
        return false;
    }
    
    // Extract CUDA device pointer
    void* devicePtr;
    size_t pitch;
    if (!ExtractCudaDevicePtr(m_frame, devicePtr, pitch)) {
        return false;
    }
    
    // Set CUDA frame data
    outFrame.cudaPtr = devicePtr;
    outFrame.cudaPitch = pitch;
    outFrame.width = m_frame->width;
    outFrame.height = m_frame->height;
    outFrame.isHardwareCuda = true;
    
    // Determine if this is YUV format (most hardware decoded frames are)
    outFrame.isYUV = true; // CUDA frames are typically in YUV format (NV12, etc.)
    
    LOG_DEBUG("CUDA hardware frame processed successfully - Size: ", outFrame.width, "x", outFrame.height,
              ", CUDA ptr: 0x", std::hex, outFrame.cudaPtr, std::dec, ", pitch: ", outFrame.cudaPitch);
    
    return true;
}

#endif // HAVE_CUDA

// DecodedFrame implementation
DecodedFrame::~DecodedFrame() {
    if (data) {
        delete[] data;
        data = nullptr;
    }
#ifdef HAVE_CUDA
    if (cudaResource) {
        cuGraphicsUnregisterResource(static_cast<CUgraphicsResource>(cudaResource));
        cudaResource = nullptr;
    }
#endif
}

DecodedFrame& DecodedFrame::operator=(const DecodedFrame& other) {
    if (this != &other) {
        // Clean up existing data
        if (data) {
            delete[] data;
            data = nullptr;
        }
#ifdef HAVE_CUDA
        if (cudaResource) {
            cuGraphicsUnregisterResource(static_cast<CUgraphicsResource>(cudaResource));
            cudaResource = nullptr;
        }
#endif
        
        // Copy members
        texture = other.texture;
        presentationTime = other.presentationTime;
        valid = other.valid;
        isYUV = other.isYUV;
        keyframe = other.keyframe;
        format = other.format;
        width = other.width;
        height = other.height;
        pitch = other.pitch;
        
        // Copy data if present
        if (other.data && width > 0 && height > 0 && pitch > 0) {
            size_t dataSize = pitch * height;
            data = new uint8_t[dataSize];
            memcpy(data, other.data, dataSize);
        }
        
#ifdef HAVE_CUDA
        // Copy CUDA fields (note: graphics resource is not copyable)
        cudaPtr = other.cudaPtr;
        cudaPitch = other.cudaPitch;
        cudaResource = nullptr; // Don't copy graphics resource
        isHardwareCuda = other.isHardwareCuda;
#endif
    }
    return *this;
}