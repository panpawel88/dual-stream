#include "VideoDecoder.h"
#include "Logger.h"
#include <iostream>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libswscale/swscale.h>
}

VideoDecoder::VideoDecoder()
    : m_initialized(false)
    , m_useHardwareDecoding(false)
    , m_codec(nullptr)
    , m_codecContext(nullptr)
    , m_hwDeviceContext(nullptr)
    , m_frame(nullptr)
    , m_hwFrame(nullptr) {
}

VideoDecoder::~VideoDecoder() {
    Cleanup();
}

bool VideoDecoder::Initialize(AVCodecParameters* codecParams, const DecoderInfo& decoderInfo, ID3D11Device* d3dDevice, AVRational streamTimebase) {
    if (m_initialized) {
        Cleanup();
    }
    
    if (!codecParams || !d3dDevice) {
        std::cerr << "Invalid parameters for VideoDecoder initialization\n";
        return false;
    }
    
    m_d3dDevice = d3dDevice;
    m_d3dDevice->GetImmediateContext(&m_d3dContext);
    m_decoderInfo = decoderInfo;
    m_streamTimebase = streamTimebase;
    
    LOG_INFO("Initializing video decoder with ", decoderInfo.name);
    
    // Allocate frames
    m_frame = av_frame_alloc();
    m_hwFrame = av_frame_alloc();
    if (!m_frame || !m_hwFrame) {
        std::cerr << "Failed to allocate AVFrame structures\n";
        Cleanup();
        return false;
    }
    
    // Initialize decoder based on type
    bool success = false;
    if (decoderInfo.type == DecoderType::NVDEC && decoderInfo.available) {
        success = InitializeHardwareDecoder(codecParams);
        if (success) {
            m_useHardwareDecoding = true;
            LOG_INFO("Hardware decoding enabled");
        } else {
            LOG_INFO("Hardware decoding failed, falling back to software");
            success = InitializeSoftwareDecoder(codecParams);
            m_useHardwareDecoding = false;
        }
    } else {
        success = InitializeSoftwareDecoder(codecParams);
        m_useHardwareDecoding = false;
        LOG_INFO("Software decoding enabled");
    }
    
    if (!success) {
        std::cerr << "Failed to initialize video decoder\n";
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
#if HAVE_CUDA
    // Find appropriate NVDEC decoder
    const char* decoderName = nullptr;
    if (codecParams->codec_id == AV_CODEC_ID_H264) {
        decoderName = "h264_cuvid";
    } else if (codecParams->codec_id == AV_CODEC_ID_HEVC) {
        decoderName = "hevc_cuvid";
    } else {
        std::cerr << "Unsupported codec for hardware decoding\n";
        return false;
    }
    
    m_codec = avcodec_find_decoder_by_name(decoderName);
    if (!m_codec) {
        std::cerr << "Hardware decoder not found: " << decoderName << "\n";
        return false;
    }
    
    m_codecContext = avcodec_alloc_context3(m_codec);
    if (!m_codecContext) {
        std::cerr << "Failed to allocate codec context\n";
        return false;
    }
    
    // Copy codec parameters
    int ret = avcodec_parameters_to_context(m_codecContext, codecParams);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Failed to copy codec parameters: " << errorBuf << "\n";
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
        std::cerr << "Failed to open hardware codec: " << errorBuf << "\n";
        return false;
    }
    
    return true;
#else
    std::cerr << "Hardware decoding not supported (CUDA not available)\n";
    return false;
#endif
}

bool VideoDecoder::InitializeSoftwareDecoder(AVCodecParameters* codecParams) {
    m_codec = avcodec_find_decoder(codecParams->codec_id);
    if (!m_codec) {
        std::cerr << "Software decoder not found for codec\n";
        return false;
    }
    
    m_codecContext = avcodec_alloc_context3(m_codec);
    if (!m_codecContext) {
        std::cerr << "Failed to allocate codec context\n";
        return false;
    }
    
    // Copy codec parameters
    int ret = avcodec_parameters_to_context(m_codecContext, codecParams);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Failed to copy codec parameters: " << errorBuf << "\n";
        return false;
    }
    
    // Open codec
    ret = avcodec_open2(m_codecContext, m_codec, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Failed to open software codec: " << errorBuf << "\n";
        return false;
    }
    
    return true;
}

bool VideoDecoder::CreateHardwareDeviceContext() {
#if HAVE_CUDA
    int ret = av_hwdevice_ctx_create(&m_hwDeviceContext, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Failed to create CUDA device context: " << errorBuf << "\n";
        return false;
    }
    
    return true;
#else
    return false;
#endif
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
        std::cerr << "Expected hardware frame but got software frame\n";
        return false;
    }
    
    // Transfer frame from GPU to CPU memory
    if (!TransferHardwareFrame()) {
        return false;
    }
    
    // Create D3D11 texture from the transferred frame
    return CreateTextureFromFrame(m_hwFrame, outFrame.texture);
}

bool VideoDecoder::ProcessSoftwareFrame(DecodedFrame& outFrame) {
    return CreateTextureFromFrame(m_frame, outFrame.texture);
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

bool VideoDecoder::IsHardwareFrame(AVFrame* frame) const {
    if (!frame) {
        return false;
    }
    
    // Check if the frame format is a hardware pixel format
    return frame->format == AV_PIX_FMT_CUDA || 
           frame->format == AV_PIX_FMT_D3D11 ||
           frame->format == AV_PIX_FMT_DXVA2_VLD ||
           frame->hw_frames_ctx != nullptr;
}

bool VideoDecoder::TransferHardwareFrame() {
    if (!m_frame || !m_hwFrame) {
        return false;
    }
    
    // Transfer hardware frame to system memory
    int ret = av_hwframe_transfer_data(m_hwFrame, m_frame, 0);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cerr << "Failed to transfer hardware frame: " << errorBuf << "\n";
        return false;
    }
    
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
}