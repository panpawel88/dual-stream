#include "HardwareDecoder.h"
#include <iostream>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavcodec/avcodec.h>
}

bool HardwareDecoder::s_initialized = false;
std::vector<DecoderInfo> HardwareDecoder::s_availableDecoders;

bool HardwareDecoder::Initialize() {
    if (s_initialized) {
        return true;
    }
    
    std::cout << "Initializing hardware decoder detection...\n";
    
    DetectHardwareDecoders();
    
    std::cout << "Available decoders:\n";
    for (const auto& decoder : s_availableDecoders) {
        std::cout << "  - " << decoder.name << " (" << (decoder.available ? "Available" : "Unavailable") << ")\n";
    }
    
    s_initialized = true;
    return true;
}

void HardwareDecoder::Cleanup() {
    if (s_initialized) {
        s_availableDecoders.clear();
        s_initialized = false;
    }
}

std::vector<DecoderInfo> HardwareDecoder::GetAvailableDecoders() {
    return s_availableDecoders;
}

DecoderInfo HardwareDecoder::GetBestDecoder(AVCodecID codecId) {
    if (!s_initialized) {
        DecoderInfo softwareDecoder;
        softwareDecoder.type = DecoderType::SOFTWARE;
        softwareDecoder.name = "Software";
        softwareDecoder.available = true;
        return softwareDecoder;
    }
    
    // Prefer NVDEC for H264/H265 if available
    for (const auto& decoder : s_availableDecoders) {
        if (decoder.available && decoder.type == DecoderType::NVDEC && 
            SupportsCodec(decoder, codecId)) {
            return decoder;
        }
    }
    
    // Fallback to software decoding
    DecoderInfo softwareDecoder;
    softwareDecoder.type = DecoderType::SOFTWARE;
    softwareDecoder.name = "Software";
    softwareDecoder.available = true;
    return softwareDecoder;
}

bool HardwareDecoder::SupportsCodec(const DecoderInfo& decoder, AVCodecID codecId) {
    switch (decoder.type) {
        case DecoderType::NVDEC:
            return (codecId == AV_CODEC_ID_H264 || codecId == AV_CODEC_ID_HEVC);
        case DecoderType::SOFTWARE:
            return true;
        default:
            return false;
    }
}

void HardwareDecoder::DetectHardwareDecoders() {
    s_availableDecoders.clear();
    
    // Test NVDEC availability
    DecoderInfo nvdecDecoder;
    nvdecDecoder.type = DecoderType::NVDEC;
    nvdecDecoder.name = "NVIDIA NVDEC";
    nvdecDecoder.hwDeviceType = AV_HWDEVICE_TYPE_CUDA;
    nvdecDecoder.available = TestNVDECAvailability();
    s_availableDecoders.push_back(nvdecDecoder);
    
    // Software decoding is always available
    DecoderInfo softwareDecoder;
    softwareDecoder.type = DecoderType::SOFTWARE;
    softwareDecoder.name = "Software";
    softwareDecoder.available = true;
    s_availableDecoders.push_back(softwareDecoder);
}

bool HardwareDecoder::TestNVDECAvailability() {
#if HAVE_CUDA
    AVBufferRef* hwDeviceCtx = nullptr;
    
    // Try to create CUDA hardware device context
    int ret = av_hwdevice_ctx_create(&hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        std::cout << "NVDEC not available: Failed to create CUDA device context: " << errorBuf << "\n";
        return false;
    }
    
    // Test if we can find NVDEC decoders
    bool h264Available = false;
    bool h265Available = false;
    
    // Check for H264 NVDEC decoder
    const AVCodec* h264Decoder = avcodec_find_decoder_by_name("h264_cuvid");
    if (h264Decoder) {
        h264Available = true;
        std::cout << "H264 NVDEC decoder found\n";
    }
    
    // Check for H265 NVDEC decoder  
    const AVCodec* h265Decoder = avcodec_find_decoder_by_name("hevc_cuvid");
    if (h265Decoder) {
        h265Available = true;
        std::cout << "H265 NVDEC decoder found\n";
    }
    
    av_buffer_unref(&hwDeviceCtx);
    
    if (h264Available || h265Available) {
        std::cout << "NVDEC hardware decoding available\n";
        return true;
    } else {
        std::cout << "NVDEC hardware decoders not found\n";
        return false;
    }
#else
    std::cout << "NVDEC not available: CUDA support not compiled in\n";
    return false;
#endif
}