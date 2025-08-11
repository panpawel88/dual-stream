#include "VideoValidator.h"
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

bool VideoValidator::s_initialized = false;

bool VideoValidator::Initialize() {
    if (s_initialized) {
        return true;
    }
    
    // Initialize FFmpeg
    av_log_set_level(AV_LOG_WARNING);
    
    std::cout << "FFmpeg version: " << av_version_info() << "\n";
    std::cout << "Initializing video validation...\n";
    
    s_initialized = true;
    return true;
}

void VideoValidator::Cleanup() {
    if (s_initialized) {
        s_initialized = false;
    }
}

VideoInfo VideoValidator::GetVideoInfo(const std::string& filePath) {
    VideoInfo info;
    
    if (!s_initialized) {
        info.errorMessage = "VideoValidator not initialized";
        return info;
    }
    
    AVFormatContext* formatContext = nullptr;
    
    // Open input file
    int ret = avformat_open_input(&formatContext, filePath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        info.errorMessage = "Cannot open file " + filePath + ": " + std::string(errorBuf);
        return info;
    }
    
    // Retrieve stream information
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        info.errorMessage = "Cannot find stream info for " + filePath + ": " + std::string(errorBuf);
        avformat_close_input(&formatContext);
        return info;
    }
    
    // Find video stream
    int videoStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }
    
    if (videoStreamIndex == -1) {
        info.errorMessage = "No video stream found in " + filePath;
        avformat_close_input(&formatContext);
        return info;
    }
    
    AVStream* videoStream = formatContext->streams[videoStreamIndex];
    AVCodecParameters* codecParams = videoStream->codecpar;
    
    // Extract video information
    info.width = codecParams->width;
    info.height = codecParams->height;
    info.duration = formatContext->duration;
    
    // Calculate frame rate
    if (videoStream->avg_frame_rate.num != 0 && videoStream->avg_frame_rate.den != 0) {
        info.frameRate = av_q2d(videoStream->avg_frame_rate);
    } else if (videoStream->r_frame_rate.num != 0 && videoStream->r_frame_rate.den != 0) {
        info.frameRate = av_q2d(videoStream->r_frame_rate);
    }
    
    // Get codec name
    const AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
    if (codec) {
        info.codecName = codec->name;
    }
    
    // Validate codec support (H264/H265)
    if (codecParams->codec_id != AV_CODEC_ID_H264 && codecParams->codec_id != AV_CODEC_ID_HEVC) {
        info.errorMessage = "Unsupported codec in " + filePath + ". Only H264 and H265 are supported. Found: " + info.codecName;
        avformat_close_input(&formatContext);
        return info;
    }
    
    // Validate resolution
    if (info.width <= 0 || info.height <= 0) {
        info.errorMessage = "Invalid video resolution in " + filePath + ": " + std::to_string(info.width) + "x" + std::to_string(info.height);
        avformat_close_input(&formatContext);
        return info;
    }
    
    std::cout << "Video info for " << filePath << ":\n";
    std::cout << "  Resolution: " << info.width << "x" << info.height << "\n";
    std::cout << "  Frame rate: " << info.frameRate << " FPS\n";
    std::cout << "  Codec: " << info.codecName << "\n";
    std::cout << "  Duration: " << (info.duration / AV_TIME_BASE) << " seconds\n";
    
    info.valid = true;
    
    avformat_close_input(&formatContext);
    return info;
}

bool VideoValidator::ValidateCompatibility(const VideoInfo& video1, const VideoInfo& video2, std::string& errorMessage) {
    if (!video1.valid) {
        errorMessage = "Video 1 is invalid: " + video1.errorMessage;
        return false;
    }
    
    if (!video2.valid) {
        errorMessage = "Video 2 is invalid: " + video2.errorMessage;
        return false;
    }
    
    // Check if resolutions match
    if (video1.width != video2.width || video1.height != video2.height) {
        errorMessage = "Video resolutions do not match. Video 1: " + 
                      std::to_string(video1.width) + "x" + std::to_string(video1.height) +
                      ", Video 2: " + std::to_string(video2.width) + "x" + std::to_string(video2.height);
        return false;
    }
    
    std::cout << "Video compatibility validation passed\n";
    std::cout << "Both videos: " << video1.width << "x" << video1.height << "\n";
    
    return true;
}