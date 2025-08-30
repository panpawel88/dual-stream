#include "VideoValidator.h"
#include "core/Logger.h"
#include "core/Config.h"
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
    
    LOG_INFO("FFmpeg version: ", av_version_info());
    LOG_INFO("Initializing video validation...");
    
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
    
    // Check against maximum configured resolution
    Config* config = Config::GetInstance();
    int maxWidth = config->GetInt("video.max_resolution_width", 7680);
    int maxHeight = config->GetInt("video.max_resolution_height", 4320);
    
    if (info.width > maxWidth || info.height > maxHeight) {
        info.errorMessage = "Video resolution " + std::to_string(info.width) + "x" + std::to_string(info.height) + 
                           " exceeds maximum supported resolution of " + std::to_string(maxWidth) + "x" + std::to_string(maxHeight) + 
                           " in " + filePath;
        avformat_close_input(&formatContext);
        return info;
    }
    
    LOG_INFO("Video info for ", filePath, ":");
    LOG_INFO("  Resolution: ", info.width, "x", info.height);
    LOG_INFO("  Frame rate: ", info.frameRate, " FPS");
    LOG_INFO("  Codec: ", info.codecName);
    LOG_INFO("  Duration: ", (info.duration / AV_TIME_BASE), " seconds");
    
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
    
    LOG_INFO("Video compatibility validation passed");
    LOG_INFO("Video 1: ", video1.width, "x", video1.height);
    LOG_INFO("Video 2: ", video2.width, "x", video2.height);
    
    return true;
}