#pragma once

#include <string>

struct VideoInfo {
    int width;
    int height;
    double frameRate;
    int64_t duration;
    std::string codecName;
    bool valid;
    std::string errorMessage;
    
    VideoInfo() : width(0), height(0), frameRate(0.0), duration(0), valid(false) {}
};

class VideoValidator {
public:
    static bool Initialize();
    static void Cleanup();
    static VideoInfo GetVideoInfo(const std::string& filePath);
    static bool ValidateCompatibility(const VideoInfo& video1, const VideoInfo& video2, std::string& errorMessage);
    
private:
    static bool s_initialized;
};