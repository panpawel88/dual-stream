#pragma once

#include <string>
#include <vector>

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
    static bool ValidateCompatibility(const std::vector<VideoInfo>& videos, std::string& errorMessage);
    static bool ValidateMultipleVideos(const std::vector<std::string>& videoPaths, std::vector<VideoInfo>& videoInfos, std::string& errorMessage);
    
private:
    static bool s_initialized;
};