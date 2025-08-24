#pragma once

#include <string>
#include <vector>
#include "video/switching/VideoSwitchingStrategy.h"

struct VideoPlayerArgs {
    std::string video1Path;
    std::string video2Path;
    SwitchingAlgorithm switchingAlgorithm;
    double playbackSpeed;
    bool valid;
    bool debugLogging;
    std::string errorMessage;
    
    VideoPlayerArgs() : switchingAlgorithm(SwitchingAlgorithm::IMMEDIATE), playbackSpeed(1.0), valid(false), debugLogging(false) {}
};

class CommandLineParser {
public:
    static VideoPlayerArgs Parse(int argc, char* argv[]);
    
private:
    static bool FileExists(const std::string& path);
    static bool HasValidExtension(const std::string& path);
};