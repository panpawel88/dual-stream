#pragma once

#include <string>
#include <vector>
#include "VideoSwitchingStrategy.h"

struct VideoPlayerArgs {
    std::string video1Path;
    std::string video2Path;
    SwitchingAlgorithm switchingAlgorithm;
    bool valid;
    bool debugLogging;
    std::string errorMessage;
    
    VideoPlayerArgs() : switchingAlgorithm(SwitchingAlgorithm::IMMEDIATE), valid(false), debugLogging(false) {}
};

class CommandLineParser {
public:
    static VideoPlayerArgs Parse(int argc, char* argv[]);
    
private:
    static bool FileExists(const std::string& path);
    static bool HasValidExtension(const std::string& path);
};