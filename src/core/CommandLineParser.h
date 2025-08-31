#pragma once

#include <string>
#include <vector>
#include <optional>
#include "video/switching/VideoSwitchingStrategy.h"
#include "video/triggers/SwitchingTriggerFactory.h"

struct VideoPlayerArgs {
    std::string video1Path;
    std::string video2Path;
    std::optional<SwitchingAlgorithm> switchingAlgorithm;
    TriggerType triggerType;
    double playbackSpeed;
    bool valid;
    bool debugLogging;
    std::string errorMessage;
    std::string configPath;
    
    VideoPlayerArgs() : switchingAlgorithm(std::nullopt), triggerType(TriggerType::KEYBOARD), playbackSpeed(1.0), valid(false), debugLogging(false) {}
};

class CommandLineParser {
public:
    static VideoPlayerArgs Parse(int argc, char* argv[]);
    
private:
    static bool FileExists(const std::string& path);
    static bool HasValidExtension(const std::string& path);
};