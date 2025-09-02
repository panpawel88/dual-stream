#include "CommandLineParser.h"
#include "Logger.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

VideoPlayerArgs CommandLineParser::Parse(int argc, char* argv[]) {
    VideoPlayerArgs args;
    
    if (argc < 2) {
        args.errorMessage = "Usage: " + std::string(argv[0]) + " <video1.mp4> [video2.mp4 video3.mp4 ...] [options]";
        args.errorMessage += "\nOptions:";
        args.errorMessage += "\n  --config=<path>                Configuration file path (default: config/default.ini)";
        args.errorMessage += "\n  --debug, -d                    Enable debug logging";
        args.errorMessage += "\n  --switching-algorithm=<alg>    Switching algorithm: immediate (default), predecoded, keyframe-sync";
        args.errorMessage += "\n  --trigger=<type>               Trigger type: keyboard (default), face_detection";
        args.errorMessage += "\n  --speed=<speed>                Playback speed: 0.05, 0.1, 0.2, 0.5, 1.0 (default), 2.0, 5.0, 10.0";
        args.errorMessage += "\nNote: Command line options override configuration file settings";
        return args;
    }
    
    // Parse video file arguments first (everything before options starting with --)
    int firstOptionIndex = argc; // Default to end if no options found
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--" || arg.substr(0, 1) == "-") {
            firstOptionIndex = i;
            break;
        }
    }
    
    // Collect video paths
    for (int i = 1; i < firstOptionIndex; i++) {
        args.videoPaths.push_back(argv[i]);
    }
    
    // Validate we have at least one video
    if (args.videoPaths.empty()) {
        args.errorMessage = "At least one video file must be specified";
        return args;
    }
    
    // Parse optional arguments
    const std::string algorithmErrorMsg = "\nAvailable algorithms: immediate, predecoded, keyframe-sync";
    const std::string triggerErrorMsg = "\nAvailable trigger types: keyboard, face, face_detection";
    const std::string speedErrorMsg = "\nSupported speeds: 0.05, 0.1, 0.2, 0.5, 1.0";
    
    for (int i = firstOptionIndex; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--debug" || arg == "-d") {
            args.debugLogging = true;
        } else if (arg.find("--config=") == 0) {
            args.configPath = arg.substr(9); // Skip "--config="
        } else if (arg.find("--switching-algorithm=") == 0 || arg.find("-s=") == 0) {
            std::string algorithmName;
            if (arg.find("--switching-algorithm=") == 0) {
                algorithmName = arg.substr(22); // Skip "--switching-algorithm="
            } else {
                algorithmName = arg.substr(3); // Skip "-s="
            }
            
            SwitchingAlgorithm parsedAlgorithm = VideoSwitchingStrategyFactory::ParseAlgorithm(algorithmName);
            if (parsedAlgorithm == static_cast<SwitchingAlgorithm>(-1)) {
                args.errorMessage = "Unknown switching algorithm: " + algorithmName + algorithmErrorMsg;
                return args;
            }
            args.switchingAlgorithm = parsedAlgorithm;
        } else if (arg.find("--trigger=") == 0 || arg.find("-t=") == 0) {
            std::string triggerName;
            if (arg.find("--trigger=") == 0) {
                triggerName = arg.substr(10); // Skip "--trigger="
            } else {
                triggerName = arg.substr(3); // Skip "-t="
            }
            
            args.triggerType = SwitchingTriggerFactory::ParseTriggerType(triggerName);
        } else if (arg.find("--speed=") == 0) {
            std::string speedStr = arg.substr(8); // Skip "--speed="
            try {
                double speed = std::stod(speedStr);
                if (speed == 0.05 || speed == 0.1 || speed == 0.2 || speed == 0.5 || speed == 1.0) {
                    args.playbackSpeed = speed;
                } else {
                    args.errorMessage = "Invalid playback speed: " + speedStr + speedErrorMsg;
                    return args;
                }
            } catch (const std::exception& e) {
                args.errorMessage = "Invalid playback speed format: " + speedStr + speedErrorMsg;
                return args;
            }
        } else {
            args.errorMessage = "Unknown option: " + arg;
            args.errorMessage += "\nValid options: --config=<path>, --debug, --switching-algorithm=<algorithm>, --trigger=<trigger>, --speed=<speed>";
            return args;
        }
    }
    
    // Check if all video files exist and have valid extensions
    for (size_t i = 0; i < args.videoPaths.size(); i++) {
        const std::string& videoPath = args.videoPaths[i];
        
        if (!FileExists(videoPath)) {
            args.errorMessage = "Video file " + std::to_string(i + 1) + " does not exist: " + videoPath;
            return args;
        }
        
        if (!HasValidExtension(videoPath)) {
            args.errorMessage = "Video file " + std::to_string(i + 1) + " must be an MP4 file: " + videoPath;
            return args;
        }
    }
    
    args.valid = true;
    return args;
}

bool CommandLineParser::FileExists(const std::string& path) {
    try {
        return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
    } catch (const std::filesystem::filesystem_error& e) {
        LOG_ERROR("Filesystem error checking file ", path, ": ", e.what());
        return false;
    }
}

bool CommandLineParser::HasValidExtension(const std::string& path) {
    try {
        std::filesystem::path filePath(path);
        std::string extension = filePath.extension().string();
        
        // Convert to lowercase for comparison
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        return extension == ".mp4";
    } catch (const std::exception& e) {
        LOG_ERROR("Error checking file extension for ", path, ": ", e.what());
        return false;
    }
}