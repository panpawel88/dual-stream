#include "CommandLineParser.h"
#include "Logger.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

VideoPlayerArgs CommandLineParser::Parse(int argc, char* argv[]) {
    VideoPlayerArgs args;
    
    if (argc < 3) {
        args.errorMessage = "Usage: " + std::string(argv[0]) + " <video1.mp4> <video2.mp4> [--debug] [--switching-algorithm=<algorithm>] [--speed=<speed>]";
        args.errorMessage += "\nSwitching algorithms: immediate (default), predecoded, keyframe-sync";
        args.errorMessage += "\nPlayback speeds: 0.05, 0.1, 0.2, 0.5, 1.0 (default)";
        return args;
    }
    
    args.video1Path = argv[1];
    args.video2Path = argv[2];
    
    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--debug" || arg == "-d") {
            args.debugLogging = true;
        } else if (arg.find("--switching-algorithm=") == 0) {
            std::string algorithmName = arg.substr(22); // Skip "--switching-algorithm="
            args.switchingAlgorithm = VideoSwitchingStrategyFactory::ParseAlgorithm(algorithmName);
            if (args.switchingAlgorithm == static_cast<SwitchingAlgorithm>(-1)) {
                args.errorMessage = "Unknown switching algorithm: " + algorithmName;
                args.errorMessage += "\nAvailable algorithms: immediate, predecoded, keyframe-sync";
                return args;
            }
        } else if (arg.find("-s=") == 0) {
            std::string algorithmName = arg.substr(3); // Skip "-s="
            args.switchingAlgorithm = VideoSwitchingStrategyFactory::ParseAlgorithm(algorithmName);
            if (args.switchingAlgorithm == static_cast<SwitchingAlgorithm>(-1)) {
                args.errorMessage = "Unknown switching algorithm: " + algorithmName;
                args.errorMessage += "\nAvailable algorithms: immediate, predecoded, keyframe-sync";
                return args;
            }
        } else if (arg.find("--speed=") == 0) {
            std::string speedStr = arg.substr(8); // Skip "--speed="
            try {
                double speed = std::stod(speedStr);
                if (speed == 0.05 || speed == 0.1 || speed == 0.2 || speed == 0.5 || speed == 1.0) {
                    args.playbackSpeed = speed;
                } else {
                    args.errorMessage = "Invalid playback speed: " + speedStr;
                    args.errorMessage += "\nSupported speeds: 0.05, 0.1, 0.2, 0.5, 1.0";
                    return args;
                }
            } catch (const std::exception& e) {
                args.errorMessage = "Invalid playback speed format: " + speedStr;
                args.errorMessage += "\nSupported speeds: 0.05, 0.1, 0.2, 0.5, 1.0";
                return args;
            }
        } else {
            args.errorMessage = "Unknown option: " + arg;
            args.errorMessage += "\nValid options: --debug, --switching-algorithm=<algorithm>, --speed=<speed>";
            return args;
        }
    }
    
    // Check if files exist
    if (!FileExists(args.video1Path)) {
        args.errorMessage = "Video file 1 does not exist: " + args.video1Path;
        return args;
    }
    
    if (!FileExists(args.video2Path)) {
        args.errorMessage = "Video file 2 does not exist: " + args.video2Path;
        return args;
    }
    
    // Check file extensions
    if (!HasValidExtension(args.video1Path)) {
        args.errorMessage = "Video file 1 must be an MP4 file: " + args.video1Path;
        return args;
    }
    
    if (!HasValidExtension(args.video2Path)) {
        args.errorMessage = "Video file 2 must be an MP4 file: " + args.video2Path;
        return args;
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