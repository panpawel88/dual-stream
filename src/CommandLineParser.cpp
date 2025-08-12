#include "CommandLineParser.h"
#include "Logger.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

VideoPlayerArgs CommandLineParser::Parse(int argc, char* argv[]) {
    VideoPlayerArgs args;
    
    if (argc < 3 || argc > 4) {
        args.errorMessage = "Usage: " + std::string(argv[0]) + " <video1.mp4> <video2.mp4> [--debug]";
        return args;
    }
    
    args.video1Path = argv[1];
    args.video2Path = argv[2];
    
    // Check for debug flag
    if (argc == 4) {
        std::string debugFlag = argv[3];
        if (debugFlag == "--debug" || debugFlag == "-d") {
            args.debugLogging = true;
        } else {
            args.errorMessage = "Unknown option: " + debugFlag + ". Use --debug or -d for debug logging.";
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