#pragma once

#include <string>
#include <vector>

struct VideoPlayerArgs {
    std::string video1Path;
    std::string video2Path;
    bool valid;
    std::string errorMessage;
    
    VideoPlayerArgs() : valid(false) {}
};

class CommandLineParser {
public:
    static VideoPlayerArgs Parse(int argc, char* argv[]);
    
private:
    static bool FileExists(const std::string& path);
    static bool HasValidExtension(const std::string& path);
};