# Core System Services

This directory provides foundational services used throughout the application, including command line parsing, logging, and FFmpeg initialization.

## Architecture Overview

The core system provides clean, reusable services that abstract common functionality:

```
src/core/
├── CommandLineParser.h/cpp   # Command line argument processing
├── Logger.h/cpp             # Application logging system  
└── FFmpegInitializer.h/cpp  # FFmpeg library initialization
```

## Core Components

### CommandLineParser
**File:** `CommandLineParser.h/cpp`
**Purpose:** Advanced command line argument parsing with validation

**Supported Arguments Structure:**
```cpp
struct VideoPlayerArgs {
    std::string video1Path;              // Required: First video file
    std::string video2Path;              // Required: Second video file
    SwitchingAlgorithm switchingAlgorithm; // Optional: --algorithm, -a
    TriggerType triggerType;             // Optional: --trigger, -t
    double playbackSpeed;                // Optional: --speed, -s
    bool debugLogging;                   // Optional: --debug, -d
    bool valid;                          // Parse result validation
    std::string errorMessage;            // Detailed error information
};
```

**Command Line Interface:**
```bash
# Basic usage (required arguments)
./dual_stream video1.mp4 video2.mp4

# Advanced usage with optional parameters
./dual_stream video1.mp4 video2.mp4 --algorithm predecoded --speed 1.5 --debug
./dual_stream video1.mp4 video2.mp4 -a keyframe-sync -t keyboard -s 0.5 -d

# Algorithm options
--algorithm immediate     # Default: seek new video to current time  
--algorithm predecoded    # Decode both streams simultaneously
--algorithm keyframe-sync # Wait for synchronized keyframes

# Trigger options  
--trigger keyboard        # Default: keys 1 and 2

# Playback speed
--speed 0.5              # Half speed
--speed 2.0              # Double speed

# Debug logging
--debug                  # Enable verbose debug output
```

**Validation Features:**
```cpp
class CommandLineParser {
private:
    static bool FileExists(const std::string& path);     // File existence validation
    static bool HasValidExtension(const std::string& path); // .mp4 extension check
    
public:
    static VideoPlayerArgs Parse(int argc, char* argv[]);
};
```

**Error Handling:**
- **File Validation:** Checks file existence and .mp4 extension
- **Parameter Validation:** Validates algorithm and trigger type strings
- **Range Checking:** Ensures playback speed is positive
- **Detailed Errors:** Specific error messages for each failure type

### Logger System
**File:** `Logger.h/cpp`
**Purpose:** Centralized logging with configurable verbosity levels

**Log Levels:**
```cpp
enum class LogLevel {
    Debug,    // Detailed debugging information
    Info,     // General information messages
    Warning,  // Warning conditions
    Error     // Error conditions
};
```

**Singleton Pattern:**
```cpp
class Logger {
public:
    static Logger& GetInstance();           // Singleton access
    void SetLogLevel(LogLevel level);       // Configure verbosity
    void Log(LogLevel level, const Args&... args); // Variadic template logging
    
private:
    Logger() = default;                     // Private constructor
    LogLevel m_currentLevel = LogLevel::Info; // Default level
    mutable std::mutex m_mutex;             // Thread safety
};
```

**Logging Macros:**
```cpp
#define LOG_DEBUG(...) Logger::GetInstance().Log(LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...)  Logger::GetInstance().Log(LogLevel::Info, __VA_ARGS__)
#define LOG_WARNING(...) Logger::GetInstance().Log(LogLevel::Warning, __VA_ARGS__)
#define LOG_ERROR(...) Logger::GetInstance().Log(LogLevel::Error, __VA_ARGS__)
```

**Usage Examples:**
```cpp
// Simple logging
LOG_INFO("Application started");
LOG_ERROR("Failed to initialize renderer");

// Formatted logging with multiple arguments
LOG_INFO("Video resolution: ", width, "x", height);
LOG_DEBUG("Decoded frame at time: ", presentationTime, " seconds");
LOG_WARNING("Hardware decoder unavailable, using software fallback");

// Conditional logging based on level
Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
```

**Output Format:**
```
[INFO] DualStream Video Player v1.0.0
[INFO] Video 1: /path/to/video1.mp4
[INFO] Video 2: /path/to/video2.mp4
[DEBUG] Decoded frame at time: 1.2345 seconds
[WARNING] Hardware decoder unavailable, using software fallback
[ERROR] Failed to initialize renderer
```

### FFmpegInitializer
**File:** `FFmpegInitializer.h/cpp`
**Purpose:** Centralized FFmpeg library initialization and configuration

**RAII Initialization:**
```cpp
class FFmpegInitializer {
public:
    FFmpegInitializer();                    // Constructor initializes FFmpeg
    ~FFmpegInitializer();                   // Destructor cleans up (if needed)
    
    bool Initialize();                      // Explicit initialization
    void Cleanup();                         // Explicit cleanup
    
private:
    bool m_initialized = false;             // Initialization state tracking
};
```

**Initialization Responsibilities:**
- **FFmpeg Library Init:** Calls required FFmpeg initialization functions
- **Logging Configuration:** Sets FFmpeg log level to reduce noise  
- **Version Reporting:** Reports FFmpeg version information
- **Error Handling:** Validates successful initialization

**Integration Pattern:**
```cpp
int main(int argc, char* argv[]) {
    // Parse command line
    VideoPlayerArgs args = CommandLineParser::Parse(argc, argv);
    if (!args.valid) {
        LOG_ERROR("Error: ", args.errorMessage);
        return 1;
    }
    
    // Configure logging
    Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
    
    // Initialize FFmpeg
    FFmpegInitializer ffmpegInit;
    if (!ffmpegInit.Initialize()) {
        LOG_ERROR("Failed to initialize FFmpeg");
        return 1;
    }
    
    // Continue with application logic...
}
```

## Command Line Processing Details

### Argument Parsing Logic
```cpp
VideoPlayerArgs CommandLineParser::Parse(int argc, char* argv[]) {
    VideoPlayerArgs args;
    
    // Require minimum 3 arguments (program + 2 videos)
    if (argc < 3) {
        args.errorMessage = "Usage: dual_stream <video1> <video2> [options]";
        return args;
    }
    
    // Extract required video paths
    args.video1Path = argv[1];
    args.video2Path = argv[2];
    
    // Validate video files
    if (!FileExists(args.video1Path) || !HasValidExtension(args.video1Path)) {
        args.errorMessage = "Invalid video file: " + args.video1Path;
        return args;
    }
    
    // Process optional arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--algorithm" || arg == "-a") {
            if (i + 1 < argc) {
                args.switchingAlgorithm = VideoSwitchingStrategyFactory::ParseAlgorithm(argv[++i]);
            }
        } else if (arg == "--speed" || arg == "-s") {
            if (i + 1 < argc) {
                args.playbackSpeed = std::stod(argv[++i]);
                if (args.playbackSpeed <= 0.0) {
                    args.errorMessage = "Playback speed must be positive";
                    return args;
                }
            }
        } else if (arg == "--debug" || arg == "-d") {
            args.debugLogging = true;
        }
        // Additional argument processing...
    }
    
    args.valid = true;
    return args;
}
```

### File Validation
```cpp
bool CommandLineParser::FileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

bool CommandLineParser::HasValidExtension(const std::string& path) {
    size_t lastDot = path.find_last_of('.');
    if (lastDot == std::string::npos) return false;
    
    std::string extension = path.substr(lastDot);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return extension == ".mp4";
}
```

## Logging System Implementation

### Thread Safety
```cpp
template<typename... Args>
void Logger::Log(LogLevel level, const Args&... args) {
    if (level < m_currentLevel) return;  // Level filtering
    
    std::lock_guard<std::mutex> lock(m_mutex);  // Thread safety
    
    // Output level prefix
    switch (level) {
        case LogLevel::Debug:   std::cout << "[DEBUG] "; break;
        case LogLevel::Info:    std::cout << "[INFO] "; break;
        case LogLevel::Warning: std::cout << "[WARNING] "; break;
        case LogLevel::Error:   std::cout << "[ERROR] "; break;
    }
    
    // Variadic template argument processing
    ((std::cout << args), ...);
    std::cout << std::endl;
}
```

### Variadic Template Support
**Modern C++ Feature:** Type-safe, efficient argument processing
```cpp
// Supports any number and type of arguments
LOG_INFO("Video ", videoNumber, " resolution: ", width, "x", height, " at ", frameRate, " FPS");

// Automatic type conversion and formatting
LOG_DEBUG("PTS: ", pts, ", DTS: ", dts, ", valid: ", (frame.valid ? "true" : "false"));
```

## FFmpeg Integration

### Library Initialization
```cpp
bool FFmpegInitializer::Initialize() {
    if (m_initialized) return true;
    
    // Configure FFmpeg logging to reduce noise
    av_log_set_level(AV_LOG_WARNING);
    
    // Report FFmpeg version
    LOG_INFO("FFmpeg version: ", av_version_info());
    
    // Additional FFmpeg setup as needed...
    
    m_initialized = true;
    return true;
}
```

### Error Handling Integration
```cpp
// FFmpeg error reporting using core logging system
char errorBuf[AV_ERROR_MAX_STRING_SIZE];
av_strerror(ret, errorBuf, sizeof(errorBuf));
LOG_ERROR("FFmpeg operation failed: ", errorBuf, " (code: ", ret, ")");
```

## Integration Patterns

### Application Initialization Sequence
```cpp
int main(int argc, char* argv[]) {
    // 1. Parse command line arguments
    VideoPlayerArgs args = CommandLineParser::Parse(argc, argv);
    if (!args.valid) {
        LOG_ERROR("Error: ", args.errorMessage);
        return 1;
    }
    
    // 2. Configure logging based on arguments
    Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
    
    // 3. Initialize FFmpeg
    FFmpegInitializer ffmpegInit;
    if (!ffmpegInit.Initialize()) {
        LOG_ERROR("Failed to initialize FFmpeg");
        return 1;
    }
    
    // 4. Log application startup information
    LOG_INFO("DualStream Video Player v1.0.0");
    LOG_INFO("Video 1: ", args.video1Path);
    LOG_INFO("Video 2: ", args.video2Path);
    LOG_INFO("Switching Algorithm: ", VideoSwitchingStrategyFactory::GetAlgorithmName(args.switchingAlgorithm));
    LOG_INFO("Playback Speed: ", args.playbackSpeed, "x");
    
    // 5. Continue with application initialization...
}
```

### Cross-System Logging
```cpp
// Video system using core logging
LOG_INFO("Initializing video decoder with ", decoderInfo.name);

// Rendering system using core logging  
LOG_INFO("Initialized ", RendererFactory::GetRendererName(), " renderer");

// UI system using core logging
LOG_INFO("Window created. Press 1/2 to switch videos, F11 for fullscreen, ESC to exit");
```

This core system provides essential application services with clean interfaces, comprehensive error handling, and seamless integration across all application subsystems.