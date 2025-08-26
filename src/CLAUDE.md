# DualStream Video Player - Source Code Architecture

This directory contains the complete source code for the DualStream Video Player, a sophisticated dual-video switching application with hardware acceleration and multiple rendering backends.

## System Architecture Overview

The application is organized into five major subsystems, each with clear responsibilities and well-defined interfaces:

```
src/
├── main.cpp             # Application entry point and orchestration
├── core/                # Foundation services (logging, CLI, FFmpeg init)
├── ui/                  # Window management and input handling
├── video/               # Complete video processing pipeline
└── rendering/           # Multi-backend rendering system
```

## Application Entry Point

### main.cpp
**Purpose:** Application orchestration and main execution loop
**Key Responsibilities:**
- Command line argument processing and validation
- System initialization sequence coordination  
- Main event loop with cooperative multitasking
- Error handling and graceful shutdown

**Architecture Pattern:**
```cpp
int main(int argc, char* argv[]) {
    // 1. Core Services Initialization
    VideoPlayerArgs args = CommandLineParser::Parse(argc, argv);
    Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
    FFmpegInitializer ffmpegInit;
    
    // 2. Video Validation
    VideoInfo video1Info = VideoValidator::GetVideoInfo(args.video1Path);
    VideoInfo video2Info = VideoValidator::GetVideoInfo(args.video2Path);
    
    // 3. UI System Creation
    Window window;
    window.Create("DualStream Video Player", windowWidth, windowHeight);
    
    // 4. Rendering System Initialization  
    auto renderer = RendererFactory::CreateRenderer();
    renderer->Initialize(window.GetHandle(), windowWidth, windowHeight);
    
    // 5. Video System Initialization
    VideoManager videoManager;
    videoManager.Initialize(args.video1Path, args.video2Path, renderer.get(), 
                           args.switchingAlgorithm, args.playbackSpeed);
    
    // 6. Input System Setup
    auto switchingTrigger = SwitchingTriggerFactory::Create(args.triggerType, &window);
    videoManager.SetSwitchingTrigger(std::move(switchingTrigger));
    
    // 7. Main Execution Loop
    while (window.ProcessMessages()) {
        // Dynamic window resizing
        // Input trigger processing  
        // Frame-rate controlled video updates
        // Texture conversion and rendering
    }
}
```

## Subsystem Integration Architecture

### Initialization Sequence
**Dependency Order:** Core → UI → Rendering → Video → Input
```cpp
// Stage 1: Core Services
CommandLineParser → Logger → FFmpegInitializer

// Stage 2: UI Foundation  
Window Creation → Display Configuration

// Stage 3: Rendering Backend
RendererFactory → IRenderer → Hardware Detection

// Stage 4: Video Processing
VideoValidator → VideoManager → Strategy/Trigger Setup

// Stage 5: System Coordination
Main Event Loop → Cooperative Processing
```

### Data Flow Architecture
**Frame Processing Pipeline:**
```
Video Files → VideoDemuxer → VideoDecoder → VideoSwitchingStrategy
      ↓
DecodedFrame → TextureConverter → RenderTexture → IRenderer
      ↓
Display Output
```

**Input Processing Pipeline:**
```
Win32 Messages → Window → ISwitchingTrigger → VideoManager
      ↓
VideoSwitchingStrategy → Video Stream Switching
```

## Major System Capabilities

### Multi-Resolution Video Support
**Change from Original Requirements:** Videos no longer need identical resolutions
```cpp
// Dynamic window sizing based on maximum video dimensions
int maxVideoWidth = std::max(video1Info.width, video2Info.width);
int maxVideoHeight = std::max(video1Info.height, video2Info.height);

// Window size constrained by display resolution
int windowWidth = std::min(maxVideoWidth, GetSystemMetrics(SM_CXSCREEN));
int windowHeight = std::min(maxVideoHeight, GetSystemMetrics(SM_CYSCREEN));
```

### Advanced Switching Strategies
**Strategy Pattern Implementation:** Pluggable switching algorithms
```cpp
enum class SwitchingAlgorithm {
    IMMEDIATE,      // Default: seek and resume at current time
    PREDECODED,     // Simultaneous decoding for zero-latency switching
    KEYFRAME_SYNC   // Synchronized keyframe switching for quality
};

auto strategy = VideoSwitchingStrategyFactory::Create(args.switchingAlgorithm);
videoManager.SetSwitchingStrategy(std::move(strategy));
```

### Dual Rendering Backend
**Compile-Time Selection:** DirectX 11 or OpenGL rendering
```cpp
#if USE_OPENGL_RENDERER
    // OpenGL 4.6 Core Profile with CUDA interop
    auto renderer = std::make_unique<OpenGLRenderer>();
#else  
    // DirectX 11 with D3D11VA hardware acceleration
    auto renderer = std::make_unique<D3D11Renderer>();
#endif
```

### Hardware Acceleration Pipeline
**Multi-Backend Hardware Support:**
- **DirectX Path:** NVDEC decode → D3D11 texture → D3D11 render
- **OpenGL Path:** NVDEC decode → CUDA memory → OpenGL interop
- **Software Fallback:** CPU decode → format conversion → GPU upload

### Dynamic Window Management
**Modern UI Features:**
- **Resizable Windows:** Dynamic renderer adjustment on size changes
- **Fullscreen Support:** F11 toggle with state preservation
- **Multi-Monitor Aware:** Window size limited by display resolution

## Performance Characteristics

### Memory Usage Patterns
```cpp
// Switching Strategy Impact on Memory
IMMEDIATE Strategy:    ~1x base memory (single active stream)
PREDECODED Strategy:   ~2x base memory (dual active streams)  
KEYFRAME_SYNC Strategy: ~1x base memory (single active stream)

// Rendering Backend Memory
Hardware Rendering: GPU memory only (zero CPU↔GPU transfers)
Software Rendering: CPU + GPU memory (temporary double buffering)
```

### CPU/GPU Load Distribution
```cpp
// Hardware Acceleration (Optimal)
Container Parsing:  ~5% CPU
Hardware Decoding: ~10% GPU (NVDEC units)
Format Conversion: ~5% GPU (shaders)
Rendering:        ~5% GPU (3D pipeline)

// Software Fallback
Container Parsing:  ~5% CPU  
Software Decoding: ~40% CPU
Format Conversion: ~20% CPU
Rendering:        ~5% GPU
```

### Threading Model
**Single-Threaded Design Benefits:**
- **Simplicity:** No threading complexity or synchronization overhead
- **Deterministic:** Predictable execution order and timing
- **Integration-Friendly:** Clean Win32 message loop integration
- **Debug-Friendly:** Simplified debugging and profiling

## Error Handling Strategy

### Layered Error Handling
```cpp
// Level 1: Core Service Failures (Fatal)
if (!ffmpegInit.Initialize()) {
    LOG_ERROR("Failed to initialize FFmpeg");
    return 1;  // Immediate application exit
}

// Level 2: Subsystem Failures (Fatal)  
if (!renderer->Initialize(...)) {
    LOG_ERROR("Failed to initialize renderer");
    return 1;  // Cannot continue without rendering
}

// Level 3: Runtime Failures (Recoverable)
if (!videoManager.UpdateFrame()) {
    LOG_ERROR("Failed to update video frame");
    // Continue with last valid frame
}

// Level 4: Hardware Fallback (Automatic)
if (!InitializeHardwareDecoder(...)) {
    LOG_INFO("Hardware decoding failed, falling back to software");
    InitializeSoftwareDecoder(...);  // Graceful degradation
}
```

### Validation and Compatibility
```cpp
// Comprehensive input validation
VideoInfo video1Info = VideoValidator::GetVideoInfo(args.video1Path);
VideoInfo video2Info = VideoValidator::GetVideoInfo(args.video2Path);

std::string compatibilityError;
if (!VideoValidator::ValidateCompatibility(video1Info, video2Info, compatibilityError)) {
    LOG_ERROR("Error: ", compatibilityError);
    return 1;
}
```

## Configuration and Extensibility

### Command Line Configuration
```bash
# Algorithm Selection
./dual_stream video1.mp4 video2.mp4 --algorithm predecoded

# Playback Speed Control
./dual_stream video1.mp4 video2.mp4 --speed 1.5

# Debug Output  
./dual_stream video1.mp4 video2.mp4 --debug

# Combined Options
./dual_stream video1.mp4 video2.mp4 -a keyframe-sync -s 0.8 -d
```

### Extensibility Framework
**Adding New Switching Strategies:**
1. Extend `VideoSwitchingStrategy` base class
2. Add to `SwitchingAlgorithm` enum
3. Update `VideoSwitchingStrategyFactory`

**Adding New Trigger Types:**
1. Implement `ISwitchingTrigger` interface  
2. Add to `TriggerType` enum
3. Update `SwitchingTriggerFactory`

**Adding New Renderers:**
1. Implement `IRenderer` interface
2. Add to `RendererType` enum  
3. Update `RendererFactory` and build system

## Subsystem Documentation

For detailed implementation information, see the individual subsystem documentation:

### Core Foundation
- **[core/CLAUDE.md](core/CLAUDE.md)** - Foundation services (logging, CLI parsing, FFmpeg init)

### User Interface  
- **[ui/CLAUDE.md](ui/CLAUDE.md)** - Window management, input handling, fullscreen support

### Video Processing Pipeline
- **[video/CLAUDE.md](video/CLAUDE.md)** - Complete video system overview
- **[video/demux/CLAUDE.md](video/demux/CLAUDE.md)** - Container parsing and packet extraction
- **[video/decode/CLAUDE.md](video/decode/CLAUDE.md)** - Hardware-accelerated decoding  
- **[video/switching/CLAUDE.md](video/switching/CLAUDE.md)** - Video switching strategies
- **[video/triggers/CLAUDE.md](video/triggers/CLAUDE.md)** - Input handling and switching triggers

### Rendering System
- **[rendering/CLAUDE.md](rendering/CLAUDE.md)** - Multi-backend rendering with hardware acceleration

## Build Configuration

### Renderer Selection
```cmake
# CMake configuration for renderer selection
option(USE_OPENGL_RENDERER "Use OpenGL renderer instead of DirectX 11" OFF)

if(USE_OPENGL_RENDERER)
    add_definitions(-DUSE_OPENGL_RENDERER=1)
    # Link OpenGL and CUDA libraries
else()
    add_definitions(-DUSE_OPENGL_RENDERER=0) 
    # Link DirectX 11 libraries
endif()
```

### Hardware Acceleration Support
```cmake
# CUDA support detection
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    add_definitions(-DHAVE_CUDA=1)
    enable_language(CUDA)
else()
    add_definitions(-DHAVE_CUDA=0)
endif()
```

This source architecture provides a robust, extensible foundation for dual-video switching with comprehensive hardware acceleration support and clean separation of concerns across all major subsystems.