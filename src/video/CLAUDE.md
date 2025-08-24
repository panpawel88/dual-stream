# Video Processing System

This directory contains the complete video processing pipeline, from container parsing through hardware-accelerated decoding to intelligent switching between multiple video streams.

## System Architecture

The video system is organized into five major subsystems, each with clear responsibilities and interfaces:

```
src/video/
├── demux/          # Container parsing and packet extraction
├── decode/         # Hardware-accelerated video decoding  
├── switching/      # Video switching strategies and algorithms
├── triggers/       # Input handling for switching decisions
├── VideoManager    # Central coordination and stream management
└── VideoValidator  # Video file validation and compatibility checking
```

## Core Components

### VideoManager
**File:** `VideoManager.h/cpp`
**Purpose:** Central orchestrator for all video operations

**Key Responsibilities:**
- **Stream Coordination:** Manages two video streams with synchronized playback
- **Strategy Integration:** Delegates switching behavior to pluggable strategies  
- **Trigger Processing:** Handles input events through trigger system
- **Timing Management:** Maintains accurate playback timing and frame rates
- **Renderer Integration:** Coordinates with both D3D11 and OpenGL renderers

**Architecture Pattern:**
```cpp
class VideoManager {
private:
    VideoStream m_videos[2];                                    // Dual video streams
    std::unique_ptr<VideoSwitchingStrategy> m_switchingStrategy; // Pluggable switching
    std::unique_ptr<ISwitchingTrigger> m_switchingTrigger;      // Input handling
    
    // Timing and synchronization
    std::chrono::steady_clock::time_point m_playbackStartTime;
    double m_frameInterval;    // Based on video frame rate
    double m_playbackSpeed;    // Configurable playback speed
};
```

### VideoStream Structure
**Definition:** `VideoManager.h`
**Purpose:** Complete video stream encapsulation
```cpp
struct VideoStream {
    VideoDemuxer demuxer;      // Container parsing
    VideoDecoder decoder;      // Hardware/software decoding
    DecodedFrame currentFrame; // Current decoded frame
    VideoState state;          // Playback state
    double currentTime;        // Current playback position
    double duration;           // Stream duration
    bool initialized;          // Initialization status
};
```

### VideoValidator  
**File:** `VideoValidator.h/cpp`
**Purpose:** Video file validation and compatibility checking

**Key Features:**
- **Codec Validation:** Ensures H.264/H.265 codec support
- **Resolution Extraction:** Extracts video dimensions and properties
- **Compatibility Checking:** Validates video file compatibility (no longer requires identical resolutions)
- **Error Reporting:** Detailed error messages for validation failures

## Subsystem Integration

### Container Parsing → Decoding
**Directory:** [demux/CLAUDE.md](demux/CLAUDE.md)
```cpp
// VideoManager initialization flow
VideoStream& stream = m_videos[0];

// 1. Open container and extract stream info
stream.demuxer.Open(filePath);
AVCodecParameters* codecParams = stream.demuxer.GetCodecParameters();

// 2. Initialize decoder with container's codec parameters  
DecoderInfo decoderInfo = HardwareDecoder::GetBestDecoder(stream.demuxer.GetCodecID());
stream.decoder.Initialize(codecParams, decoderInfo, d3dDevice, stream.demuxer.GetTimeBase());

// 3. Set stream properties from container
stream.duration = stream.demuxer.GetDuration();
```

### Decoding → Switching  
**Directory:** [decode/CLAUDE.md](decode/CLAUDE.md) → [switching/CLAUDE.md](switching/CLAUDE.md)
```cpp
// Switching strategies coordinate with decoders
bool VideoSwitchingStrategy::UpdateFrame() {
    VideoStream& activeStream = m_streams[static_cast<int>(m_activeVideo)];
    
    // Read packet from demuxer
    AVPacket packet;
    activeStream.demuxer.ReadFrame(&packet);
    
    // Send to decoder
    activeStream.decoder.SendPacket(&packet);
    
    // Receive decoded frame
    DecodedFrame newFrame;
    if (activeStream.decoder.ReceiveFrame(newFrame)) {
        activeStream.currentFrame = newFrame;
    }
}
```

### Trigger → Switching
**Directory:** [triggers/CLAUDE.md](triggers/CLAUDE.md) → [switching/CLAUDE.md](switching/CLAUDE.md)
```cpp
// Main application loop coordination
while (window.ProcessMessages()) {
    // 1. Update input triggers
    videoManager.UpdateSwitchingTrigger();        // ISwitchingTrigger::Update()
    
    // 2. Process any triggered switches  
    if (videoManager.ProcessSwitchingTriggers()) { // Check ShouldSwitchToVideo1/2()
        // Switch executed via VideoSwitchingStrategy::SwitchToVideo()
    }
    
    // 3. Update video frames
    videoManager.UpdateFrame();                   // VideoSwitchingStrategy::UpdateFrame()
}
```

## Advanced Features

### Multi-Resolution Support
**Change from Original Requirements:** Videos no longer need identical resolutions
```cpp
// VideoManager handles different resolutions automatically
int maxVideoWidth = std::max(video1Info.width, video2Info.width);
int maxVideoHeight = std::max(video1Info.height, video2Info.height);

// Window sized to accommodate largest video
if (!window.Create("FFmpeg Video Player", maxVideoWidth, maxVideoHeight)) {
    // Window creation with dynamic sizing
}
```

### Playback Speed Control
**New Feature:** Variable playback speed support
```cpp
bool VideoManager::Initialize(..., double playbackSpeed) {
    m_playbackSpeed = playbackSpeed;
    
    // Adjust frame timing based on speed
    double adjustedFrameInterval = m_frameInterval / m_playbackSpeed;
}

// Timing calculations account for speed
double VideoManager::GetCurrentTime() const {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - m_playbackStartTime);
    return m_pausedTime + (elapsed.count() / 1000000.0) * m_playbackSpeed;
}
```

### Frame Timing Optimization
**Performance Feature:** Efficient frame rate matching
```cpp
bool VideoManager::ShouldUpdateFrame() const {
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastFrameTime);
    double elapsedSeconds = timeSinceLastFrame.count() / 1000000.0;
    
    // Only decode when display timing requires new frame
    double adjustedFrameInterval = m_frameInterval / m_playbackSpeed;
    return elapsedSeconds >= adjustedFrameInterval;
}
```

## Error Handling and Recovery

### Stream Failure Recovery
```cpp
bool VideoManager::InitializeVideoStream(VideoStream& stream, const std::string& filePath) {
    // Cascade failure handling
    if (!stream.demuxer.Open(filePath)) {
        LOG_ERROR("Failed to open video file: ", filePath);
        return false;  // Fail fast on container issues
    }
    
    // Hardware decoder with software fallback
    DecoderInfo decoderInfo = HardwareDecoder::GetBestDecoder(stream.demuxer.GetCodecID());
    if (!stream.decoder.Initialize(codecParams, decoderInfo)) {
        LOG_ERROR("Failed to initialize decoder for: ", filePath);
        return false;  // Decoder initialization critical
    }
    
    stream.initialized = true;
    return true;
}
```

### Runtime Error Handling
```cpp
bool VideoManager::UpdateFrame() {
    if (!m_switchingStrategy->UpdateFrame()) {
        LOG_ERROR("Strategy failed to update frame");
        // Strategy handles its own recovery, critical errors bubble up
        return false;
    }
    return true;
}
```

## Performance Characteristics

### Memory Usage Patterns
- **Standard Mode:** Single active video stream (~baseline memory)
- **Predecoded Mode:** Both streams active (~2x baseline memory)
- **Frame Buffers:** Minimal buffering (current frame only)

### CPU/GPU Load Distribution
- **Container Parsing:** Minimal CPU overhead (demuxing)
- **Hardware Decoding:** GPU acceleration when available (NVDEC/D3D11VA)
- **Software Fallback:** CPU decoding with format conversion
- **Switching Overhead:** Strategy-dependent (immediate vs keyframe vs predecoded)

### Threading Model
- **Single-Threaded Design:** All video operations on main thread
- **Synchronous Processing:** Frame-by-frame processing without threading complexity
- **Integration-Friendly:** Clean integration with Win32 message loop

## Command Line Integration

### Video System Parameters
```bash
# Algorithm selection
./ffmpeg_player video1.mp4 video2.mp4 --algorithm immediate
./ffmpeg_player video1.mp4 video2.mp4 --algorithm predecoded  
./ffmpeg_player video1.mp4 video2.mp4 --algorithm keyframe-sync

# Trigger selection
./ffmpeg_player video1.mp4 video2.mp4 --trigger keyboard

# Playback speed control  
./ffmpeg_player video1.mp4 video2.mp4 --speed 0.5  # Half speed
./ffmpeg_player video1.mp4 video2.mp4 --speed 2.0  # Double speed
```

## Subsystem Documentation

For detailed implementation information, see the individual subsystem documentation:

- **[demux/CLAUDE.md](demux/CLAUDE.md)** - Container parsing and packet extraction
- **[decode/CLAUDE.md](decode/CLAUDE.md)** - Hardware-accelerated decoding system  
- **[switching/CLAUDE.md](switching/CLAUDE.md)** - Video switching strategies and algorithms
- **[triggers/CLAUDE.md](triggers/CLAUDE.md)** - Input handling and switching triggers

Each subsystem is designed for independence and testability, with clear interfaces enabling easy extension and modification of video processing behavior.