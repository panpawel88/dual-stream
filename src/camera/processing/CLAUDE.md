# Camera Frame Processing System

This directory implements the frame processing and delivery infrastructure for the camera system, providing multi-threaded frame distribution and computer vision integration.

## Architecture Overview

The frame processing system uses a publisher-subscriber pattern to distribute camera frames to multiple consumers while protecting the main rendering thread:

```
src/camera/processing/
├── ICameraFrameListener.h           # Interface for frame consumers
├── CameraFramePublisher.h/cpp       # Multi-threaded frame distribution
├── FaceDetectionSwitchingTrigger.h/cpp # Face detection integration
└── CLAUDE.md                        # This documentation
```

## Core Components

### ICameraFrameListener Interface
**File:** `ICameraFrameListener.h`
**Purpose:** Standard interface for all frame processing components

**Key Interface:**
```cpp
class ICameraFrameListener {
public:
    virtual ~ICameraFrameListener() = default;
    
    virtual FrameProcessingResult ProcessFrame(const CameraFrame& frame) = 0;
    virtual ListenerPriority GetPriority() const = 0;
    virtual std::string GetListenerId() const = 0;
    virtual bool CanProcessFormat(CameraFormat format) const = 0;
    virtual bool IsEnabled() const { return true; }
};
```

**Priority System:**
```cpp
enum class ListenerPriority {
    CRITICAL = 0,    // Must process immediately (switching triggers)
    HIGH = 1,        # Time-sensitive processing
    NORMAL = 2,      # Standard processing
    LOW = 3          # Background processing
};
```

### CameraFramePublisher
**File:** `CameraFramePublisher.h/cpp`
**Purpose:** Multi-threaded frame distribution system

**Key Features:**
- **Thread Pool:** Configurable worker threads for frame processing
- **Priority Queues:** Critical listeners process first
- **Frame Aging:** Automatic dropping of old frames to prevent buildup
- **Reference Counting:** Automatic frame cleanup when all consumers finish
- **Load Balancing:** Frame distribution across available worker threads

**Publisher Architecture:**
```cpp
class CameraFramePublisher {
public:
    void RegisterListener(std::shared_ptr<ICameraFrameListener> listener);
    void UnregisterListener(const std::string& listenerId);
    void PublishFrame(const CameraFrame& frame);
    void SetMaxWorkerThreads(size_t count);
    void SetMaxFrameAge(double ageMs);

private:
    std::vector<std::shared_ptr<ICameraFrameListener>> m_listeners;
    std::vector<std::thread> m_workerThreads;
    std::queue<FrameProcessingTask> m_frameQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;
};
```

**Threading Model:**
```
Camera Capture Thread → CameraFramePublisher::PublishFrame()
                              ↓
                     Priority Queue System
                              ↓
           Worker Thread Pool (configurable size)
                              ↓
        Parallel Frame Processing (multiple listeners)
                              ↓
              Automatic Frame Reference Cleanup
```

## Computer Vision Integration

### FaceDetectionSwitchingTrigger
**File:** `FaceDetectionSwitchingTrigger.h/cpp`
**Purpose:** Face detection-based video switching integration

**Dual Interface Implementation:**
```cpp
class FaceDetectionSwitchingTrigger : 
    public ICameraFrameListener,      # Processes camera frames
    public ISwitchingTrigger {        # Controls video switching

public:
    // Camera frame processing
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override;
    ListenerPriority GetPriority() const override { return ListenerPriority::CRITICAL; }
    
    // Video switching control
    bool ShouldSwitchToVideo1() const override;
    bool ShouldSwitchToVideo2() const override;
    void Reset() override;
    
    // Face detection configuration
    bool InitializeFaceDetection();
    void SetFaceDetectionConfig(const FaceDetectionConfig& config);
};
```

**Detection Logic:**
```cpp
// Face detection processing flow
FrameProcessingResult ProcessFrame(const CameraFrame& frame) {
    // Convert to grayscale for efficient processing
    cv::Mat grayFrame;
    cv::cvtColor(frame.cpu.mat, grayFrame, cv::COLOR_BGR2GRAY);
    
    // Detect faces using Haar cascades
    std::vector<cv::Rect> faces;
    m_faceCascade.detectMultiScale(grayFrame, faces, 
        m_config.scaleFactor, m_config.minNeighbors, 0,
        cv::Size(m_config.minFaceSize, m_config.minFaceSize));
    
    // Update switching state based on face count
    UpdateSwitchingState(faces.size());
    
    return FrameProcessingResult::SUCCESS;
}
```

**Switching Behavior:**
- **0 Faces:** No switching action
- **1 Face:** Switch to Video 1 (single person mode)
- **2+ Faces:** Switch to Video 2 (multiple person mode)
- **Stability:** Requires consistent detection across multiple frames
- **Cooldown:** 2-second minimum between switches to prevent rapid switching

## Performance Characteristics

### Threading Performance
```
Configuration Options:
├── Worker Threads: 1-8 threads (default: 2)
├── Frame Queue Size: 1-20 frames (default: 5)
├── Frame Age Limit: 50-500ms (default: 100ms)
└── Priority Processing: Enabled by default

Performance Impact:
├── Face Detection: ~10-30% CPU per frame (640x480)
├── Frame Distribution: ~2% CPU (background threads)
├── Queue Management: ~1% CPU (mutex overhead)
└── Total System Impact: ~15-35% CPU at 30fps
```

### Memory Usage
```
Base Components:
├── CameraFramePublisher: ~50KB
├── Per Listener: ~10-20KB
├── Face Detection Cascades: ~5MB (loaded once)
└── Worker Thread Stacks: ~1MB per thread

Frame Processing:
├── Frame Reference Counting: Minimal overhead
├── Priority Queues: ~100KB maximum
├── Detection Buffers: ~2MB (temporary OpenCV processing)
└── Total Active Memory: ~8-12MB
```

## Configuration System

### Face Detection Configuration
**File Integration:** Integrates with existing Config system
```ini
[camera.face_detection]
enabled = true
min_face_size = 30
scale_factor = 1.1
min_neighbors = 3
stability_frames = 5
switch_cooldown_ms = 2000
multiple_face_threshold = 2
```

**Configuration Structure:**
```cpp
struct FaceDetectionConfig {
    int minFaceSize = 30;                    # Minimum face size in pixels
    double scaleFactor = 1.1;                # Detection scale factor
    int minNeighbors = 3;                    # Haar cascade neighbors
    int stabilityFrames = 5;                 # Frames required for switch
    double switchCooldownMs = 2000.0;        # Cooldown between switches
    int multipleFaceThreshold = 2;           # Face count for Video 2
};
```

### Publisher Configuration
```ini
[camera.processing]
max_worker_threads = 2
max_frame_queue_size = 5
max_frame_age_ms = 100.0
enable_frame_skipping = true
enable_priority_processing = true
```

## Error Handling and Recovery

### Listener Management
**Robust Error Handling:** System continues operation even if listeners fail
- **Exception Isolation:** Listener exceptions don't crash the publisher
- **Automatic Disabling:** Failing listeners automatically disabled
- **Error Reporting:** Comprehensive logging for debugging
- **Graceful Degradation:** System continues with remaining listeners

### Performance Protection
**Main Thread Protection:** All processing happens on background threads
- **Queue Limits:** Prevents memory buildup under load
- **Frame Dropping:** Automatic dropping of old frames
- **Load Monitoring:** Performance statistics for optimization
- **Resource Cleanup:** Automatic cleanup of failed processing tasks

### Resource Management
**RAII Patterns:** Proper resource management throughout
- **Thread Lifecycle:** Automatic thread creation and cleanup
- **Memory Management:** Reference counting prevents leaks  
- **OpenCV Resources:** Proper cleanup of computer vision resources
- **Configuration Updates:** Safe runtime configuration changes

## Integration Points

### With Camera System
```cpp
// Camera manager integration
CameraManager cameraManager;
auto framePublisher = cameraManager.GetFramePublisher();

// Register face detection trigger
auto faceDetector = std::make_shared<FaceDetectionSwitchingTrigger>();
faceDetector->InitializeFaceDetection();
framePublisher->RegisterListener(faceDetector);
```

### With Video Switching System  
```cpp
// Video manager integration
VideoManager videoManager;
videoManager.SetSwitchingTrigger(std::move(faceDetector));

// The trigger now controls video switching based on face detection
```

### Custom Listener Implementation
```cpp
class CustomVisionProcessor : public ICameraFrameListener {
public:
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override {
        // Your computer vision processing here
        cv::Mat& image = frame.cpu.mat;
        
        // Example: Object detection, motion tracking, etc.
        ProcessComputerVision(image);
        
        return FrameProcessingResult::SUCCESS;
    }
    
    ListenerPriority GetPriority() const override { 
        return ListenerPriority::NORMAL; 
    }
    
    std::string GetListenerId() const override { 
        return "custom_vision_processor"; 
    }
    
    bool CanProcessFormat(CameraFormat format) const override {
        return format == CameraFormat::BGR8 || format == CameraFormat::RGB8;
    }
};
```

This frame processing system provides a robust, thread-safe foundation for computer vision integration while maintaining excellent performance characteristics and comprehensive error handling.