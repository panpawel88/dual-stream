# Camera Frame Processing System

This directory implements the frame processing and delivery infrastructure for the camera system, providing per-listener thread pools and computer vision integration with complete isolation between listeners.

## Architecture Overview

The frame processing system uses a per-listener processor pattern where each listener gets its own dedicated thread and circular buffer queue, ensuring complete isolation and parallel processing:

```
src/camera/processing/
├── ICameraFrameListener.h/cpp       # Interface for frame consumers
├── CircularBuffer.h                 # Thread-safe circular buffer template
├── ListenerProcessor.h/cpp          # Per-listener processing engine
├── CameraFramePublisher.h/cpp       # Per-listener frame distribution
├── FaceDetectionSwitchingTrigger.h/cpp # Face detection integration
└── CLAUDE.md                        # This documentation
```

## Core Components

### ICameraFrameListener Interface
**File:** `ICameraFrameListener.h/cpp`
**Purpose:** Standard interface for all frame processing components with queue configuration preferences

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

    // Queue configuration preferences (new)
    virtual bool HasCustomQueueConfig() const { return false; }
    virtual size_t GetPreferredQueueSize() const { return 0; }
    virtual OverflowPolicy GetPreferredOverflowPolicy() const;
    virtual double GetPreferredMaxFrameAgeMs() const { return 100.0; }
};
```

**Priority System:**
```cpp
enum class ListenerPriority {
    LOW = 0,        // Background processing, non-critical
    NORMAL = 10,    // Standard processing priority
    HIGH = 20,      // Time-sensitive processing
    CRITICAL = 30   // Real-time processing (e.g., switching triggers)
};
```

### CircularBuffer Template
**File:** `CircularBuffer.h`
**Purpose:** Thread-safe circular buffer for per-listener frame queuing

**Key Features:**
- **Generic Template:** Works with any data type
- **Overflow Policies:** DROP_OLDEST, DROP_NEWEST, BLOCK
- **Non-blocking Operations:** TryPush/TryPop for zero-latency operations
- **Blocking Operations:** Push/Pop with optional timeouts
- **Statistics Tracking:** Comprehensive performance monitoring
- **Shutdown Support:** Clean thread termination

**Overflow Policies:**
```cpp
enum class OverflowPolicy {
    DROP_OLDEST,    // Drop oldest item to make room (default)
    DROP_NEWEST,    // Drop the new item being added
    BLOCK           // Block until space is available
};
```

### ListenerProcessor
**File:** `ListenerProcessor.h/cpp`
**Purpose:** Individual processor for each camera frame listener with dedicated thread and queue

**Key Features:**
- **Dedicated Thread:** Each listener gets its own processing thread
- **Individual Queue:** Circular buffer for frame buffering per listener
- **Independent Configuration:** Per-listener queue size, overflow policy, frame age limits
- **Performance Statistics:** Detailed monitoring per processor
- **Thread Naming:** Named threads for debugging support
- **Graceful Error Handling:** Automatic listener disabling on critical errors

**Architecture:**
```cpp
class ListenerProcessor {
public:
    explicit ListenerProcessor(CameraFrameListenerPtr listener,
                              const ListenerProcessorConfig& config);

    bool Start();
    void Stop();
    bool EnqueueFrame(const CameraFrame& frame);

    ListenerProcessorStats GetStats() const;
    size_t GetQueueSize() const;
    void ClearQueue();

private:
    std::unique_ptr<std::thread> m_processorThread;
    std::unique_ptr<CircularBuffer<TimestampedFrame>> m_frameQueue;
    CameraFrameListenerPtr m_listener;
};
```

### CameraFramePublisher (New Architecture)
**File:** `CameraFramePublisher.h/cpp`
**Purpose:** Per-listener frame distribution with complete isolation

**Key Features:**
- **Per-Listener Isolation:** Each listener processes frames independently
- **Zero Blocking:** Slow listeners cannot affect others
- **Parallel Processing:** True multi-threaded frame processing
- **Individual Configuration:** Per-listener queue settings
- **Enhanced Monitoring:** Detailed per-listener statistics

**Publisher Architecture:**
```cpp
class CameraFramePublisher {
public:
    bool PublishFrame(const CameraFrame& frame);
    bool RegisterListener(CameraFrameListenerPtr listener);
    bool UnregisterListener(const std::string& listenerId);

    // Per-listener monitoring
    std::optional<ListenerProcessorStats> GetListenerStats(const std::string& listenerId) const;
    std::unordered_map<std::string, ListenerProcessorStats> GetAllListenerStats() const;
    size_t GetListenerQueueSize(const std::string& listenerId) const;
    bool ClearListenerQueue(const std::string& listenerId);

private:
    std::unordered_map<std::string, std::unique_ptr<ListenerProcessor>> m_processors;
    std::vector<CameraFrameListenerPtr> m_listeners;
};
```

**New Threading Model:**
```
Camera Capture Thread → CameraFramePublisher::PublishFrame()
                              ↓
                    Independent Frame Distribution
                              ↓
    ┌─────────────────┬─────────────────┬─────────────────┐
    ▼                 ▼                 ▼                 ▼
Thread A          Thread B          Thread C          Thread D
Queue A           Queue B           Queue C           Queue D
Listener A        Listener B        Listener C        Listener D
    │                 │                 │                 │
    └─────────────────┴─────────────────┴─────────────────┘
                    Complete Isolation
                (No blocking between listeners)
```

## Performance Characteristics

### Architecture Comparison
**Before (Shared Thread Pool):**
```
Frame → Worker Thread Pool → Sequential Processing
frame 1 → thread 1 → [listenerA, listenerB, listenerC] (sequential)
frame 2 → thread 2 → [listenerA, listenerB, listenerC] (sequential)
Problem: Slow listenerB blocks listenerA and listenerC
```

**After (Per-Listener Threads):**
```
Frame → Independent Processors → Parallel Processing
frame 1 → [threadA→listenerA, threadB→listenerB, threadC→listenerC] (parallel)
frame 2 → [threadA→listenerA, threadB→listenerB, threadC→listenerC] (parallel)
Benefit: Slow listenerB doesn't affect listenerA or listenerC
```

### Threading Performance
```
Configuration Options (Per Listener):
├── Queue Size: 1-20 frames (default: 3)
├── Overflow Policy: DROP_OLDEST/DROP_NEWEST/BLOCK
├── Frame Age Limit: 10-1000ms (default: 100ms)
└── Thread Priority: Configurable per listener

Performance Impact:
├── Face Detection: ~10-30% CPU per frame (640x480)
├── Frame Distribution: ~1% CPU per listener (parallel)
├── Queue Management: ~0.5% CPU per listener
└── Total System Impact: Scales linearly with listener count
```

### Memory Usage
```
Base Components:
├── CameraFramePublisher: ~100KB
├── Per ListenerProcessor: ~50KB each
├── Per Circular Buffer: ~Queue_Size * Frame_Size
├── Face Detection Cascades: ~5MB (loaded once)
└── Thread Stacks: ~1MB per listener thread

Frame Processing:
├── Frame Reference Counting: Minimal overhead
├── Per-Listener Queues: Configurable (default: 3 frames each)
├── Detection Buffers: ~2MB (temporary OpenCV processing)
└── Total Active Memory: ~10MB + (Listeners * Queue_Memory)
```

## Configuration System

### Publisher Configuration (Updated)
```ini
[camera.processing]
# Global defaults for new listeners
max_frame_queue_size = 3                    # Default queue size
max_frame_age_ms = 100.0                   # Default frame age limit
enable_frame_skipping = true               # Default age checking
use_listener_preferences = true            # Use listener-specific preferences
enable_performance_logging = false         # Performance logging
```

**Configuration Structure (New):**
```cpp
struct PublisherConfig {
    // Default listener processor configuration
    ListenerProcessorConfig defaultListenerConfig;

    // Global publisher settings
    bool useListenerPreferences = true;    // Use listener-specific preferences
    bool enablePerformanceLogging = false; // Log performance statistics

    // Backward compatibility (deprecated)
    size_t maxFrameQueueSize = 3;
    double maxFrameAgeMs = 100.0;
    bool enableFrameSkipping = true;
};

struct ListenerProcessorConfig {
    size_t queueSize = 3;                              // Circular buffer size
    OverflowPolicy overflowPolicy = OverflowPolicy::DROP_OLDEST;
    double maxFrameAgeMs = 100.0;                      // Frame age limit
    bool enableFrameAgeCheck = true;                   // Enable age checking
    bool enableStatistics = true;                      // Enable statistics
    std::string threadName = "";                       // Thread name (auto-generated)
};
```

### Face Detection Configuration
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

## Computer Vision Integration

### FaceDetectionSwitchingTrigger
**File:** `FaceDetectionSwitchingTrigger.h/cpp`
**Purpose:** Face detection-based video switching with optimized queue configuration

**Enhanced Implementation:**
```cpp
class FaceDetectionSwitchingTrigger :
    public ICameraFrameListener,      // Processes camera frames
    public ISwitchingTrigger {        // Controls video switching

public:
    // Camera frame processing
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override;
    ListenerPriority GetPriority() const override { return ListenerPriority::CRITICAL; }

    // Queue preferences (optimized for real-time processing)
    size_t GetPreferredQueueSize() const override { return 2; }  // Small queue for low latency
    OverflowPolicy GetPreferredOverflowPolicy() const override { return OverflowPolicy::DROP_OLDEST; }
    double GetPreferredMaxFrameAgeMs() const override { return 50.0; }  // Aggressive frame dropping

    // Video switching control
    bool ShouldSwitchToVideo1() const override;
    bool ShouldSwitchToVideo2() const override;
};
```

**Switching Behavior:**
- **0 Faces:** No switching action
- **1 Face:** Switch to Video 1 (single person mode)
- **2+ Faces:** Switch to Video 2 (multiple person mode)
- **Stability:** Requires consistent detection across multiple frames
- **Cooldown:** 2-second minimum between switches to prevent rapid switching
- **Optimized Queue:** Small queue size (2 frames) for minimal latency

## Error Handling and Recovery

### Listener Isolation
**Complete Isolation:** Each listener operates independently
- **Exception Isolation:** Listener exceptions don't affect other listeners
- **Automatic Recovery:** Failed listeners are disabled individually
- **Independent Queues:** Queue overflow in one listener doesn't affect others
- **Separate Statistics:** Per-listener monitoring and debugging

### Performance Protection
**Per-Listener Resource Management:**
- **Individual Queue Limits:** Prevents memory buildup per listener
- **Independent Frame Dropping:** Each listener drops frames based on its own age limits
- **Separate Thread Management:** Failed threads don't affect other processors
- **Graceful Degradation:** System continues with remaining functional listeners

### Resource Management
**Enhanced RAII Patterns:**
- **Per-Listener Cleanup:** Individual processor cleanup on failure
- **Thread Lifecycle Management:** Automatic thread creation/destruction per listener
- **Circular Buffer Management:** Memory-efficient reference counting
- **Configuration Hot-Swapping:** Safe runtime configuration updates per listener

## Integration Points

### With Camera System
```cpp
// Camera manager integration (same interface)
CameraManager cameraManager;
auto framePublisher = cameraManager.GetFramePublisher();

// Register face detection trigger
auto faceDetector = std::make_shared<FaceDetectionSwitchingTrigger>();
faceDetector->InitializeFaceDetection();
framePublisher->RegisterListener(faceDetector);

// Each listener now gets its own processor automatically
```

### With Video Switching System
```cpp
// Video manager integration (unchanged)
VideoManager videoManager;
videoManager.SetSwitchingTrigger(std::move(faceDetector));

// Face detection now runs on dedicated thread with optimized queue
```

### Custom Listener Implementation
```cpp
class CustomVisionProcessor : public ICameraFrameListener {
public:
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override {
        // Your computer vision processing here
        cv::Mat& image = frame.cpu.mat;

        // Heavy processing doesn't block other listeners
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

    // Custom queue configuration for heavy processing
    size_t GetPreferredQueueSize() const override { return 5; }  // Larger buffer
    OverflowPolicy GetPreferredOverflowPolicy() const override {
        return OverflowPolicy::BLOCK;  // Don't drop frames for accuracy
    }
    double GetPreferredMaxFrameAgeMs() const override { return 500.0; }  // Allow older frames
};
```

## Monitoring and Statistics

### Per-Listener Statistics
```cpp
struct ListenerProcessorStats {
    uint64_t framesEnqueued;           // Frames added to this listener's queue
    uint64_t framesProcessed;          // Successfully processed frames
    uint64_t framesDroppedQueue;       // Dropped due to full queue
    uint64_t framesDroppedAge;         // Dropped due to age limit
    uint64_t framesSkipped;            // Intentionally skipped frames
    uint64_t framesFailed;             // Processing failures

    double averageProcessingTimeMs;     // Average processing time
    double averageQueueDepth;          // Average queue utilization
    double maxQueueDepth;              // Peak queue utilization

    double GetProcessingRate() const;   // Frames per second
    double GetSuccessRate() const;      // Success percentage
};

// Usage
auto stats = publisher.GetListenerStats("face_detection_trigger");
if (stats) {
    LOG_INFO("Face detector: {}fps, {:.1f}% success rate",
             stats->GetProcessingRate(), stats->GetSuccessRate() * 100);
}
```

### Global Publisher Statistics
```cpp
struct PublisherStats {
    uint64_t framesPublished;          // Total frames published
    uint64_t totalFrameEnqueues;       // Total enqueue operations
    uint64_t successfulEnqueues;       // Successful enqueues
    uint64_t failedEnqueues;           // Failed enqueues
    int activeListeners;               // Number of registered listeners
    int enabledListeners;              // Number of enabled listeners
    int runningProcessors;             // Number of running processors
};
```

## Benefits of New Architecture

### Complete Isolation
- **Zero Blocking:** Slow face detection doesn't affect motion tracking
- **Independent Failures:** One listener crash doesn't affect others
- **Separate Configuration:** Each listener can have optimal settings
- **Individual Monitoring:** Per-listener performance tracking

### Better Performance
- **True Parallelism:** All listeners process frames simultaneously
- **Optimal Resource Usage:** Each listener uses exactly what it needs
- **Reduced Contention:** No shared queues or worker threads
- **Scalable Design:** Performance scales linearly with listener count

### Enhanced Reliability
- **Fault Isolation:** Problems are contained to individual listeners
- **Graceful Degradation:** System continues with partial functionality
- **Hot-Swappable:** Listeners can be added/removed at runtime
- **Self-Healing:** Failed listeners are automatically disabled

This enhanced frame processing system provides complete isolation between listeners while maintaining excellent performance characteristics and comprehensive error handling. The per-listener thread pool architecture ensures that slow or failing listeners cannot impact the performance of other computer vision components.