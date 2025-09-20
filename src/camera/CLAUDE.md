# Camera Frame Delivery System

This directory implements a comprehensive camera frame delivery module designed for computer vision processing tasks, specifically optimized for face detection and video switching integration.

## Architecture Overview

The camera system follows the established patterns from the video player architecture:

```
src/camera/
├── CameraFrame.h                        # Frame abstraction with cv::Mat integration
├── CameraManager.h/cpp                  # Central coordinator (like VideoManager)
├── sources/                             # Camera source implementations
│   ├── ICameraSource.h                  # Abstract camera interface
│   ├── CameraSourceFactory.h/cpp        # Factory pattern for camera creation
│   ├── OpenCVCameraSource.h/cpp         # OpenCV VideoCapture implementation
│   ├── RealSenseCameraSource.h/cpp      # Intel RealSense implementation
│   └── CLAUDE.md                        # Camera source system documentation
├── processing/                          # Frame processing and delivery
│   ├── ICameraFrameListener.h           # Interface for frame consumers
│   ├── CameraFramePublisher.h/cpp       # Multi-threaded frame distribution
│   ├── FaceDetectionSwitchingTrigger.h/cpp # Example CV processing integration
│   └── CLAUDE.md                        # Frame processing system documentation
├── ui/                                  # Camera control UI components
│   ├── CameraControlUI.h/cpp            # ImGui-based camera control interface
│   ├── CameraFrameTexture.h/cpp         # Camera frame-to-texture conversion for UI
│   └── CLAUDE.md                        # Camera UI system documentation
└── CLAUDE.md                           # This documentation
```

## Core Design Principles

### 1. CPU-Focused Processing Pipeline
Unlike the video system's GPU acceleration focus, the camera system is optimized for CPU-based computer vision with UI integration:

- **OpenCV Integration:** Direct `cv::Mat` support for zero-copy processing
- **CPU Memory Management:** Efficient reference counting for multi-consumer access
- **No GPU Transfers:** Frames stay in system RAM for CV algorithms
- **UI Texture Conversion:** Camera frames converted to GPU textures for ImGui display

### 2. Publisher-Subscriber Pattern
Thread-safe frame delivery system protects the main rendering thread:

- **Background Delivery:** Frame processing runs on dedicated worker threads
- **Reference Counting:** Automatic frame disposal when all consumers finish
- **Priority Processing:** Critical listeners (like switching triggers) process first
- **Load Management:** Frame dropping and age limits prevent memory buildup

### 3. Strategy Pattern Integration
Follows the existing architecture patterns for consistency:

- **ICameraSource:** Abstract interface like IRenderer
- **CameraSourceFactory:** Factory pattern like RendererFactory
- **Multiple Implementations:** OpenCV and RealSense sources
- **IUIDrawable Integration:** Camera UI components follow existing UI patterns
- **Normalized Property System:** 0.0-1.0 range for consistent camera property control

## Component Details

### CameraFrame Abstraction
**File:** `CameraFrame.h`
**Purpose:** Generic camera frame representation optimized for computer vision

**Key Features:**
```cpp
struct CameraFrame {
    CameraFormat format;        // BGR8, RGB8, GRAY8, DEPTH16
    int width, height;
    cv::Mat cpu.mat;           // Direct OpenCV integration
    std::shared_ptr<FrameData> frameData; // Reference counting
    
    static CameraFrame CreateFromMat(const cv::Mat& mat, CameraFormat format);
    static CameraFrame CreateCPUFrame(int w, int h, CameraFormat fmt, const uint8_t* data, int pitch);
};
```

**Supported Formats:**
- **BGR8/RGB8:** Standard 8-bit color (OpenCV compatible)
- **BGRA8/RGBA8:** Alpha channel support
- **GRAY8:** Grayscale for efficient face detection
- **DEPTH16:** 16-bit depth data (RealSense)

### Camera Source System
**Base Interface:** `ICameraSource.h`
**Implementations:** OpenCV VideoCapture, Intel RealSense

**Camera Source Lifecycle:**
```cpp
// 1. Device enumeration
auto devices = CameraSourceFactory::EnumerateAllDevices();

// 2. Source creation
auto source = CameraSourceFactory::CreateForDevice(devices[0]);

// 3. Configuration and initialization
CameraConfig config{640, 480, 30.0, CameraFormat::BGR8};
source->Initialize(deviceInfo, config);

// 4. Frame capture (sync or async)
source->SetFrameCallback([](const CameraFrame& frame) {
    // Async frame delivery
});
source->StartCapture();
```

### Frame Processing System
**Publisher:** `CameraFramePublisher.h` - Multi-threaded frame distribution
**Listeners:** `ICameraFrameListener.h` - Consumer interface

### Camera Control UI System
**UI Component:** `CameraControlUI.h` - ImGui-based camera control interface
**Frame Display:** `CameraFrameTexture.h` - Camera frame-to-texture conversion for live preview

**Processing Pipeline:**
```cpp
// 1. Frame publisher receives frames from camera
publisher.PublishFrame(frame);

// 2. Background workers deliver to registered listeners
for (auto& listener : enabledListeners) {
    FrameProcessingResult result = listener->ProcessFrame(frame);
}

// 3. Reference counting ensures frame cleanup
// Frame automatically deleted when all consumers finish
```

**Performance Features:**
- **Worker Threads:** Configurable thread pool (default: 2 threads)
- **Frame Aging:** Automatic dropping of old frames (default: 100ms)
- **Priority Queues:** Critical listeners process first
- **Load Balancing:** Frame distribution across worker threads

### Integration with Video Switching

**Face Detection Trigger:** `FaceDetectionSwitchingTrigger.h`
**Integration:** Implements both `ICameraFrameListener` and `ISwitchingTrigger`

**Switching Logic:**
```cpp
class FaceDetectionSwitchingTrigger : public ISwitchingTrigger, public ICameraFrameListener {
    // Computer vision processing
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override {
        std::vector<cv::Rect> faces = DetectFaces(frame.cpu.mat);
        UpdateSwitchingState(faces.size());
    }
    
    // Video switching integration  
    bool ShouldSwitchToVideo1() const override { return m_shouldSwitchToVideo1; }
    bool ShouldSwitchToVideo2() const override { return m_shouldSwitchToVideo2; }
};
```

**Switching Behavior:**
- **1 Face Detected:** Switch to Video 1
- **2+ Faces Detected:** Switch to Video 2
- **No Faces:** No switching action
- **Stability Required:** Multiple consistent frames before switching
- **Cooldown Period:** 2 second minimum between switches

## Usage Examples

### Basic Camera Setup
```cpp
// Initialize camera manager
CameraManager cameraManager;
cameraManager.InitializeAuto(); // Use best available camera

// Start capture
cameraManager.StartCapture();

// Register face detection trigger
auto faceDetector = std::make_shared<FaceDetectionSwitchingTrigger>();
faceDetector->InitializeFaceDetection(); // Load Haar cascades
cameraManager.RegisterFrameListener(faceDetector);

// Integrate with video switching
videoManager.SetSwitchingTrigger(std::move(faceDetector));

// Register camera control UI (new feature)
auto cameraUI = std::make_shared<CameraControlUI>();
if (cameraUI->Initialize(&cameraManager, renderer.get())) {
    UIRegistry::GetInstance().RegisterDrawable(cameraUI.get());
    cameraManager.RegisterFrameListener(cameraUI); // For live preview
}
```

### Custom Frame Processing
```cpp
class CustomVisionProcessor : public ICameraFrameListener {
public:
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override {
        // Direct OpenCV processing
        cv::Mat& image = frame.cpu.mat;
        
        // Your computer vision algorithms here
        std::vector<cv::Rect> objects = DetectObjects(image);
        ProcessDetectionResults(objects);
        
        return FrameProcessingResult::SUCCESS;
    }
    
    ListenerPriority GetPriority() const override { return ListenerPriority::NORMAL; }
    std::string GetListenerId() const override { return "custom_vision_processor"; }
    bool CanProcessFormat(CameraFormat format) const override {
        return format == CameraFormat::BGR8; // Support BGR8 only
    }
};
```

## Performance Characteristics

### Memory Usage
- **Base System:** ~10MB (camera manager + publisher)
- **Per Frame:** 1-10MB depending on resolution (640x480 = ~1MB)
- **Reference Counting:** Shared frames across multiple consumers
- **Automatic Cleanup:** Frames deleted when all consumers finish

### CPU Load Distribution
- **Camera Capture:** ~5% CPU (1 thread)
- **Frame Distribution:** ~2% CPU (configurable worker threads)  
- **Face Detection:** ~10-30% CPU per frame (depends on resolution)
- **Total Impact:** ~20-40% CPU for 30fps face detection

### Threading Model
- **Camera Thread:** Dedicated capture thread per camera source
- **Worker Threads:** Configurable pool for frame processing (default: 2)
- **Main Thread Protection:** All processing happens on background threads
- **Thread-Safe:** Comprehensive mutex protection for shared data

## Configuration Options

### Camera Configuration
```cpp
struct CameraConfig {
    int width = 640, height = 480;      // Frame resolution
    double frameRate = 30.0;            // Capture frame rate
    CameraFormat format = CameraFormat::BGR8; // Frame format
    bool enableDepth = false;           // RealSense depth capture
    double brightness = -1.0;           // Camera brightness (-1.0 = auto, 0.0-1.0 normalized)
    double contrast = -1.0;             // Camera contrast (-1.0 = auto, 0.0-1.0 normalized)
};
```

### Camera Property Control (New Feature)
```cpp
// Runtime property adjustment via normalized values (0.0-1.0)
cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, 0.7); // 70% brightness
cameraManager.SetCameraProperty(CameraPropertyType::CONTRAST, 0.5);   // 50% contrast
cameraManager.SetCameraProperty(CameraPropertyType::SATURATION, 0.8); // 80% saturation

// Batch property updates
CameraProperties props;
props.brightness = 0.6;
props.contrast = 0.4;
props.saturation = 0.7;
cameraManager.SetCameraProperties(props);

// Query supported properties
std::set<CameraPropertyType> supported = cameraManager.GetSupportedProperties();
if (supported.count(CameraPropertyType::GAIN)) {
    // Camera supports gain control
}
```

### Camera UI Configuration
```ini
[camera_ui]
enable_camera_ui = true          # Enable camera UI even without face detection
preview_enabled = true           # Show live camera preview
preview_max_width = 640          # Maximum preview resolution
preview_max_height = 480
preview_fps = 10.0              # Preview refresh rate
```

### Publisher Configuration  
```cpp
struct PublisherConfig {
    size_t maxFrameQueueSize = 5;       // Frame buffer size
    size_t maxWorkerThreads = 2;        // Processing threads
    double maxFrameAgeMs = 100.0;       // Frame age limit
    bool enableFrameSkipping = true;    // Allow dropping frames
    bool enablePriorityProcessing = true; // Priority-based processing
};
```

### Face Detection Configuration
```cpp
struct FaceDetectionConfig {
    int minFaceSize = 30;               // Minimum face size (pixels)
    double scaleFactor = 1.1;           // Detection scale factor
    int minNeighbors = 3;               // Haar cascade neighbors
    int stabilityFrames = 5;            // Frames before switching
    double switchCooldownMs = 2000.0;   // Cooldown between switches
    int multipleFaceThreshold = 2;      // Faces for Video 2
};
```

## Build Integration

### CMake Configuration
The camera system integrates with the existing build system:

```cmake
# Enable camera support (ON by default)
option(ENABLE_CAMERA_SUPPORT "Enable camera capture support" ON)

# OpenCV support (required for face detection)
find_package(OpenCV QUIET COMPONENTS core imgproc objdetect videoio)

# Intel RealSense support (optional)
find_package(realsense2 QUIET)

# ImGui integration for camera UI
find_package(imgui REQUIRED)

# Conditional compilation based on available libraries
add_definitions(-DHAVE_OPENCV=1)      # If OpenCV found
add_definitions(-DHAVE_REALSENSE=1)   # If RealSense found
add_definitions(-DHAVE_CAMERA_UI=1)   # Camera UI always available with ImGui
```

### Dependency Requirements
- **OpenCV:** 4.0+ recommended (core, imgproc, objdetect, videoio modules)
- **Intel RealSense:** 2.50+ (optional, for depth cameras)
- **C++17 Standard:** Required for std::shared_ptr and threading features

## Error Handling and Recovery

### Camera Source Failures
- **Device Detection:** Graceful handling of missing cameras
- **Initialization Failures:** Detailed error messages and fallback options
- **Runtime Errors:** Automatic restart attempts and error reporting

### Frame Processing Failures
- **Listener Exceptions:** Automatic listener disabling on critical errors
- **Memory Pressure:** Frame dropping and queue size limits
- **Performance Monitoring:** Statistics tracking for optimization

### Integration Safety
- **Thread Safety:** All operations protected by appropriate mutexes
- **Resource Cleanup:** RAII patterns ensure proper cleanup
- **Graceful Degradation:** System continues operating with reduced features

## Detailed Component Documentation

For comprehensive technical information about each camera subsystem:

### Camera Source System
- **[sources/CLAUDE.md](sources/CLAUDE.md)** - Camera source abstraction, device enumeration, OpenCV and RealSense implementations, normalized property control system

### Frame Processing System
- **[processing/CLAUDE.md](processing/CLAUDE.md)** - Multi-threaded frame distribution, computer vision integration, face detection switching

### Camera UI System
- **[ui/CLAUDE.md](ui/CLAUDE.md)** - ImGui-based camera control interface, live preview system, multi-backend frame texture conversion

This camera system provides a robust foundation for computer vision integration while maintaining consistency with the existing video player architecture and protecting the main rendering thread performance.