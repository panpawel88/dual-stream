# Camera Source System

This directory implements the camera source abstraction layer, providing unified access to different camera hardware through a common interface.

## Architecture Overview

The camera source system uses a factory pattern to create camera sources from different backends:

```
src/camera/sources/
├── ICameraSource.h              # Abstract camera interface
├── CameraSourceFactory.h/cpp    # Factory for camera source creation
├── OpenCVCameraSource.h/cpp     # OpenCV VideoCapture implementation
├── RealSenseCameraSource.h/cpp  # Intel RealSense implementation
└── CLAUDE.md                    # This documentation
```

## Core Abstraction

### ICameraSource Interface
**File:** `ICameraSource.h`
**Purpose:** Unified interface for all camera hardware types

**Key Interface:**
```cpp
class ICameraSource {
public:
    virtual ~ICameraSource() = default;
    
    // Device management
    virtual bool Initialize(const CameraDeviceInfo& deviceInfo, const CameraConfig& config) = 0;
    virtual void Shutdown() = 0;
    virtual bool IsInitialized() const = 0;
    
    // Capture control
    virtual bool StartCapture() = 0;
    virtual bool StopCapture() = 0;
    virtual bool IsCapturing() const = 0;
    
    // Frame access
    virtual bool CaptureFrame(CameraFrame& frame) = 0;  # Synchronous capture
    virtual void SetFrameCallback(FrameCallback callback) = 0;  # Asynchronous capture
    
    // Device information
    virtual CameraSourceType GetSourceType() const = 0;
    virtual std::string GetDeviceId() const = 0;
    virtual CameraCapabilities GetCapabilities() const = 0;
    
    // Runtime property control (normalized values 0.0-1.0)
    virtual bool SetCameraProperty(CameraPropertyType property, double value) = 0;
    virtual bool GetCameraProperty(CameraPropertyType property, double& value) const = 0;
    virtual bool SetCameraProperties(const CameraProperties& properties) = 0;
    virtual CameraProperties GetCameraProperties() const = 0;
    virtual std::set<CameraPropertyType> GetSupportedProperties() const = 0;
};
```

**Camera Source Types:**
```cpp
enum class CameraSourceType {
    OpenCV,      # OpenCV VideoCapture (webcams, USB cameras)
    RealSense,   # Intel RealSense depth cameras
    DirectShow,  # Future: Direct DirectShow integration
    V4L2         # Future: Linux V4L2 support
};
```

### CameraSourceFactory
**File:** `CameraSourceFactory.h/cpp`
**Purpose:** Dynamic camera source creation and device enumeration

**Factory Methods:**
```cpp
class CameraSourceFactory {
public:
    // Device enumeration
    static std::vector<CameraDeviceInfo> EnumerateAllDevices();
    static std::vector<CameraDeviceInfo> EnumerateDevicesOfType(CameraSourceType type);
    
    // Source creation
    static std::unique_ptr<ICameraSource> CreateForDevice(const CameraDeviceInfo& deviceInfo);
    static std::unique_ptr<ICameraSource> CreateBestAvailable();
    
    // Capability queries
    static bool IsSourceTypeAvailable(CameraSourceType type);
    static std::vector<CameraSourceType> GetAvailableSourceTypes();
};
```

**Device Selection Logic:**
```cpp
std::unique_ptr<ICameraSource> CreateBestAvailable() {
    // Try sources in order of preference
    std::vector<CameraSourceType> preferenceOrder = {
        CameraSourceType::RealSense,  // Prefer depth cameras
        CameraSourceType::OpenCV      // Fallback to standard cameras
    };
    
    for (auto sourceType : preferenceOrder) {
        if (IsSourceTypeAvailable(sourceType)) {
            auto devices = EnumerateDevicesOfType(sourceType);
            if (!devices.empty()) {
                return CreateForDevice(devices[0]);  // Use first available
            }
        }
    }
    return nullptr;  // No cameras available
}
```

## Camera Source Implementations

### OpenCVCameraSource
**File:** `OpenCVCameraSource.h/cpp`
**Purpose:** Standard webcam and USB camera support via OpenCV

**Key Features:**
- **Wide Compatibility:** Works with most USB cameras and webcams
- **Format Support:** BGR8, RGB8, GRAY8 color formats
- **Normalized Property Control:** Brightness, contrast, saturation, gain with 0.0-1.0 range
- **Smart Range Detection:** Automatic detection of camera capability ranges
- **Resolution Control:** Dynamic resolution changes
- **Frame Rate Control:** Configurable capture frame rates
- **Thread-Safe Property Updates:** Runtime property changes without capture interruption

**Implementation Details:**
```cpp
class OpenCVCameraSource : public ICameraSource {
public:
    bool Initialize(const CameraDeviceInfo& deviceInfo, const CameraConfig& config) override {
        m_capture = std::make_unique<cv::VideoCapture>(deviceInfo.deviceIndex);
        
        if (!m_capture->isOpened()) {
            LOG_ERROR("Failed to open OpenCV camera ", deviceInfo.deviceIndex);
            return false;
        }
        
        // Configure capture settings
        m_capture->set(cv::CAP_PROP_FRAME_WIDTH, config.width);
        m_capture->set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
        m_capture->set(cv::CAP_PROP_FPS, config.frameRate);
        
        return true;
    }
    
private:
    std::unique_ptr<cv::VideoCapture> m_capture;
    std::thread m_captureThread;
    FrameCallback m_frameCallback;
};
```

**Supported Cameras:**
- Standard USB webcams
- Laptop built-in cameras
- USB 2.0/3.0 cameras
- MJPEG and raw frame cameras
- Multiple simultaneous cameras

### RealSenseCameraSource
**File:** `RealSenseCameraSource.h/cpp`
**Purpose:** Intel RealSense depth camera integration

**Key Features:**
- **Depth + Color:** Simultaneous depth and color stream capture
- **Hardware Acceleration:** RealSense SDK optimized processing
- **Depth Formats:** 16-bit depth data with millimeter precision
- **Alignment:** Hardware depth-to-color alignment
- **Advanced Features:** Motion detection, background subtraction
- **Experimental Property Support:** Auto-exposure mean intensity control with normalized values

**Implementation Details:**
```cpp
class RealSenseCameraSource : public ICameraSource {
public:
    bool Initialize(const CameraDeviceInfo& deviceInfo, const CameraConfig& config) override {
        // Configure RealSense pipeline
        rs2::config rs_config;
        rs_config.enable_stream(RS2_STREAM_COLOR, config.width, config.height, 
                               RS2_FORMAT_BGR8, static_cast<int>(config.frameRate));
        
        if (config.enableDepth) {
            rs_config.enable_stream(RS2_STREAM_DEPTH, config.width, config.height, 
                                   RS2_FORMAT_Z16, static_cast<int>(config.frameRate));
        }
        
        try {
            m_pipeline.start(rs_config);
            return true;
        } catch (const rs2::error& e) {
            LOG_ERROR("RealSense initialization failed: ", e.what());
            return false;
        }
    }
    
private:
    rs2::pipeline m_pipeline;
    rs2::align m_align_to_color{RS2_STREAM_COLOR};  # Depth-color alignment
    std::thread m_captureThread;
};
```

**RealSense Benefits:**
- **Depth Information:** 3D spatial understanding
- **Background Removal:** Hardware background subtraction  
- **Face Analysis:** 3D face detection and tracking
- **Hand Tracking:** Gesture recognition capabilities
- **Multiple Models:** Support for D400, D500 series cameras

## Device Configuration System

### CameraConfig Structure
**Comprehensive Configuration:** All camera parameters in single structure
```cpp
struct CameraConfig {
    // Basic capture settings
    int width = 640;
    int height = 480;
    double frameRate = 30.0;
    CameraFormat format = CameraFormat::BGR8;
    
    // RealSense specific
    bool enableDepth = false;
    bool enableInfrared = false;
    bool alignDepthToColor = true;
    
    // Camera properties (normalized values)
    double brightness = -1.0;   # -1.0 = auto, 0.0-1.0 normalized range
    double contrast = -1.0;     # -1.0 = auto, 0.0-1.0 normalized range

    // Advanced settings
    bool enableAutoFocus = true;
    bool enableAutoWhiteBalance = true;
};
```

### CameraDeviceInfo Structure
**Device Identification:** Complete device information for source selection
```cpp
struct CameraDeviceInfo {
    std::string deviceId;           # Unique device identifier
    std::string displayName;        # Human-readable name
    CameraSourceType sourceType;    # Source implementation type
    int deviceIndex;                # Index for OpenCV cameras
    std::string serialNumber;       # Hardware serial (RealSense)
    
    // Capabilities
    std::vector<CameraFormat> supportedFormats;
    std::vector<Resolution> supportedResolutions;
    double maxFrameRate;
    bool supportsDepth;
    bool supportsInfrared;
};
```

## Normalized Property Control System

### CameraPropertyType Enumeration
**Property Types:** All supported camera properties with normalized control
```cpp
enum class CameraPropertyType {
    BRIGHTNESS,         // Camera brightness (0.0-1.0)
    CONTRAST,           // Camera contrast (0.0-1.0)
    SATURATION,         // Camera saturation (0.0-1.0)
    GAIN,               // Camera gain (0.0-1.0)
    AE_MEAN_INTENSITY_SETPOINT  // RealSense auto-exposure target (0.0-1.0)
};
```

### CameraProperties Structure
**Batch Property Updates:** Multiple properties in single operation
```cpp
struct CameraProperties {
    double brightness = std::numeric_limits<double>::quiet_NaN();  // NaN means unchanged
    double contrast = std::numeric_limits<double>::quiet_NaN();
    double saturation = std::numeric_limits<double>::quiet_NaN();
    double gain = std::numeric_limits<double>::quiet_NaN();
    double ae_mean_intensity_setpoint = std::numeric_limits<double>::quiet_NaN();

    bool HasChanges() const;  // Check if any property is set
    void Reset();             // Reset all properties to unchanged state
};
```

### Property Implementation Details
**OpenCV Property Mapping:**
```cpp
// OpenCV properties use hardware-specific ranges, normalized to 0.0-1.0
bool OpenCVCameraSource::SetCameraProperty(CameraPropertyType property, double value) {
    // Input value is normalized 0.0-1.0, convert to camera's native range
    double min, max, current;
    int cvProperty = GetOpenCVProperty(property);

    // Get camera's native range
    GetPropertyRange(cvProperty, min, max);

    // Convert normalized value to native range
    double nativeValue = min + (value * (max - min));

    return m_capture->set(cvProperty, nativeValue);
}
```

**Smart Range Detection:**
```cpp
// Intelligent detection of camera property ranges
void DetectPropertyRanges() {
    for (auto property : {CV_CAP_PROP_BRIGHTNESS, CV_CAP_PROP_CONTRAST, CV_CAP_PROP_SATURATION}) {
        double originalValue = m_capture->get(property);

        // Test range by setting extreme values
        m_capture->set(property, 0.0);
        double minValue = m_capture->get(property);

        m_capture->set(property, 1.0);
        double maxValue = m_capture->get(property);

        // Restore original value
        m_capture->set(property, originalValue);

        // Store detected range
        m_propertyRanges[property] = {minValue, maxValue};
    }
}
```

### Property Support Detection
**Runtime Capability Query:**
```cpp
std::set<CameraPropertyType> OpenCVCameraSource::GetSupportedProperties() const {
    std::set<CameraPropertyType> supported;

    // Test each property by attempting to set/get values
    if (TestPropertySupport(CV_CAP_PROP_BRIGHTNESS)) {
        supported.insert(CameraPropertyType::BRIGHTNESS);
    }
    if (TestPropertySupport(CV_CAP_PROP_CONTRAST)) {
        supported.insert(CameraPropertyType::CONTRAST);
    }
    // ... test other properties

    return supported;
}
```

## Frame Processing Integration

### Asynchronous Capture
**Background Thread Processing:** Non-blocking frame capture
```cpp
void OpenCVCameraSource::CaptureThreadFunction() {
    cv::Mat frame;
    
    while (m_isCapturing) {
        if (m_capture->read(frame)) {
            // Convert to CameraFrame format
            CameraFrame cameraFrame = CameraFrame::CreateFromMat(frame, m_config.format);
            
            // Deliver to callback (CameraFramePublisher)
            if (m_frameCallback) {
                m_frameCallback(cameraFrame);
            }
        } else {
            LOG_WARNING("Failed to capture frame from camera");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
```

### Synchronous Capture
**Direct Frame Access:** For applications requiring immediate frame access
```cpp
bool CaptureFrame(CameraFrame& frame) override {
    cv::Mat mat;
    if (m_capture->read(mat)) {
        frame = CameraFrame::CreateFromMat(mat, m_config.format);
        return true;
    }
    return false;
}
```

## Error Handling and Recovery

### Initialization Failures
**Graceful Degradation:** System continues with available cameras
- **Device Enumeration Errors:** Skip failed devices, continue with others
- **Initialization Failures:** Detailed error logging and fallback options
- **Configuration Errors:** Use default settings when specific settings fail

### Runtime Error Recovery
**Robust Operation:** System recovers from temporary failures
- **Frame Capture Failures:** Retry logic with backoff
- **Device Disconnection:** Automatic detection and recovery attempts
- **Resource Exhaustion:** Proper cleanup and resource management

### Multi-Camera Support
**Resource Management:** Efficient handling of multiple simultaneous cameras
- **Resource Sharing:** Proper USB bandwidth management
- **Thread Management:** One capture thread per camera source
- **Memory Management:** Efficient frame sharing between cameras

## Performance Characteristics

### OpenCV Performance
```
Typical Performance (640x480 @ 30fps):
├── CPU Usage: ~5-15% per camera
├── Memory Usage: ~50MB per camera
├── USB Bandwidth: ~25MB/s (uncompressed)
├── Latency: 33-50ms (USB + processing)
└── Simultaneous Cameras: Up to 4 (USB bandwidth limited)
```

### RealSense Performance  
```
Typical Performance (640x480 @ 30fps, Color + Depth):
├── CPU Usage: ~10-20% per camera
├── Memory Usage: ~100MB per camera (color + depth)
├── USB Bandwidth: ~50MB/s (color + depth)
├── Latency: 16-33ms (hardware optimized)
└── Simultaneous Cameras: Up to 2 (USB 3.0 required)
```

## Integration Example

### Basic Camera Setup
```cpp
// Enumerate available cameras
auto devices = CameraSourceFactory::EnumerateAllDevices();
LOG_INFO("Found ", devices.size(), " camera devices");

// Create camera source for best available device
auto cameraSource = CameraSourceFactory::CreateBestAvailable();
if (!cameraSource) {
    LOG_ERROR("No cameras available");
    return false;
}

// Configure camera with normalized properties
CameraConfig config;
config.width = 640;
config.height = 480;
config.frameRate = 30.0;
config.format = CameraFormat::BGR8;
config.brightness = 0.7;  // 70% brightness (normalized)
config.contrast = 0.5;    // 50% contrast (normalized)

if (!cameraSource->Initialize(devices[0], config)) {
    LOG_ERROR("Failed to initialize camera");
    return false;
}

// Set up frame callback
cameraSource->SetFrameCallback([](const CameraFrame& frame) {
    // Process frame (delivered to CameraFramePublisher)
    CameraFramePublisher::GetInstance().PublishFrame(frame);
});

// Start capture
cameraSource->StartCapture();

// Runtime property adjustment (new feature)
cameraSource->SetCameraProperty(CameraPropertyType::BRIGHTNESS, 0.8); // 80% brightness
cameraSource->SetCameraProperty(CameraPropertyType::SATURATION, 0.6);  // 60% saturation

// Batch property updates
CameraProperties props;
props.brightness = 0.75;
props.contrast = 0.4;
props.saturation = 0.8;
cameraSource->SetCameraProperties(props);

// Query supported properties
auto supported = cameraSource->GetSupportedProperties();
for (auto prop : supported) {
    double value;
    if (cameraSource->GetCameraProperty(prop, value)) {
        LOG_INFO("Property value: ", value);
    }
}
```

### Multi-Camera Setup
```cpp
// Create multiple camera sources
std::vector<std::unique_ptr<ICameraSource>> cameras;
auto devices = CameraSourceFactory::EnumerateAllDevices();

for (const auto& device : devices) {
    auto camera = CameraSourceFactory::CreateForDevice(device);
    if (camera && camera->Initialize(device, config)) {
        cameras.push_back(std::move(camera));
    }
}

LOG_INFO("Initialized ", cameras.size(), " cameras");
```

This camera source system provides a flexible, extensible foundation for camera hardware integration with comprehensive error handling, multi-backend support, and normalized property control for consistent cross-camera operation.