# Camera Property Control Usage Guide

This document demonstrates how to use the new runtime camera property control system.

## Overview

The camera property control system allows you to adjust camera properties (brightness, contrast, exposure, saturation, gain) at runtime without restarting the camera capture. The system is thread-safe and designed for use with UI controls or automated systems.

## Key Features

- **Runtime Control**: Change properties while camera is capturing
- **Thread Safety**: Safe to call from any thread
- **Non-blocking**: Property changes are queued and applied asynchronously
- **Validation**: Automatic range checking (0-100 for all properties)
- **Error Handling**: Comprehensive error reporting

## Basic Usage

### 1. Initialize Camera System

```cpp
#include "src/camera/CameraManager.h"

// Create and initialize camera manager
CameraManager cameraManager;

// Auto-initialize with best available camera
if (!cameraManager.InitializeAuto()) {
    std::cerr << "Failed to initialize camera: " << cameraManager.GetLastError() << std::endl;
    return false;
}

// Start capture
if (!cameraManager.StartCapture()) {
    std::cerr << "Failed to start capture: " << cameraManager.GetLastError() << std::endl;
    return false;
}
```

### 2. Set Individual Properties

```cpp
// Set brightness to 75% (valid range: 0-100)
if (cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, 75)) {
    std::cout << "Brightness set successfully" << std::endl;
} else {
    std::cerr << "Failed to set brightness: " << cameraManager.GetLastError() << std::endl;
}

// Set contrast to 60%
cameraManager.SetCameraProperty(CameraPropertyType::CONTRAST, 60);

// Set exposure to 40%
cameraManager.SetCameraProperty(CameraPropertyType::EXPOSURE, 40);
```

### 3. Get Current Property Values

```cpp
int brightness, contrast, exposure;

// Get individual properties
if (cameraManager.GetCameraProperty(CameraPropertyType::BRIGHTNESS, brightness)) {
    std::cout << "Current brightness: " << brightness << std::endl;
}

if (cameraManager.GetCameraProperty(CameraPropertyType::CONTRAST, contrast)) {
    std::cout << "Current contrast: " << contrast << std::endl;
}

// Get all properties at once
CameraProperties allProps = cameraManager.GetAllCameraProperties();
std::cout << "All properties - Brightness: " << allProps.brightness
          << ", Contrast: " << allProps.contrast
          << ", Exposure: " << allProps.exposure << std::endl;
```

### 4. Batch Property Updates

```cpp
// Create properties structure
CameraProperties batchProps;
batchProps.brightness = 50;
batchProps.contrast = 50;
batchProps.exposure = 45;
// Leave other properties unchanged (they remain -1)

// Apply all changes atomically
if (cameraManager.SetCameraProperties(batchProps)) {
    std::cout << "Batch properties applied successfully" << std::endl;
}
```

### 5. Query Property Ranges

```cpp
// Get supported range for a property
CameraPropertyRange brightnessRange = cameraManager.GetPropertyRange(CameraPropertyType::BRIGHTNESS);

if (brightnessRange.supported) {
    std::cout << "Brightness range: " << brightnessRange.min << "-" << brightnessRange.max
              << " (default: " << brightnessRange.defaultValue
              << ", step: " << brightnessRange.step << ")" << std::endl;
} else {
    std::cout << "Brightness control not supported by this camera" << std::endl;
}
```

## Property Types

The following camera properties are available:

```cpp
enum class CameraPropertyType {
    BRIGHTNESS,    // Camera brightness (0-100)
    CONTRAST,      // Camera contrast (0-100)
    EXPOSURE,      // Camera exposure (0-100)
    SATURATION,    // Camera saturation (0-100)
    GAIN           // Camera gain (0-100)
};
```

## Thread Safety

All camera property operations are thread-safe and can be called from:
- UI event handlers
- Background processing threads
- Timer callbacks
- Network request handlers

```cpp
// Example: UI slider callback
void OnBrightnessSliderChanged(int value) {
    // This is safe to call from UI thread
    cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, value);
}

// Example: Automatic adjustment in background thread
void AutoAdjustExposure() {
    std::thread([&cameraManager]() {
        while (adjusting) {
            // Calculate optimal exposure...
            int newExposure = CalculateOptimalExposure();

            // Safe to call from background thread
            cameraManager.SetCameraProperty(CameraPropertyType::EXPOSURE, newExposure);

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
}
```

## Error Handling

```cpp
// Always check return values for important operations
if (!cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, userValue)) {
    std::string error = cameraManager.GetLastError();
    LOG_ERROR("Failed to set brightness: " << error);

    // Show error to user or fallback to default
    ShowUserError("Camera brightness adjustment failed: " + error);
}

// Validation is automatic - invalid values are rejected
bool success = cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, 150); // Invalid
// success will be false, error will be "Invalid property value: 150"
```

## Performance Considerations

- **Non-blocking**: Property changes return immediately and are applied asynchronously
- **Efficient**: Changes are batched and applied between frame captures
- **Minimal overhead**: Uses atomic flags for lightweight synchronization
- **No frame drops**: Property changes don't interrupt video capture

```cpp
// These calls return immediately - no blocking
cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, 50);
cameraManager.SetCameraProperty(CameraPropertyType::CONTRAST, 60);
cameraManager.SetCameraProperty(CameraPropertyType::EXPOSURE, 40);

// Properties are applied on next frame capture cycle
// Camera continues capturing without interruption
```

## Integration Example: UI Controls

```cpp
class CameraControlDialog {
private:
    CameraManager& m_cameraManager;

public:
    CameraControlDialog(CameraManager& manager) : m_cameraManager(manager) {}

    void InitializeControls() {
        // Get current values
        CameraProperties props = m_cameraManager.GetAllCameraProperties();

        // Set slider values
        brightnessSlider.SetValue(props.brightness >= 0 ? props.brightness : 50);
        contrastSlider.SetValue(props.contrast >= 0 ? props.contrast : 50);
        exposureSlider.SetValue(props.exposure >= 0 ? props.exposure : 50);

        // Get ranges for slider limits
        auto brightnessRange = m_cameraManager.GetPropertyRange(CameraPropertyType::BRIGHTNESS);
        brightnessSlider.SetRange(brightnessRange.min, brightnessRange.max);
        brightnessSlider.SetEnabled(brightnessRange.supported);
    }

    void OnBrightnessChanged(int value) {
        m_cameraManager.SetCameraProperty(CameraPropertyType::BRIGHTNESS, value);
    }

    void OnContrastChanged(int value) {
        m_cameraManager.SetCameraProperty(CameraPropertyType::CONTRAST, value);
    }

    void OnResetToDefaults() {
        // Reset all properties to defaults
        CameraProperties defaults;
        defaults.brightness = 50;
        defaults.contrast = 50;
        defaults.exposure = 50;
        defaults.saturation = 50;
        defaults.gain = 50;

        m_cameraManager.SetCameraProperties(defaults);

        // Update UI
        InitializeControls();
    }
};
```

## Implementation Details

- **Architecture**: Properties are queued in the capture thread and applied between frames
- **Memory**: Minimal overhead - single pending properties structure per camera
- **Latency**: Properties typically applied within 1-2 frame periods (33-66ms at 30fps)
- **Compatibility**: Works with any OpenCV-compatible camera
- **Extensibility**: Easy to add new property types by extending the enum and implementation

This system provides a robust, performant foundation for runtime camera control in any application requiring dynamic camera adjustment.