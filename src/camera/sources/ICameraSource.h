#pragma once

#include "../CameraFrame.h"
#include <string>
#include <functional>

/**
 * Camera source type enumeration
 */
enum class CameraSourceType {
    OPENCV_WEBCAM,      // OpenCV VideoCapture with webcam
    OPENCV_VIDEO_FILE,  // OpenCV VideoCapture with video file
    REALSENSE_DEVICE,   // Intel RealSense camera device
    MOCK_CAMERA         // Mock camera for testing
};

/**
 * Camera device information structure
 */
struct CameraDeviceInfo {
    int deviceIndex;            // Device index (for OpenCV) or serial number hash
    std::string deviceName;     // Human-readable device name
    std::string serialNumber;   // Device serial number (if available)
    CameraSourceType type;      // Type of camera source
    int maxWidth, maxHeight;    // Maximum supported resolution
    double maxFrameRate;        // Maximum supported frame rate
    bool supportsDepth;         // Whether device supports depth data
    
    CameraDeviceInfo() : deviceIndex(-1), type(CameraSourceType::OPENCV_WEBCAM),
                        maxWidth(0), maxHeight(0), maxFrameRate(0.0), supportsDepth(false) {}
};

/**
 * OpenCV camera backend types for Windows-only application
 * Optimized for DirectShow and MSMF backends available on Windows
 */
enum class CameraBackend {
    AUTO,           // Smart selection: try DirectShow first, fallback to MSMF (recommended)
    PREFER_DSHOW,   // Try DirectShow first, fallback to MSMF if DirectShow fails
    PREFER_MSMF,    // Use MSMF directly (more reliable but slower initialization)
    FORCE_DSHOW,    // Only use DirectShow, fail if not available
    FORCE_MSMF,     // Only use MSMF, fail if not available
    
    // Legacy options (for compatibility)
    DEFAULT,        // Same as AUTO
    DSHOW,          // Same as FORCE_DSHOW  
    MSMF            // Same as FORCE_MSMF
};

/**
 * Camera configuration parameters
 */
struct CameraConfig {
    int width = 640;                    // Desired frame width
    int height = 480;                   // Desired frame height
    double frameRate = 30.0;            // Desired frame rate
    CameraFormat format = CameraFormat::BGR8;  // Desired frame format
    bool enableDepth = false;           // Enable depth capture (RealSense only)
    int bufferSize = 3;                 // Internal frame buffer size
    
    // OpenCV specific settings
    CameraBackend backend = CameraBackend::AUTO;  // Camera backend selection
    int brightness = -1;                // Camera brightness (-1 = auto)
    int contrast = -1;                  // Camera contrast (-1 = auto)
    int exposure = -1;                  // Camera exposure (-1 = auto)
    
    // RealSense specific settings
    bool enableAutoExposure = true;     // Enable auto exposure
    bool enableEmitter = true;          // Enable IR emitter for depth
};

/**
 * Camera source statistics for monitoring
 */
struct CameraStats {
    uint64_t framesReceived = 0;        // Total frames received
    uint64_t framesDropped = 0;         // Frames dropped due to buffer full
    double averageFrameRate = 0.0;      // Current average frame rate
    std::chrono::steady_clock::time_point lastFrameTime;  // Last successful frame timestamp
    
    void Reset() {
        framesReceived = 0;
        framesDropped = 0;
        averageFrameRate = 0.0;
        lastFrameTime = std::chrono::steady_clock::now();
    }
};

/**
 * Frame callback function type for asynchronous frame delivery
 */
using FrameCallback = std::function<void(std::shared_ptr<const CameraFrame> frame)>;

/**
 * Abstract interface for camera sources.
 * Supports both OpenCV VideoCapture and Intel RealSense cameras.
 */
class ICameraSource {
public:
    virtual ~ICameraSource() = default;
    
    /**
     * Initialize the camera source with specified configuration.
     * @param deviceInfo Device to initialize
     * @param config Camera configuration parameters
     * @return true if initialization successful
     */
    virtual bool Initialize(const CameraDeviceInfo& deviceInfo, const CameraConfig& config) = 0;
    
    /**
     * Start camera capture.
     * @return true if capture started successfully
     */
    virtual bool StartCapture() = 0;
    
    /**
     * Stop camera capture.
     */
    virtual void StopCapture() = 0;
    
    /**
     * Check if camera is currently capturing.
     * @return true if camera is capturing frames
     */
    virtual bool IsCapturing() const = 0;
    
    /**
     * Capture a single frame (synchronous).
     * @return shared_ptr to captured frame, nullptr if capture failed
     */
    virtual std::shared_ptr<CameraFrame> CaptureFrame() = 0;
    
    /**
     * Set callback for asynchronous frame delivery.
     * @param callback Function to call when new frame is available
     */
    virtual void SetFrameCallback(FrameCallback callback) = 0;
    
    /**
     * Get current camera configuration.
     * @return Current camera configuration
     */
    virtual CameraConfig GetConfig() const = 0;
    
    /**
     * Update camera configuration (if supported).
     * @param config New configuration parameters
     * @return true if configuration updated successfully
     */
    virtual bool UpdateConfig(const CameraConfig& config) = 0;
    
    /**
     * Get camera source type.
     * @return Type of this camera source
     */
    virtual CameraSourceType GetSourceType() const = 0;
    
    /**
     * Get device information.
     * @return Device information structure
     */
    virtual CameraDeviceInfo GetDeviceInfo() const = 0;
    
    /**
     * Get current capture statistics.
     * @return Statistics structure with current stats
     */
    virtual CameraStats GetStats() const = 0;
    
    /**
     * Reset capture statistics.
     */
    virtual void ResetStats() = 0;
    
    /**
     * Get last error message (if any).
     * @return Last error message or empty string
     */
    virtual std::string GetLastError() const = 0;
    
    /**
     * Enumerate available camera devices of this source type.
     * @return Vector of available camera devices
     */
    static std::vector<CameraDeviceInfo> EnumerateDevices() {
        // Default implementation returns empty list
        // Override in derived classes
        return {};
    }
    
    /**
     * Check if this camera source type is available on the system.
     * @return true if camera source is available
     */
    virtual bool IsAvailable() const = 0;
    
    /**
     * Get source name for debugging and logging.
     * @return Human-readable source name
     */
    virtual std::string GetSourceName() const = 0;
    
protected:
    CameraDeviceInfo m_deviceInfo;
    CameraConfig m_config;
    CameraStats m_stats;
    std::string m_lastError;
    bool m_isCapturing = false;
    FrameCallback m_frameCallback;
    
    /**
     * Update frame rate statistics.
     * Call this from derived classes when a frame is received.
     */
    void UpdateFrameRateStats() {
        m_stats.framesReceived++;
        
        auto now = std::chrono::steady_clock::now();
        if (m_stats.framesReceived > 1) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - m_stats.lastFrameTime);
            if (duration.count() > 0) {
                double instantFrameRate = 1000.0 / duration.count();
                // Simple moving average
                m_stats.averageFrameRate = (m_stats.averageFrameRate * 0.9) + (instantFrameRate * 0.1);
            }
        }
        m_stats.lastFrameTime = now;
    }
    
    /**
     * Record dropped frame in statistics.
     */
    void RecordDroppedFrame() {
        m_stats.framesDropped++;
    }
};