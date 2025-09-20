#include "OpenCVCameraSource.h"
#include "../../core/Logger.h"
#include <Windows.h>  // Windows-only application
#include <algorithm>

OpenCVCameraSource::OpenCVCameraSource() {
    m_stats.Reset();
    m_lastPropertyUpdate = std::chrono::steady_clock::now();
}

OpenCVCameraSource::~OpenCVCameraSource() {
    StopCapture();
}

bool OpenCVCameraSource::Initialize(const CameraDeviceInfo& deviceInfo, 
                                  const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(m_configMutex);
    
    if (!ValidateConfig(config)) {
        UpdateLastError("Invalid camera configuration");
        return false;
    }
    
    m_deviceInfo = deviceInfo;
    m_config = config;
    
    return InitializeCapture();
}

bool OpenCVCameraSource::InitializeCapture() {
    bool opened = false;
    CameraBackend backend = m_config.backend;
    
    // Open camera based on device type with smart backend selection
    if (m_deviceInfo.type == CameraSourceType::OPENCV_WEBCAM) {
        opened = TryOpenWebcam(m_deviceInfo.deviceIndex, backend);
        if (!opened) {
            UpdateLastError("Failed to open camera device " + std::to_string(m_deviceInfo.deviceIndex));
            return false;
        }
    } else if (m_deviceInfo.type == CameraSourceType::OPENCV_VIDEO_FILE) {
        opened = TryOpenVideoFile(m_deviceInfo.deviceName, backend);
        if (!opened) {
            UpdateLastError("Failed to open video file: " + m_deviceInfo.deviceName);
            return false;
        }
    }
    
    if (!m_capture.isOpened()) {
        UpdateLastError("Camera capture is not opened");
        return false;
    }
    
    return ConfigureCamera();
}

bool OpenCVCameraSource::ConfigureCamera() {
    // Set resolution
    m_capture.set(cv::CAP_PROP_FRAME_WIDTH, m_config.width);
    m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, m_config.height);
    
    // Set frame rate (if supported)
    if (m_config.frameRate > 0) {
        m_capture.set(cv::CAP_PROP_FPS, m_config.frameRate);
    }
    
    // Set camera properties from config (using normalized 0-1 range)
    ConfigureProperty(CameraPropertyType::BRIGHTNESS, m_config.brightness, m_currentProperties.brightness);
    ConfigureProperty(CameraPropertyType::CONTRAST, m_config.contrast, m_currentProperties.contrast);
    ConfigureProperty(CameraPropertyType::SATURATION, -1.0, m_currentProperties.saturation);
    ConfigureProperty(CameraPropertyType::GAIN, -1.0, m_currentProperties.gain);

    // Verify actual settings
    int actualWidth = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int actualHeight = static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actualFPS = m_capture.get(cv::CAP_PROP_FPS);
    
    if (actualWidth != m_config.width || actualHeight != m_config.height) {
        // Update config with actual values
        m_config.width = actualWidth;
        m_config.height = actualHeight;
    }
    
    if (actualFPS > 0) {
        m_config.frameRate = actualFPS;
    }
    
    return true;
}

bool OpenCVCameraSource::StartCapture() {
    if (m_isCapturing) {
        return true;
    }
    
    if (!m_capture.isOpened()) {
        UpdateLastError("Camera not initialized");
        return false;
    }
    
    m_shouldStop = false;
    m_isCapturing = true;
    m_stats.Reset();
    
    // Start capture thread for async frame delivery
    m_captureThread = std::make_unique<std::thread>(&OpenCVCameraSource::CaptureThreadFunc, this);
    
    return true;
}

void OpenCVCameraSource::StopCapture() {
    if (!m_isCapturing) {
        return;
    }
    
    m_shouldStop = true;
    m_isCapturing = false;
    
    if (m_captureThread && m_captureThread->joinable()) {
        m_captureThread->join();
        m_captureThread.reset();
    }
    
    if (m_capture.isOpened()) {
        m_capture.release();
    }
}

bool OpenCVCameraSource::IsCapturing() const {
    return m_isCapturing;
}

std::shared_ptr<CameraFrame> OpenCVCameraSource::CaptureFrame() {
    if (!m_isCapturing) {
        UpdateLastError("Camera not capturing");
        return nullptr;
    }

    if (!m_config.enableSyncCapture) {
        UpdateLastError("Synchronous capture is disabled");
        return nullptr;
    }

    std::unique_lock<std::mutex> lock(m_frameMutex);

    // Wait for new frame with timeout
    if (!m_frameCondition.wait_for(lock, std::chrono::milliseconds(100),
                                  [this]() { return m_hasNewFrame || m_shouldStop; })) {
        UpdateLastError("Timeout waiting for frame");
        return nullptr;
    }

    if (m_shouldStop) {
        return nullptr;
    }

    if (m_hasNewFrame && !m_currentFrame.empty()) {
        auto frame = ConvertMatToFrame(m_currentFrame);
        m_hasNewFrame = false;
        return frame;
    }

    UpdateLastError("No frame available");
    return nullptr;
}

void OpenCVCameraSource::SetFrameCallback(FrameCallback callback) {
    m_frameCallback = callback;
}

void OpenCVCameraSource::CaptureThreadFunc() {
    cv::Mat frame;

    while (!m_shouldStop && m_capture.isOpened()) {
        // Apply pending property changes with rate limiting to avoid FPS drops
        if (m_hasPendingProperties.load()) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastPropertyUpdate);

            if (elapsed.count() >= PROPERTY_UPDATE_INTERVAL_MS) {
                ApplyPendingProperties();
                m_lastPropertyUpdate = now;
            }
        }

        if (!m_capture.read(frame)) {
            if (m_deviceInfo.type == CameraSourceType::OPENCV_VIDEO_FILE) {
                // End of video file - could loop here if desired
                break;
            } else {
                // Camera read error
                RecordDroppedFrame();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
        }
        
        if (frame.empty()) {
            RecordDroppedFrame();
            continue;
        }
        
        UpdateFrameRateStats();
        
        // Update frame buffer for synchronous capture only if enabled
        if (m_config.enableSyncCapture) {
            {
                std::lock_guard<std::mutex> lock(m_frameMutex);
                frame.copyTo(m_currentFrame);
                m_frameTimestamp = std::chrono::steady_clock::now();
                m_hasNewFrame = true;
            }
            m_frameCondition.notify_one();
        }
        
        // Call async callback if set
        if (m_frameCallback) {
            auto cameraFrame = ConvertMatToFrame(frame);
            if (cameraFrame && cameraFrame->IsValid()) {
                m_frameCallback(cameraFrame);
            }
        }
        
        // Frame rate limiting
        double frameInterval = 1000.0 / m_config.frameRate;
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(frameInterval)));
    }
}

std::shared_ptr<CameraFrame> OpenCVCameraSource::ConvertMatToFrame(const cv::Mat& mat) {
    if (mat.empty()) {
        return nullptr;
    }

    // Determine format based on channels
    CameraFormat format;
    if (mat.channels() == 1) {
        format = CameraFormat::GRAY8;
    } else if (mat.channels() == 3) {
        format = CameraFormat::BGR8;  // OpenCV default
    } else if (mat.channels() == 4) {
        format = CameraFormat::BGRA8;
    } else {
        return nullptr;  // Unsupported format
    }

    return CameraFrame::CreateFromMat(mat, format);
}

CameraConfig OpenCVCameraSource::GetConfig() const {
    std::lock_guard<std::mutex> lock(m_configMutex);
    return m_config;
}

bool OpenCVCameraSource::UpdateConfig(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(m_configMutex);
    
    if (!ValidateConfig(config)) {
        UpdateLastError("Invalid camera configuration");
        return false;
    }
    
    bool wasCapturing = m_isCapturing;
    if (wasCapturing) {
        StopCapture();
    }
    
    m_config = config;
    bool result = ConfigureCamera();
    
    if (wasCapturing && result) {
        StartCapture();
    }
    
    return result;
}

CameraSourceType OpenCVCameraSource::GetSourceType() const {
    return m_deviceInfo.type;
}

CameraDeviceInfo OpenCVCameraSource::GetDeviceInfo() const {
    return m_deviceInfo;
}

CameraStats OpenCVCameraSource::GetStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

void OpenCVCameraSource::ResetStats() {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    m_stats.Reset();
}

std::string OpenCVCameraSource::GetLastError() const {
    return m_lastError;
}

bool OpenCVCameraSource::IsAvailable() const {
    return true;  // OpenCV is always available if compiled with it
}

std::string OpenCVCameraSource::GetSourceName() const {
    return "OpenCV VideoCapture";
}

std::vector<CameraDeviceInfo> OpenCVCameraSource::EnumerateDevices() {
    LOG_INFO("Starting camera device enumeration...");
    std::vector<CameraDeviceInfo> devices;
    int consecutiveFailures = 0;
    
    // Try to open cameras from index 0 to MAX_CAMERA_INDEX with simplified approach
    for (int i = 0; i < MAX_CAMERA_INDEX; ++i) {
        LOG_DEBUG("Testing camera index ", i);
        cv::VideoCapture testCap;
        bool opened = false;
        std::string backendUsed = "none";
        
        try {
            // Try DirectShow first (faster initialization when it works)
            opened = testCap.open(i, cv::CAP_DSHOW);
            if (opened) {
                backendUsed = "DirectShow";
                LOG_DEBUG("Camera ", i, " opened successfully with DirectShow backend");
            } else {
                // Fall back to MSMF if DirectShow fails
                opened = testCap.open(i, cv::CAP_MSMF);
                if (opened) {
                    backendUsed = "MSMF";
                    LOG_DEBUG("Camera ", i, " opened successfully with MSMF backend (DirectShow failed)");
                }
            }
            
            if (opened && testCap.isOpened()) {
                CameraDeviceInfo info = CreateWebcamDevice(i);
                
                // Get camera capabilities with error checking
                info.maxWidth = static_cast<int>(testCap.get(cv::CAP_PROP_FRAME_WIDTH));
                info.maxHeight = static_cast<int>(testCap.get(cv::CAP_PROP_FRAME_HEIGHT));
                info.maxFrameRate = testCap.get(cv::CAP_PROP_FPS);
                
                // Validate the retrieved values
                if (info.maxWidth > 0 && info.maxHeight > 0) {
                    LOG_INFO("Found camera ", i, " (", info.deviceName, ") - Resolution: ", 
                            info.maxWidth, "x", info.maxHeight, " @ ", info.maxFrameRate, 
                            " FPS using ", backendUsed, " backend");
                    devices.push_back(info);
                    consecutiveFailures = 0;  // Reset failure counter
                } else {
                    LOG_WARNING("Camera ", i, " opened but has invalid capabilities - skipping");
                    consecutiveFailures++;
                }
                
                testCap.release();
            } else {
                LOG_DEBUG("Camera ", i, " not available (both DirectShow and MSMF failed)");
                consecutiveFailures++;
            }
        } catch (...) {
            // Ignore problematic cameras
            LOG_DEBUG("Camera ", i, " threw exception during testing - skipping");
            consecutiveFailures++;
        }
        
        // Early exit: if we have 2 consecutive failures and at least one camera found,
        // assume no more cameras are available
        if (consecutiveFailures >= 2 && !devices.empty()) {
            LOG_DEBUG("Early exit after ", consecutiveFailures, " consecutive failures with ", devices.size(), " cameras found");
            break;
        }
        
        // Complete failure early exit: if first 3 indices fail, likely no cameras at all
        if (i >= 2 && devices.empty() && consecutiveFailures >= 3) {
            LOG_DEBUG("No cameras found after testing indices 0-", i, " - stopping enumeration");
            break;
        }
    }
    
    LOG_INFO("Camera enumeration complete - found ", devices.size(), " camera(s)");
    return devices;
}

CameraDeviceInfo OpenCVCameraSource::CreateWebcamDevice(int deviceIndex) {
    CameraDeviceInfo info;
    info.deviceIndex = deviceIndex;
    info.deviceName = "Camera " + std::to_string(deviceIndex);
    info.serialNumber = "opencv_" + std::to_string(deviceIndex);
    info.type = CameraSourceType::OPENCV_WEBCAM;
    info.supportsDepth = false;
    return info;
}

CameraDeviceInfo OpenCVCameraSource::CreateVideoFileDevice(const std::string& filePath) {
    CameraDeviceInfo info;
    info.deviceIndex = -1;
    info.deviceName = filePath;
    info.serialNumber = "file_" + std::to_string(std::hash<std::string>{}(filePath));
    info.type = CameraSourceType::OPENCV_VIDEO_FILE;
    info.supportsDepth = false;
    return info;
}

bool OpenCVCameraSource::ValidateConfig(const CameraConfig& config) {
    return config.width > 0 && config.height > 0 && config.frameRate > 0;
}

void OpenCVCameraSource::UpdateLastError(const std::string& error) {
    m_lastError = error;
}

double OpenCVCameraSource::GetOpenCVProperty(int propId) const {
    if (m_capture.isOpened()) {
        return m_capture.get(propId);
    }
    return -1.0;
}

bool OpenCVCameraSource::SetOpenCVProperty(int propId, double value) {
    if (!m_capture.isOpened()) {
        return false;
    }

    // Get current value for comparison
    double currentValue = m_capture.get(propId);

    // Try to set the new value
    bool success = m_capture.set(propId, value);

    if (success) {
        // Verify the value was actually changed
        double actualValue = m_capture.get(propId);

        // Some cameras don't change values or have limited ranges
        if (std::abs(actualValue - currentValue) < 0.001) {
            LOG_DEBUG("Property ", propId, " value unchanged (current: ", currentValue, ", requested: ", value, ")");
            // Still consider it a success - the camera might not support this range
        } else {
            LOG_DEBUG("Property ", propId, " changed from ", currentValue, " to ", actualValue, " (requested: ", value, ")");
        }
    } else {
        LOG_WARNING("Failed to set property ", propId, " to ", value);
    }

    return success;
}

OpenCVCameraSource::CameraPropertyRange OpenCVCameraSource::DetectPropertyRange(int openCVPropId) const {
    CameraPropertyRange range;

    if (!m_capture.isOpened()) {
        return range;
    }

    // Check cache first
    auto it = m_propertyRangeCache.find(openCVPropId);
    if (it != m_propertyRangeCache.end()) {
        return it->second;
    }

    // Get current value
    double currentValue = m_capture.get(openCVPropId);
    if (currentValue < 0) {
        return range; // Property not supported
    }

    // Store current value for default calculation
    double currentAsDefault = currentValue;

    // Use property-specific heuristics to determine range
    if (openCVPropId == cv::CAP_PROP_SATURATION) {
        // Most webcams use 0-255 range for saturation, even if they start with low values
        // Use 0-255 as default unless the current value is clearly in a different range
        if (currentValue >= 1000) {
            // Very high values suggest extended range
            range.min = 0;
            range.max = 10000;
        } else if (currentValue > 255.0) {
            // Above 255 suggests larger range
            range.min = 0;
            range.max = 1000;
        } else {
            // Default to 0-255 for saturation (most common)
            range.min = 0;
            range.max = 255;
        }
    } else {
        // General heuristics for other properties based on current value
        if (currentValue >= 10000) {
            // Large values suggest 16-bit range (0-65535)
            range.min = 0;
            range.max = 65535;
        } else if (currentValue >= 1000) {
            // Medium-large values suggest extended range (0-10000)
            range.min = 0;
            range.max = 10000;
        } else if (currentValue >= 100) {
            // Medium values suggest 8-bit range (0-255)
            range.min = 0;
            range.max = 255;
        } else if (currentValue >= 10) {
            // Smaller values suggest percentage range (0-100)
            range.min = 0;
            range.max = 100;
        } else {
            // Small values suggest normalized range (0-1)
            range.min = 0;
            range.max = 1;
        }
    }

    range.supported = true;
    range.step = 1;

    // Calculate default value manually to avoid circular dependency
    double normalized = (currentAsDefault - static_cast<double>(range.min)) / (static_cast<double>(range.max) - static_cast<double>(range.min));
    normalized = std::max(0.0, std::min(1.0, normalized)); // Clamp to [0,1]
    range.defaultValue = static_cast<int>(normalized * 100.0);

    LOG_DEBUG("Estimated property ", openCVPropId, " range: [", range.min, ", ", range.max, "], default: ", range.defaultValue);

    // Cache the result
    m_propertyRangeCache[openCVPropId] = range;

    return range;
}

double OpenCVCameraSource::ConvertToNormalizedValue(double cameraValue, const CameraPropertyRange& range) const {
    if (!range.supported || range.max <= range.min) {
        // Fallback: assume 0-1 range
        return cameraValue;
    }

    // Map camera range to 0-1 normalized range
    double normalized = (cameraValue - static_cast<double>(range.min)) / (static_cast<double>(range.max) - static_cast<double>(range.min));
    return std::max(0.0, std::min(1.0, normalized)); // Clamp to [0,1]
}

double OpenCVCameraSource::ConvertFromNormalizedValue(double normalizedValue, const CameraPropertyRange& range) const {
    if (!range.supported || range.max <= range.min) {
        // Fallback: assume 0-1 range
        return normalizedValue;
    }

    // Map normalized range (0-1) to camera range
    double clamped = std::max(0.0, std::min(1.0, normalizedValue)); // Clamp to [0,1]
    return static_cast<double>(range.min) + clamped * (static_cast<double>(range.max) - static_cast<double>(range.min));
}

int OpenCVCameraSource::ConvertBackendToOpenCV(CameraBackend backend) const {
    switch (backend) {
        case CameraBackend::FORCE_DSHOW:
        case CameraBackend::DSHOW:
            return cv::CAP_DSHOW;
        case CameraBackend::FORCE_MSMF:
        case CameraBackend::MSMF:
            return cv::CAP_MSMF;
        case CameraBackend::AUTO:
        case CameraBackend::DEFAULT:
        case CameraBackend::PREFER_DSHOW:
        case CameraBackend::PREFER_MSMF:
        default:
            return -1;  // Use smart selection logic
    }
}

CameraBackend OpenCVCameraSource::GetOptimalBackend() const {
    // Windows-only application: prefer DirectShow with MSMF fallback
    return CameraBackend::PREFER_DSHOW;
}

bool OpenCVCameraSource::TryOpenWebcam(int deviceIndex, CameraBackend backend) {
    LOG_DEBUG("Attempting to open camera ", deviceIndex, " with backend configuration");
    
    switch (backend) {
        case CameraBackend::FORCE_DSHOW:
        case CameraBackend::DSHOW:
            // Force DirectShow only
            LOG_DEBUG("Forcing DirectShow backend for camera ", deviceIndex);
            if (m_capture.open(deviceIndex, cv::CAP_DSHOW)) {
                LOG_INFO("Camera ", deviceIndex, " opened successfully with DirectShow backend");
                m_lastError = "Using DirectShow backend";
                return true;
            } else {
                LOG_ERROR("Failed to open camera ", deviceIndex, " with DirectShow backend");
                return false;
            }
            
        case CameraBackend::FORCE_MSMF:
        case CameraBackend::MSMF:
            // Force MSMF only
            LOG_DEBUG("Forcing MSMF backend for camera ", deviceIndex);
            if (m_capture.open(deviceIndex, cv::CAP_MSMF)) {
                LOG_INFO("Camera ", deviceIndex, " opened successfully with MSMF backend");
                m_lastError = "Using MSMF backend";
                return true;
            } else {
                LOG_ERROR("Failed to open camera ", deviceIndex, " with MSMF backend");
                return false;
            }
            
        case CameraBackend::PREFER_MSMF:
            // Try MSMF first (reliable but slower)
            LOG_DEBUG("Trying MSMF backend first for camera ", deviceIndex);
            if (m_capture.open(deviceIndex, cv::CAP_MSMF)) {
                LOG_INFO("Camera ", deviceIndex, " opened successfully with MSMF backend");
                m_lastError = "Using MSMF backend";
                return true;
            }
            // No fallback for PREFER_MSMF
            LOG_ERROR("MSMF backend failed for camera ", deviceIndex, ", no fallback configured");
            m_lastError = "MSMF backend failed";
            return false;
            
        case CameraBackend::AUTO:
        case CameraBackend::DEFAULT:
        case CameraBackend::PREFER_DSHOW:
        default:
            // Try DirectShow first (faster when it works), fallback to MSMF
            LOG_DEBUG("Trying DirectShow first for camera ", deviceIndex, ", with MSMF fallback");
            if (m_capture.open(deviceIndex, cv::CAP_DSHOW)) {
                LOG_INFO("Camera ", deviceIndex, " opened successfully with DirectShow backend");
                m_lastError = "Using DirectShow backend";
                return true;
            }
            
            LOG_WARNING("DirectShow failed for camera ", deviceIndex, ", trying MSMF fallback");
            if (m_capture.open(deviceIndex, cv::CAP_MSMF)) {
                LOG_INFO("Camera ", deviceIndex, " opened successfully with MSMF backend (DirectShow failed)");
                m_lastError = "Using MSMF backend (DirectShow failed)";
                return true;
            }
            
            LOG_ERROR("Both DirectShow and MSMF backends failed for camera ", deviceIndex);
            m_lastError = "Both DirectShow and MSMF backends failed";
            return false;
    }
}

bool OpenCVCameraSource::TryOpenVideoFile(const std::string& filename, CameraBackend backend) {
    LOG_DEBUG("Attempting to open video file: ", filename);
    
    switch (backend) {
        case CameraBackend::FORCE_DSHOW:
        case CameraBackend::DSHOW:
            LOG_DEBUG("Forcing DirectShow backend for video file");
            if (m_capture.open(filename, cv::CAP_DSHOW)) {
                LOG_INFO("Video file opened successfully with DirectShow backend");
                m_lastError = "Using DirectShow backend for video file";
                return true;
            } else {
                LOG_ERROR("Failed to open video file with DirectShow backend");
                return false;
            }
            
        case CameraBackend::FORCE_MSMF:
        case CameraBackend::MSMF:
            LOG_DEBUG("Forcing MSMF backend for video file");
            if (m_capture.open(filename, cv::CAP_MSMF)) {
                LOG_INFO("Video file opened successfully with MSMF backend");
                m_lastError = "Using MSMF backend for video file";
                return true;
            } else {
                LOG_ERROR("Failed to open video file with MSMF backend");
                return false;
            }
            
        case CameraBackend::PREFER_MSMF:
            LOG_DEBUG("Trying MSMF backend first for video file");
            if (m_capture.open(filename, cv::CAP_MSMF)) {
                LOG_INFO("Video file opened successfully with MSMF backend");
                m_lastError = "Using MSMF backend for video file";
                return true;
            }
            LOG_ERROR("MSMF backend failed for video file, no fallback configured");
            return false;
            
        case CameraBackend::AUTO:
        case CameraBackend::DEFAULT:
        case CameraBackend::PREFER_DSHOW:
        default:
            // Try DirectShow first, fallback to MSMF, then default
            LOG_DEBUG("Trying DirectShow first for video file, with multiple fallbacks");
            if (m_capture.open(filename, cv::CAP_DSHOW)) {
                LOG_INFO("Video file opened successfully with DirectShow backend");
                m_lastError = "Using DirectShow backend for video file";
                return true;
            }
            
            LOG_WARNING("DirectShow failed for video file, trying MSMF fallback");
            if (m_capture.open(filename, cv::CAP_MSMF)) {
                LOG_INFO("Video file opened successfully with MSMF backend (DirectShow failed)");
                m_lastError = "Using MSMF backend for video file (DirectShow failed)";
                return true;
            }
            
            LOG_WARNING("MSMF failed for video file, trying default backend");
            // Final fallback - let OpenCV choose
            if (m_capture.open(filename)) {
                LOG_INFO("Video file opened successfully with default backend (DSHOW and MSMF failed)");
                m_lastError = "Using default backend for video file";
                return true;
            }
            
            LOG_ERROR("All backends failed for video file: ", filename);
            return false;
    }
}

// Runtime property control implementation
bool OpenCVCameraSource::SetCameraProperty(CameraPropertyType property, double value) {
    if (!ValidatePropertyValue(property, value)) {
        UpdateLastError("Invalid property value: " + std::to_string(value));
        return false;
    }

    std::lock_guard<std::mutex> lock(m_propertyMutex);

    // Update pending properties
    switch (property) {
        case CameraPropertyType::BRIGHTNESS:
            m_pendingProperties.brightness = value;
            break;
        case CameraPropertyType::CONTRAST:
            m_pendingProperties.contrast = value;
            break;
        case CameraPropertyType::SATURATION:
            m_pendingProperties.saturation = value;
            break;
        case CameraPropertyType::GAIN:
            m_pendingProperties.gain = value;
            break;
        default:
            UpdateLastError("Unsupported property type");
            return false;
    }

    // Signal that properties need to be applied
    m_hasPendingProperties.store(true);
    return true;
}

bool OpenCVCameraSource::GetCameraProperty(CameraPropertyType property, double& value) const {
    std::lock_guard<std::mutex> lock(m_propertyMutex);

    switch (property) {
        case CameraPropertyType::BRIGHTNESS:
            value = m_currentProperties.brightness;
            return !std::isnan(m_currentProperties.brightness);
        case CameraPropertyType::CONTRAST:
            value = m_currentProperties.contrast;
            return !std::isnan(m_currentProperties.contrast);
        case CameraPropertyType::SATURATION:
            value = m_currentProperties.saturation;
            return !std::isnan(m_currentProperties.saturation);
        case CameraPropertyType::GAIN:
            value = m_currentProperties.gain;
            return !std::isnan(m_currentProperties.gain);
        default:
            return false;
    }
}

bool OpenCVCameraSource::SetCameraProperties(const CameraProperties& properties) {
    if (!properties.HasChanges()) {
        return true; // No changes to apply
    }

    std::lock_guard<std::mutex> lock(m_propertyMutex);

    // Validate all properties first
    if (!std::isnan(properties.brightness) && !ValidatePropertyValue(CameraPropertyType::BRIGHTNESS, properties.brightness)) {
        UpdateLastError("Invalid brightness value: " + std::to_string(properties.brightness));
        return false;
    }
    if (!std::isnan(properties.contrast) && !ValidatePropertyValue(CameraPropertyType::CONTRAST, properties.contrast)) {
        UpdateLastError("Invalid contrast value: " + std::to_string(properties.contrast));
        return false;
    }
    if (!std::isnan(properties.saturation) && !ValidatePropertyValue(CameraPropertyType::SATURATION, properties.saturation)) {
        UpdateLastError("Invalid saturation value: " + std::to_string(properties.saturation));
        return false;
    }
    if (!std::isnan(properties.gain) && !ValidatePropertyValue(CameraPropertyType::GAIN, properties.gain)) {
        UpdateLastError("Invalid gain value: " + std::to_string(properties.gain));
        return false;
    }

    // Copy valid properties to pending
    if (!std::isnan(properties.brightness)) {
        m_pendingProperties.brightness = properties.brightness;
    }
    if (!std::isnan(properties.contrast)) {
        m_pendingProperties.contrast = properties.contrast;
    }
    if (!std::isnan(properties.saturation)) {
        m_pendingProperties.saturation = properties.saturation;
    }
    if (!std::isnan(properties.gain)) {
        m_pendingProperties.gain = properties.gain;
    }

    // Signal that properties need to be applied
    m_hasPendingProperties.store(true);
    return true;
}

CameraProperties OpenCVCameraSource::GetCameraProperties() const {
    std::lock_guard<std::mutex> lock(m_propertyMutex);
    return m_currentProperties;
}

std::set<CameraPropertyType> OpenCVCameraSource::GetSupportedProperties() const {
    std::set<CameraPropertyType> supportedProperties;

    if (!m_capture.isOpened()) {
        return supportedProperties;
    }

    // Check which properties are supported by trying to get their current values
    for (auto property : {CameraPropertyType::BRIGHTNESS, CameraPropertyType::CONTRAST,
                         CameraPropertyType::SATURATION, CameraPropertyType::GAIN}) {
        int openCVProp = ConvertPropertyTypeToOpenCV(property);
        if (openCVProp != -1) {
            double currentValue = GetOpenCVProperty(openCVProp);
            if (currentValue >= 0) {
                supportedProperties.insert(property);
            }
        }
    }

    return supportedProperties;
}

void OpenCVCameraSource::ApplyPendingProperties() {
    std::lock_guard<std::mutex> lock(m_propertyMutex);

    if (!m_pendingProperties.HasChanges()) {
        m_hasPendingProperties.store(false);
        return;
    }

    LOG_DEBUG("Applying pending camera properties...");

    // Apply properties in optimal order to minimize camera reconfiguration
    // 1. Apply basic properties first (less likely to cause reconfiguration)
    if (!std::isnan(m_pendingProperties.brightness)) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::BRIGHTNESS);
        if (openCVProp != -1) {
            CameraPropertyRange range = DetectPropertyRange(openCVProp);
            double setValue = ConvertFromNormalizedValue(m_pendingProperties.brightness, range);

            LOG_DEBUG("Setting brightness: normalized value=", m_pendingProperties.brightness, ", camera value=", setValue, " (range: ", range.min, "-", range.max, ")");

            if (SetOpenCVProperty(openCVProp, setValue)) {
                double actualValue = GetOpenCVProperty(openCVProp);
                if (actualValue >= 0) {
                    m_currentProperties.brightness = ConvertToNormalizedValue(actualValue, range);
                    LOG_DEBUG("Brightness applied: target=", m_pendingProperties.brightness, ", actual camera value=", actualValue, ", normalized value=", m_currentProperties.brightness);
                }
            } else {
                LOG_WARNING("Failed to set brightness to ", m_pendingProperties.brightness);
            }
        }
    }

    if (!std::isnan(m_pendingProperties.contrast)) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::CONTRAST);
        if (openCVProp != -1) {
            CameraPropertyRange range = DetectPropertyRange(openCVProp);
            double setValue = ConvertFromNormalizedValue(m_pendingProperties.contrast, range);

            LOG_DEBUG("Setting contrast: normalized value=", m_pendingProperties.contrast, ", camera value=", setValue, " (range: ", range.min, "-", range.max, ")");

            if (SetOpenCVProperty(openCVProp, setValue)) {
                double actualValue = GetOpenCVProperty(openCVProp);
                if (actualValue >= 0) {
                    m_currentProperties.contrast = ConvertToNormalizedValue(actualValue, range);
                    LOG_DEBUG("Contrast applied: target=", m_pendingProperties.contrast, ", actual camera value=", actualValue, ", normalized value=", m_currentProperties.contrast);
                }
            } else {
                LOG_WARNING("Failed to set contrast to ", m_pendingProperties.contrast);
            }
        }
    }

    if (!std::isnan(m_pendingProperties.saturation)) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::SATURATION);
        if (openCVProp != -1) {
            CameraPropertyRange range = DetectPropertyRange(openCVProp);
            double setValue = ConvertFromNormalizedValue(m_pendingProperties.saturation, range);

            LOG_DEBUG("Setting saturation: normalized value=", m_pendingProperties.saturation, ", camera value=", setValue, " (range: ", range.min, "-", range.max, ")");

            if (SetOpenCVProperty(openCVProp, setValue)) {
                double actualValue = GetOpenCVProperty(openCVProp);
                if (actualValue >= 0) {
                    m_currentProperties.saturation = ConvertToNormalizedValue(actualValue, range);
                    LOG_DEBUG("Saturation applied: target=", m_pendingProperties.saturation, ", actual camera value=", actualValue, ", normalized value=", m_currentProperties.saturation);
                }
            } else {
                LOG_WARNING("Failed to set saturation to ", m_pendingProperties.saturation);
            }
        }
    }

    if (!std::isnan(m_pendingProperties.gain)) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::GAIN);
        if (openCVProp != -1) {
            CameraPropertyRange range = DetectPropertyRange(openCVProp);
            double setValue = ConvertFromNormalizedValue(m_pendingProperties.gain, range);

            LOG_DEBUG("Setting gain: normalized value=", m_pendingProperties.gain, ", camera value=", setValue, " (range: ", range.min, "-", range.max, ")");

            if (SetOpenCVProperty(openCVProp, setValue)) {
                double actualValue = GetOpenCVProperty(openCVProp);
                if (actualValue >= 0) {
                    m_currentProperties.gain = ConvertToNormalizedValue(actualValue, range);
                    LOG_DEBUG("Gain applied: target=", m_pendingProperties.gain, ", actual camera value=", actualValue, ", normalized value=", m_currentProperties.gain);
                }
            } else {
                LOG_WARNING("Failed to set gain to ", m_pendingProperties.gain);
            }
        }
    }


    // Clear pending properties and flag
    m_pendingProperties.Reset();
    m_hasPendingProperties.store(false);
}

int OpenCVCameraSource::ConvertPropertyTypeToOpenCV(CameraPropertyType property) const {
    switch (property) {
        case CameraPropertyType::BRIGHTNESS:
            return cv::CAP_PROP_BRIGHTNESS;
        case CameraPropertyType::CONTRAST:
            return cv::CAP_PROP_CONTRAST;
        case CameraPropertyType::SATURATION:
            return cv::CAP_PROP_SATURATION;
        case CameraPropertyType::GAIN:
            return cv::CAP_PROP_GAIN;
        default:
            return -1;
    }
}

bool OpenCVCameraSource::ValidatePropertyValue(CameraPropertyType property, double value) const {
    switch (property) {
        case CameraPropertyType::BRIGHTNESS:
        case CameraPropertyType::CONTRAST:
        case CameraPropertyType::SATURATION:
        case CameraPropertyType::GAIN:
        default:
            // All properties use 0.0-1.0 normalized range
            return value >= 0.0 && value <= 1.0;
    }
}

void OpenCVCameraSource::ConfigureProperty(CameraPropertyType property, double configValue, double& currentValue) {
    int openCVProp = ConvertPropertyTypeToOpenCV(property);
    if (openCVProp == -1) {
        return; // Property type not supported
    }

    // First, check if property is supported by getting current value
    double currentCameraValue = GetOpenCVProperty(openCVProp);
    if (currentCameraValue < 0) {
        return; // Property not supported by camera
    }

    // Property is supported, detect its range once and cache it
    CameraPropertyRange range = DetectPropertyRange(openCVProp);

    // Set config value if specified (>= 0.0)
    if (configValue >= 0.0) {
        double cameraValue = ConvertFromNormalizedValue(configValue, range);
        if (SetOpenCVProperty(openCVProp, cameraValue)) {
            // Re-read actual value after setting
            currentCameraValue = GetOpenCVProperty(openCVProp);
        }
    }

    // Update normalized current value from actual camera value (using cached range)
    if (currentCameraValue >= 0) {
        currentValue = ConvertToNormalizedValue(currentCameraValue, range);
    }
}