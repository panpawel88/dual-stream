#include "OpenCVCameraSource.h"
#include "../../core/Logger.h"
#include <Windows.h>  // Windows-only application

OpenCVCameraSource::OpenCVCameraSource() {
    m_stats.Reset();
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
    
    // Set camera properties if specified
    if (m_config.brightness >= 0) {
        SetOpenCVProperty(cv::CAP_PROP_BRIGHTNESS, m_config.brightness / 100.0);
        // Get actual value set by camera
        double actualBrightness = GetOpenCVProperty(cv::CAP_PROP_BRIGHTNESS);
        if (actualBrightness >= 0) {
            m_currentProperties.brightness = static_cast<int>(actualBrightness * 100.0);
        }
    }
    if (m_config.contrast >= 0) {
        SetOpenCVProperty(cv::CAP_PROP_CONTRAST, m_config.contrast / 100.0);
        // Get actual value set by camera
        double actualContrast = GetOpenCVProperty(cv::CAP_PROP_CONTRAST);
        if (actualContrast >= 0) {
            m_currentProperties.contrast = static_cast<int>(actualContrast * 100.0);
        }
    }
    if (m_config.exposure >= 0) {
        SetOpenCVProperty(cv::CAP_PROP_EXPOSURE, m_config.exposure / 100.0);
        // Get actual value set by camera
        double actualExposure = GetOpenCVProperty(cv::CAP_PROP_EXPOSURE);
        if (actualExposure >= 0) {
            m_currentProperties.exposure = static_cast<int>(actualExposure * 100.0);
        }
    }
    
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
        // Apply pending property changes
        if (m_hasPendingProperties.load()) {
            ApplyPendingProperties();
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
    if (m_capture.isOpened()) {
        return m_capture.set(propId, value);
    }
    return false;
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
bool OpenCVCameraSource::SetCameraProperty(CameraPropertyType property, int value) {
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
        case CameraPropertyType::EXPOSURE:
            m_pendingProperties.exposure = value;
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

bool OpenCVCameraSource::GetCameraProperty(CameraPropertyType property, int& value) const {
    std::lock_guard<std::mutex> lock(m_propertyMutex);

    switch (property) {
        case CameraPropertyType::BRIGHTNESS:
            value = m_currentProperties.brightness;
            return m_currentProperties.brightness != -1;
        case CameraPropertyType::CONTRAST:
            value = m_currentProperties.contrast;
            return m_currentProperties.contrast != -1;
        case CameraPropertyType::EXPOSURE:
            value = m_currentProperties.exposure;
            return m_currentProperties.exposure != -1;
        case CameraPropertyType::SATURATION:
            value = m_currentProperties.saturation;
            return m_currentProperties.saturation != -1;
        case CameraPropertyType::GAIN:
            value = m_currentProperties.gain;
            return m_currentProperties.gain != -1;
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
    if (properties.brightness != -1 && !ValidatePropertyValue(CameraPropertyType::BRIGHTNESS, properties.brightness)) {
        UpdateLastError("Invalid brightness value: " + std::to_string(properties.brightness));
        return false;
    }
    if (properties.contrast != -1 && !ValidatePropertyValue(CameraPropertyType::CONTRAST, properties.contrast)) {
        UpdateLastError("Invalid contrast value: " + std::to_string(properties.contrast));
        return false;
    }
    if (properties.exposure != -1 && !ValidatePropertyValue(CameraPropertyType::EXPOSURE, properties.exposure)) {
        UpdateLastError("Invalid exposure value: " + std::to_string(properties.exposure));
        return false;
    }
    if (properties.saturation != -1 && !ValidatePropertyValue(CameraPropertyType::SATURATION, properties.saturation)) {
        UpdateLastError("Invalid saturation value: " + std::to_string(properties.saturation));
        return false;
    }
    if (properties.gain != -1 && !ValidatePropertyValue(CameraPropertyType::GAIN, properties.gain)) {
        UpdateLastError("Invalid gain value: " + std::to_string(properties.gain));
        return false;
    }

    // Copy valid properties to pending
    if (properties.brightness != -1) {
        m_pendingProperties.brightness = properties.brightness;
    }
    if (properties.contrast != -1) {
        m_pendingProperties.contrast = properties.contrast;
    }
    if (properties.exposure != -1) {
        m_pendingProperties.exposure = properties.exposure;
    }
    if (properties.saturation != -1) {
        m_pendingProperties.saturation = properties.saturation;
    }
    if (properties.gain != -1) {
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

CameraPropertyRange OpenCVCameraSource::GetPropertyRange(CameraPropertyType property) const {
    // OpenCV properties typically use 0.0-1.0 range, but we present as 0-100 for user convenience
    switch (property) {
        case CameraPropertyType::BRIGHTNESS:
        case CameraPropertyType::CONTRAST:
        case CameraPropertyType::SATURATION:
        case CameraPropertyType::GAIN:
            return CameraPropertyRange{0, 100, 50, 1, true};
        case CameraPropertyType::EXPOSURE:
            // Exposure might have different range depending on camera
            return CameraPropertyRange{0, 100, 50, 1, true};
        default:
            return CameraPropertyRange{0, 100, 50, 1, false};
    }
}

void OpenCVCameraSource::ApplyPendingProperties() {
    std::lock_guard<std::mutex> lock(m_propertyMutex);

    if (!m_pendingProperties.HasChanges()) {
        m_hasPendingProperties.store(false);
        return;
    }

    // Apply each pending property
    if (m_pendingProperties.brightness != -1) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::BRIGHTNESS);
        if (openCVProp != -1 && SetOpenCVProperty(openCVProp, m_pendingProperties.brightness / 100.0)) {
            // Get actual value set by camera
            double actualValue = GetOpenCVProperty(openCVProp);
            if (actualValue >= 0) {
                m_currentProperties.brightness = static_cast<int>(actualValue * 100.0);
            }
        }
    }

    if (m_pendingProperties.contrast != -1) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::CONTRAST);
        if (openCVProp != -1 && SetOpenCVProperty(openCVProp, m_pendingProperties.contrast / 100.0)) {
            // Get actual value set by camera
            double actualValue = GetOpenCVProperty(openCVProp);
            if (actualValue >= 0) {
                m_currentProperties.contrast = static_cast<int>(actualValue * 100.0);
            }
        }
    }

    if (m_pendingProperties.exposure != -1) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::EXPOSURE);
        if (openCVProp != -1 && SetOpenCVProperty(openCVProp, m_pendingProperties.exposure / 100.0)) {
            // Get actual value set by camera
            double actualValue = GetOpenCVProperty(openCVProp);
            if (actualValue >= 0) {
                m_currentProperties.exposure = static_cast<int>(actualValue * 100.0);
            }
        }
    }

    if (m_pendingProperties.saturation != -1) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::SATURATION);
        if (openCVProp != -1 && SetOpenCVProperty(openCVProp, m_pendingProperties.saturation / 100.0)) {
            // Get actual value set by camera
            double actualValue = GetOpenCVProperty(openCVProp);
            if (actualValue >= 0) {
                m_currentProperties.saturation = static_cast<int>(actualValue * 100.0);
            }
        }
    }

    if (m_pendingProperties.gain != -1) {
        int openCVProp = ConvertPropertyTypeToOpenCV(CameraPropertyType::GAIN);
        if (openCVProp != -1 && SetOpenCVProperty(openCVProp, m_pendingProperties.gain / 100.0)) {
            // Get actual value set by camera
            double actualValue = GetOpenCVProperty(openCVProp);
            if (actualValue >= 0) {
                m_currentProperties.gain = static_cast<int>(actualValue * 100.0);
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
        case CameraPropertyType::EXPOSURE:
            return cv::CAP_PROP_EXPOSURE;
        case CameraPropertyType::SATURATION:
            return cv::CAP_PROP_SATURATION;
        case CameraPropertyType::GAIN:
            return cv::CAP_PROP_GAIN;
        default:
            return -1;
    }
}

bool OpenCVCameraSource::ValidatePropertyValue(CameraPropertyType property, int value) const {
    // For now, validate that values are in 0-100 range
    return value >= 0 && value <= 100;
}