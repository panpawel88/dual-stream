#include "OpenCVCameraSource.h"

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
    // Open camera based on device type
    if (m_deviceInfo.type == CameraSourceType::OPENCV_WEBCAM) {
        if (!m_capture.open(m_deviceInfo.deviceIndex)) {
            UpdateLastError("Failed to open camera device " + std::to_string(m_deviceInfo.deviceIndex));
            return false;
        }
    } else if (m_deviceInfo.type == CameraSourceType::OPENCV_VIDEO_FILE) {
        if (!m_capture.open(m_deviceInfo.deviceName)) {
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
        SetCameraProperty(cv::CAP_PROP_BRIGHTNESS, m_config.brightness / 100.0);
    }
    if (m_config.contrast >= 0) {
        SetCameraProperty(cv::CAP_PROP_CONTRAST, m_config.contrast / 100.0);
    }
    if (m_config.exposure >= 0) {
        SetCameraProperty(cv::CAP_PROP_EXPOSURE, m_config.exposure / 100.0);
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

bool OpenCVCameraSource::CaptureFrame(CameraFrame& frame) {
    if (!m_isCapturing) {
        UpdateLastError("Camera not capturing");
        return false;
    }
    
    std::unique_lock<std::mutex> lock(m_frameMutex);
    
    // Wait for new frame with timeout
    if (!m_frameCondition.wait_for(lock, std::chrono::milliseconds(100), 
                                  [this]() { return m_hasNewFrame || m_shouldStop; })) {
        UpdateLastError("Timeout waiting for frame");
        return false;
    }
    
    if (m_shouldStop) {
        return false;
    }
    
    if (m_hasNewFrame && !m_currentFrame.empty()) {
        frame = ConvertMatToFrame(m_currentFrame);
        m_hasNewFrame = false;
        return frame.IsValid();
    }
    
    UpdateLastError("No frame available");
    return false;
}

void OpenCVCameraSource::SetFrameCallback(FrameCallback callback) {
    m_frameCallback = callback;
}

void OpenCVCameraSource::CaptureThreadFunc() {
    cv::Mat frame;
    
    while (!m_shouldStop && m_capture.isOpened()) {
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
        
        // Update frame buffer for synchronous capture
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            frame.copyTo(m_currentFrame);
            m_frameTimestamp = std::chrono::steady_clock::now();
            m_hasNewFrame = true;
        }
        m_frameCondition.notify_one();
        
        // Call async callback if set
        if (m_frameCallback) {
            CameraFrame cameraFrame = ConvertMatToFrame(frame);
            if (cameraFrame.IsValid()) {
                m_frameCallback(cameraFrame);
            }
        }
        
        // Frame rate limiting
        double frameInterval = 1000.0 / m_config.frameRate;
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(frameInterval)));
    }
}

CameraFrame OpenCVCameraSource::ConvertMatToFrame(const cv::Mat& mat) {
    if (mat.empty()) {
        return CameraFrame{};
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
        return CameraFrame{};  // Unsupported format
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
    std::vector<CameraDeviceInfo> devices;
    
    // Try to open cameras from index 0 to MAX_CAMERA_INDEX
    for (int i = 0; i < MAX_CAMERA_INDEX; ++i) {
        cv::VideoCapture testCap(i);
        if (testCap.isOpened()) {
            CameraDeviceInfo info = CreateWebcamDevice(i);
            
            // Get camera capabilities
            info.maxWidth = static_cast<int>(testCap.get(cv::CAP_PROP_FRAME_WIDTH));
            info.maxHeight = static_cast<int>(testCap.get(cv::CAP_PROP_FRAME_HEIGHT));
            info.maxFrameRate = testCap.get(cv::CAP_PROP_FPS);
            
            devices.push_back(info);
            testCap.release();
        }
    }
    
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

double OpenCVCameraSource::GetCameraProperty(int propId) const {
    if (m_capture.isOpened()) {
        return m_capture.get(propId);
    }
    return -1.0;
}

bool OpenCVCameraSource::SetCameraProperty(int propId, double value) {
    if (m_capture.isOpened()) {
        return m_capture.set(propId, value);
    }
    return false;
}