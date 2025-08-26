#include "RealSenseCameraSource.h"

#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#include <cstring>

RealSenseCameraSource::RealSenseCameraSource() 
    : m_pipeline(std::make_unique<rs2::pipeline>())
    , m_config_rs(std::make_unique<rs2::config>())
    , m_context(std::make_unique<rs2::context>()) {
    m_stats.Reset();
}

RealSenseCameraSource::~RealSenseCameraSource() {
    StopCapture();
}

bool RealSenseCameraSource::Initialize(const CameraDeviceInfo& deviceInfo, 
                                      const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(m_configMutex);
    
    if (!ValidateConfig(config)) {
        UpdateLastError("Invalid camera configuration");
        return false;
    }
    
    if (!IsRealSenseAvailable()) {
        UpdateLastError("RealSense SDK not available");
        return false;
    }
    
    m_deviceInfo = deviceInfo;
    m_config = config;
    
    return InitializePipeline();
}

bool RealSenseCameraSource::InitializePipeline() {
    try {
        // Configure RGB stream
        m_config_rs->enable_stream(RS2_STREAM_COLOR, m_config.width, m_config.height, 
                                 RS2_FORMAT_BGR8, static_cast<int>(m_config.frameRate));
        
        // Configure depth stream if enabled
        if (m_config.enableDepth && m_deviceInfo.supportsDepth) {
            m_config_rs->enable_stream(RS2_STREAM_DEPTH, m_config.width, m_config.height, 
                                     RS2_FORMAT_Z16, static_cast<int>(m_config.frameRate));
        }
        
        // Enable specific device if serial number is provided
        if (!m_deviceInfo.serialNumber.empty() && 
            m_deviceInfo.serialNumber.find("realsense_") == 0) {
            std::string serialNum = m_deviceInfo.serialNumber.substr(10); // Remove "realsense_" prefix
            m_config_rs->enable_device(serialNum);
        }
        
        return true;
    } catch (const rs2::error& e) {
        UpdateLastError(std::string("RealSense error: ") + e.what());
        return false;
    }
}

bool RealSenseCameraSource::StartCapture() {
    if (m_isCapturing) {
        return true;
    }
    
    try {
        // Start the pipeline
        auto profile = m_pipeline->start(*m_config_rs);
        
        m_shouldStop = false;
        m_isCapturing = true;
        m_stats.Reset();
        
        // Start capture thread for async frame delivery
        m_captureThread = std::make_unique<std::thread>(&RealSenseCameraSource::CaptureThreadFunc, this);
        
        return true;
    } catch (const rs2::error& e) {
        UpdateLastError(std::string("Failed to start RealSense pipeline: ") + e.what());
        return false;
    }
}

void RealSenseCameraSource::StopCapture() {
    if (!m_isCapturing) {
        return;
    }
    
    m_shouldStop = true;
    m_isCapturing = false;
    
    if (m_captureThread && m_captureThread->joinable()) {
        m_captureThread->join();
        m_captureThread.reset();
    }
    
    try {
        m_pipeline->stop();
    } catch (const rs2::error& e) {
        // Ignore stop errors
    }
}

bool RealSenseCameraSource::IsCapturing() const {
    return m_isCapturing;
}

bool RealSenseCameraSource::CaptureFrame(CameraFrame& frame) {
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
    
    if (m_hasNewFrame && !m_currentRGBFrame.empty()) {
        // Create frame from current RGB data
        frame = CameraFrame::CreateCPUFrame(m_frameWidth, m_frameHeight, 
                                           m_config.format, 
                                           m_currentRGBFrame.data(), 
                                           m_frameWidth * 3); // BGR8 format
        
        // Add depth data if available
        if (!m_currentDepthFrame.empty()) {
            frame.realsense.depthData = reinterpret_cast<const uint8_t*>(m_currentDepthFrame.data());
            frame.realsense.depthWidth = m_depthWidth;
            frame.realsense.depthHeight = m_depthHeight;
        }
        
        m_hasNewFrame = false;
        return frame.IsValid();
    }
    
    UpdateLastError("No frame available");
    return false;
}

void RealSenseCameraSource::SetFrameCallback(FrameCallback callback) {
    m_frameCallback = callback;
}

void RealSenseCameraSource::CaptureThreadFunc() {
    while (!m_shouldStop) {
        try {
            // Wait for frames with timeout
            rs2::frameset frames = m_pipeline->wait_for_frames(100);
            
            if (frames) {
                UpdateFrameRateStats();
                
                // Process RGB frame
                rs2::frame colorFrame = frames.get_color_frame();
                if (colorFrame) {
                    int width = colorFrame.as<rs2::video_frame>().get_width();
                    int height = colorFrame.as<rs2::video_frame>().get_height();
                    int stride = colorFrame.as<rs2::video_frame>().get_stride_in_bytes();
                    
                    // Update frame buffer
                    {
                        std::lock_guard<std::mutex> lock(m_frameMutex);
                        
                        m_frameWidth = width;
                        m_frameHeight = height;
                        
                        size_t dataSize = stride * height;
                        m_currentRGBFrame.resize(dataSize);
                        std::memcpy(m_currentRGBFrame.data(), colorFrame.get_data(), dataSize);
                        
                        // Process depth frame if available
                        if (m_config.enableDepth) {
                            rs2::frame depthFrame = frames.get_depth_frame();
                            if (depthFrame) {
                                m_depthWidth = depthFrame.as<rs2::video_frame>().get_width();
                                m_depthHeight = depthFrame.as<rs2::video_frame>().get_height();
                                
                                size_t depthSize = m_depthWidth * m_depthHeight * sizeof(uint16_t);
                                m_currentDepthFrame.resize(m_depthWidth * m_depthHeight);
                                std::memcpy(m_currentDepthFrame.data(), depthFrame.get_data(), depthSize);
                            }
                        }
                        
                        m_frameTimestamp = std::chrono::steady_clock::now();
                        m_hasNewFrame = true;
                    }
                    m_frameCondition.notify_one();
                    
                    // Call async callback if set
                    if (m_frameCallback) {
                        CameraFrame cameraFrame = ConvertFramesetToFrame(frames);
                        if (cameraFrame.IsValid()) {
                            m_frameCallback(cameraFrame);
                        }
                    }
                }
            }
        } catch (const rs2::error& e) {
            RecordDroppedFrame();
            UpdateLastError(std::string("RealSense capture error: ") + e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

CameraFrame RealSenseCameraSource::ConvertFramesetToFrame(const rs2::frameset& frameset) {
    rs2::frame colorFrame = frameset.get_color_frame();
    if (!colorFrame) {
        return CameraFrame{};
    }
    
    return ConvertRGBFrame(colorFrame);
}

CameraFrame RealSenseCameraSource::ConvertRGBFrame(const rs2::frame& rgbFrame) {
    int width = rgbFrame.as<rs2::video_frame>().get_width();
    int height = rgbFrame.as<rs2::video_frame>().get_height();
    int stride = rgbFrame.as<rs2::video_frame>().get_stride_in_bytes();
    
    CameraFormat format = GetRealSenseFormat(rgbFrame.get_profile().format());
    
    return CameraFrame::CreateCPUFrame(width, height, format,
                                      static_cast<const uint8_t*>(rgbFrame.get_data()),
                                      stride);
}

CameraFormat RealSenseCameraSource::GetRealSenseFormat(int format) {
    switch (format) {
        case RS2_FORMAT_BGR8:
            return CameraFormat::BGR8;
        case RS2_FORMAT_RGB8:
            return CameraFormat::RGB8;
        case RS2_FORMAT_BGRA8:
            return CameraFormat::BGRA8;
        case RS2_FORMAT_RGBA8:
            return CameraFormat::RGBA8;
        default:
            return CameraFormat::BGR8;
    }
}

std::vector<CameraDeviceInfo> RealSenseCameraSource::EnumerateDevices() {
    std::vector<CameraDeviceInfo> devices;
    
    if (!IsRealSenseAvailable()) {
        return devices;
    }
    
    try {
        rs2::context ctx;
        auto deviceList = ctx.query_devices();
        
        for (size_t i = 0; i < deviceList.size(); ++i) {
            rs2::device dev = deviceList[i];
            
            CameraDeviceInfo info;
            info.deviceIndex = static_cast<int>(i);
            info.deviceName = dev.get_info(RS2_CAMERA_INFO_NAME);
            info.serialNumber = "realsense_" + std::string(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
            info.type = CameraSourceType::REALSENSE_DEVICE;
            info.supportsDepth = true;
            
            // Get capabilities from depth sensor
            auto sensors = dev.query_sensors();
            for (auto& sensor : sensors) {
                auto profiles = sensor.get_stream_profiles();
                for (auto& profile : profiles) {
                    if (profile.stream_type() == RS2_STREAM_COLOR) {
                        info.maxWidth = std::max(info.maxWidth, profile.as<rs2::video_stream_profile>().width());
                        info.maxHeight = std::max(info.maxHeight, profile.as<rs2::video_stream_profile>().height());
                        info.maxFrameRate = std::max(info.maxFrameRate, static_cast<double>(profile.fps()));
                    }
                }
            }
            
            devices.push_back(info);
        }
    } catch (const rs2::error& e) {
        // Return empty list on error
    }
    
    return devices;
}

bool RealSenseCameraSource::IsRealSenseAvailable() {
    return true; // Available if compiled with RealSense support
}

#else // !HAVE_REALSENSE

// Stub implementation when RealSense is not available
RealSenseCameraSource::RealSenseCameraSource() {
    m_stats.Reset();
}

RealSenseCameraSource::~RealSenseCameraSource() = default;

bool RealSenseCameraSource::Initialize(const CameraDeviceInfo&, const CameraConfig&) {
    UpdateLastError("RealSense SDK not available (not compiled with HAVE_REALSENSE)");
    return false;
}

bool RealSenseCameraSource::StartCapture() {
    UpdateLastError("RealSense SDK not available");
    return false;
}

void RealSenseCameraSource::StopCapture() {}

bool RealSenseCameraSource::IsCapturing() const { 
    return false; 
}

bool RealSenseCameraSource::CaptureFrame(CameraFrame&) {
    UpdateLastError("RealSense SDK not available");
    return false;
}

void RealSenseCameraSource::SetFrameCallback(FrameCallback) {}

void RealSenseCameraSource::CaptureThreadFunc() {}

CameraFrame RealSenseCameraSource::ConvertFramesetToFrame(const rs2::frameset&) {
    return CameraFrame{};
}

CameraFrame RealSenseCameraSource::ConvertRGBFrame(const rs2::frame&) {
    return CameraFrame{};
}

std::vector<CameraDeviceInfo> RealSenseCameraSource::EnumerateDevices() {
    return {};
}

bool RealSenseCameraSource::IsRealSenseAvailable() {
    return false;
}

#endif // HAVE_REALSENSE

// Common implementation (available regardless of RealSense support)
CameraConfig RealSenseCameraSource::GetConfig() const {
    std::lock_guard<std::mutex> lock(m_configMutex);
    return m_config;
}

bool RealSenseCameraSource::UpdateConfig(const CameraConfig& config) {
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
    bool result = true;
    
#ifdef HAVE_REALSENSE
    result = InitializePipeline();
#endif
    
    if (wasCapturing && result) {
        StartCapture();
    }
    
    return result;
}

CameraSourceType RealSenseCameraSource::GetSourceType() const {
    return CameraSourceType::REALSENSE_DEVICE;
}

CameraDeviceInfo RealSenseCameraSource::GetDeviceInfo() const {
    return m_deviceInfo;
}

CameraStats RealSenseCameraSource::GetStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

void RealSenseCameraSource::ResetStats() {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    m_stats.Reset();
}

std::string RealSenseCameraSource::GetLastError() const {
    return m_lastError;
}

bool RealSenseCameraSource::IsAvailable() const {
#ifdef HAVE_REALSENSE
    return IsRealSenseAvailable();
#else
    return false;
#endif
}

std::string RealSenseCameraSource::GetSourceName() const {
    return "Intel RealSense";
}

bool RealSenseCameraSource::ValidateConfig(const CameraConfig& config) {
    return config.width > 0 && config.height > 0 && config.frameRate > 0;
}

void RealSenseCameraSource::UpdateLastError(const std::string& error) {
    m_lastError = error;
}