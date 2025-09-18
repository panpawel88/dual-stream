#pragma once

#include "ICameraSource.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

// Forward declarations to avoid requiring RealSense headers in this header
namespace rs2 {
    class pipeline;
    class config;
    class frameset;
    class frame;
    class context;
}

/**
 * Intel RealSense camera source implementation.
 * Supports RGB and depth capture from RealSense devices.
 */
class RealSenseCameraSource : public ICameraSource {
public:
    RealSenseCameraSource();
    virtual ~RealSenseCameraSource();
    
    // ICameraSource interface implementation
    bool Initialize(const CameraDeviceInfo& deviceInfo, const CameraConfig& config) override;
    bool StartCapture() override;
    void StopCapture() override;
    bool IsCapturing() const override;
    std::shared_ptr<CameraFrame> CaptureFrame() override;
    void SetFrameCallback(FrameCallback callback) override;
    CameraConfig GetConfig() const override;
    bool UpdateConfig(const CameraConfig& config) override;
    CameraSourceType GetSourceType() const override;
    CameraDeviceInfo GetDeviceInfo() const override;
    CameraStats GetStats() const override;
    void ResetStats() override;
    std::string GetLastError() const override;
    bool IsAvailable() const override;
    std::string GetSourceName() const override;
    
    /**
     * Enumerate available RealSense devices.
     * @return Vector of available RealSense devices
     */
    static std::vector<CameraDeviceInfo> EnumerateDevices();
    
    /**
     * Check if RealSense SDK is available.
     * @return true if RealSense SDK is available
     */
    static bool IsRealSenseAvailable();

private:
    // RealSense pipeline objects (using unique_ptr to avoid header dependency)
    std::unique_ptr<rs2::pipeline> m_pipeline;
    std::unique_ptr<rs2::config> m_config_rs;
    std::unique_ptr<rs2::context> m_context;
    
    std::unique_ptr<std::thread> m_captureThread;
    mutable std::mutex m_configMutex;
    mutable std::mutex m_statsMutex;
    std::atomic<bool> m_shouldStop{false};
    std::condition_variable m_frameCondition;
    std::mutex m_frameMutex;
    
    // Frame buffer for synchronous capture
    std::vector<uint8_t> m_currentRGBFrame;
    std::vector<uint16_t> m_currentDepthFrame;
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    int m_depthWidth = 0;
    int m_depthHeight = 0;
    bool m_hasNewFrame = false;
    std::chrono::steady_clock::time_point m_frameTimestamp;
    
    // Private methods
    bool InitializePipeline();
    void CaptureThreadFunc();
    std::shared_ptr<CameraFrame> ConvertFramesetToFrame(const rs2::frameset& frameset);
    std::shared_ptr<CameraFrame> ConvertRGBFrame(const rs2::frame& rgbFrame);
    bool ValidateConfig(const CameraConfig& config);
    void UpdateLastError(const std::string& error);
    CameraFormat GetRealSenseFormat(int format);
};

