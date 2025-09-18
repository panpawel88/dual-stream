#pragma once

#include "ICameraSource.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

/**
 * OpenCV-based camera source implementation.
 * Supports webcams and video files through cv::VideoCapture.
 */
class OpenCVCameraSource : public ICameraSource {
public:
    OpenCVCameraSource();
    virtual ~OpenCVCameraSource();
    
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
     * Enumerate available OpenCV camera devices.
     * @return Vector of available camera devices
     */
    static std::vector<CameraDeviceInfo> EnumerateDevices();
    
    /**
     * Create camera source for webcam device.
     * @param deviceIndex Camera device index (0 for first camera)
     * @return Device info structure for the webcam
     */
    static CameraDeviceInfo CreateWebcamDevice(int deviceIndex);
    
    /**
     * Create camera source for video file.
     * @param filePath Path to video file
     * @return Device info structure for the video file
     */
    static CameraDeviceInfo CreateVideoFileDevice(const std::string& filePath);

private:
    cv::VideoCapture m_capture;
    std::unique_ptr<std::thread> m_captureThread;
    mutable std::mutex m_configMutex;
    mutable std::mutex m_statsMutex;
    std::atomic<bool> m_shouldStop{false};
    std::condition_variable m_frameCondition;
    std::mutex m_frameMutex;
    
    // Frame buffer for synchronous capture
    cv::Mat m_currentFrame;
    bool m_hasNewFrame = false;
    std::chrono::steady_clock::time_point m_frameTimestamp;
    
    // Private methods
    bool InitializeCapture();
    bool ConfigureCamera();
    void CaptureThreadFunc();
    std::shared_ptr<CameraFrame> ConvertMatToFrame(const cv::Mat& mat);
    bool ValidateConfig(const CameraConfig& config);
    void UpdateLastError(const std::string& error);
    
    // OpenCV property helpers
    double GetCameraProperty(int propId) const;
    bool SetCameraProperty(int propId, double value);
    
    // Backend conversion and selection helpers
    int ConvertBackendToOpenCV(CameraBackend backend) const;
    CameraBackend GetOptimalBackend() const;
    bool TryOpenWebcam(int deviceIndex, CameraBackend backend);
    bool TryOpenVideoFile(const std::string& filename, CameraBackend backend);
    
    static constexpr int MAX_CAMERA_INDEX = 10;  // Maximum camera index to check
};

