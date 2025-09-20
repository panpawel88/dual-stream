#pragma once

#include "ICameraSource.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <map>

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

    // Runtime property control
    bool SetCameraProperty(CameraPropertyType property, double value) override;
    bool GetCameraProperty(CameraPropertyType property, double& value) const override;
    bool SetCameraProperties(const CameraProperties& properties) override;
    CameraProperties GetCameraProperties() const override;
    std::set<CameraPropertyType> GetSupportedProperties() const override;
    
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
    mutable std::mutex m_propertyMutex;
    std::atomic<bool> m_shouldStop{false};
    std::atomic<bool> m_hasPendingProperties{false};
    std::condition_variable m_frameCondition;
    std::mutex m_frameMutex;
    
    // Frame buffer for synchronous capture
    cv::Mat m_currentFrame;
    bool m_hasNewFrame = false;
    std::chrono::steady_clock::time_point m_frameTimestamp;

    // Property management
    CameraProperties m_pendingProperties;
    CameraProperties m_currentProperties;
    std::chrono::steady_clock::time_point m_lastPropertyUpdate;
    static constexpr int PROPERTY_UPDATE_INTERVAL_MS = 100; // Limit property updates to avoid FPS drops

    // Property range information (internal implementation detail)
    struct CameraPropertyRange {
        int min = 0;
        int max = 100;
        int defaultValue = 50;
        int step = 1;
        bool supported = false;

        CameraPropertyRange() = default;
        CameraPropertyRange(int minVal, int maxVal, int defVal, int stepVal = 1, bool isSupported = true)
            : min(minVal), max(maxVal), defaultValue(defVal), step(stepVal), supported(isSupported) {}
    };

    // Property range cache (mutable for const methods)
    mutable std::map<int, CameraPropertyRange> m_propertyRangeCache;
    
    // Private methods
    bool InitializeCapture();
    bool ConfigureCamera();
    void CaptureThreadFunc();
    std::shared_ptr<CameraFrame> ConvertMatToFrame(const cv::Mat& mat);
    bool ValidateConfig(const CameraConfig& config);
    void UpdateLastError(const std::string& error);
    
    // OpenCV property helpers
    double GetOpenCVProperty(int propId) const;
    bool SetOpenCVProperty(int propId, double value);

    // Property range detection and conversion (internal use only)
    OpenCVCameraSource::CameraPropertyRange DetectPropertyRange(int openCVPropId) const;
    double ConvertToNormalizedValue(double cameraValue, const CameraPropertyRange& range) const;
    double ConvertFromNormalizedValue(double normalizedValue, const CameraPropertyRange& range) const;

    // Property management helpers
    void ConfigureProperty(CameraPropertyType property, double configValue, double& currentValue);
    void ApplyPendingProperties();
    int ConvertPropertyTypeToOpenCV(CameraPropertyType property) const;
    bool ValidatePropertyValue(CameraPropertyType property, double value) const;
    
    // Backend conversion and selection helpers
    int ConvertBackendToOpenCV(CameraBackend backend) const;
    CameraBackend GetOptimalBackend() const;
    bool TryOpenWebcam(int deviceIndex, CameraBackend backend);
    bool TryOpenVideoFile(const std::string& filename, CameraBackend backend);
    
    static constexpr int MAX_CAMERA_INDEX = 10;  // Maximum camera index to check
};

