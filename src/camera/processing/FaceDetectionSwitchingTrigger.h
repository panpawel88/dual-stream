#pragma once

#include "ICameraFrameListener.h"
#include "../CameraManager.h"
#include "../../video/triggers/ISwitchingTrigger.h"
#include "ui/Window.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <chrono>
#include <atomic>

/**
 * Face detection switching trigger that implements both ISwitchingTrigger and ICameraFrameListener.
 * This class bridges the camera system with the existing video switching system.
 * 
 * Switching Logic:
 * - Single face detected → Switch to Video 1
 * - Multiple faces detected → Switch to Video 2
 * - No faces detected → No switching action
 */
class FaceDetectionSwitchingTrigger : public ISwitchingTrigger, public ICameraFrameListener {
public:
    /**
     * Configuration for face detection behavior
     */
    struct FaceDetectionConfig {
        int minFaceSize = 30;               // Minimum face size in pixels
        int maxFaceSize = 300;              // Maximum face size in pixels
        double scaleFactor = 1.1;           // Scale factor for detection
        int minNeighbors = 3;               // Minimum neighbors for detection
        int stabilityFrames = 5;            // Frames to wait before triggering switch
        double switchCooldownMs = 2000.0;   // Cooldown between switches (ms)
        bool enableVisualization = false;   // Draw face rectangles on frames
        
        // Switching thresholds
        int singleFaceThreshold = 1;        // Faces needed for Video 1
        int multipleFaceThreshold = 2;      // Faces needed for Video 2
    };

    explicit FaceDetectionSwitchingTrigger(const FaceDetectionConfig& config = FaceDetectionConfig());
    ~FaceDetectionSwitchingTrigger() override;
    
    // ISwitchingTrigger interface implementation
    void Update() override;
    bool ShouldSwitchToVideo1() override;
    bool ShouldSwitchToVideo2() override;
    void Reset() override;
    std::string GetName() const override;

    // ICameraFrameListener interface implementation
    FrameProcessingResult ProcessFrame(const CameraFrame& frame) override;
    ListenerPriority GetPriority() const override;
    std::string GetListenerId() const override;
    std::string GetListenerName() const override;
    bool CanProcessFormat(CameraFormat format) const override;
    bool RequiresDepthData() const override;
    FrameProcessingStats GetStats() const override;
    void ResetStats() override;
    void OnRegistered() override;
    void OnUnregistered() override;

/**
     * Initialize face detection with Haar cascade classifier.
     * 
     * @param cascadePath Path to Haar cascade file (empty for default)
     * @return true if initialization successful
     */
    bool InitializeFaceDetection(const std::string& cascadePath = "");
    
    /**
     * Set camera manager for integration.
     * 
     * @param cameraManager Camera manager instance
     */
    void SetCameraManager(CameraManager* cameraManager);
    
    /**
     * Get current face detection configuration.
     * 
     * @return Current configuration
     */
    FaceDetectionConfig GetConfig() const;
    
    /**
     * Update face detection configuration.
     * 
     * @param config New configuration
     */
    void UpdateConfig(const FaceDetectionConfig& config);
    
    /**
     * Get last detected face count.
     * 
     * @return Number of faces detected in last frame
     */
    int GetLastFaceCount() const;
    
    /**
     * Get face detection rectangles from last frame.
     * 
     * @return Vector of face rectangles
     */
    std::vector<cv::Rect> GetLastFaceRects() const;

private:
    mutable std::mutex m_configMutex;
    mutable std::mutex m_detectionMutex;
    
    FaceDetectionConfig m_config;
    CameraManager* m_cameraManager = nullptr;

    // Face detection components
    cv::CascadeClassifier m_faceClassifier;
    bool m_detectionInitialized = false;
    
    // Detection results and state
    std::vector<cv::Rect> m_lastFaceRects;
    int m_lastFaceCount = 0;
    std::chrono::steady_clock::time_point m_lastDetectionTime;
    
    // Switching state management
    mutable std::atomic<bool> m_shouldSwitchToVideo1{false};
    mutable std::atomic<bool> m_shouldSwitchToVideo2{false};
    std::chrono::steady_clock::time_point m_lastSwitchTime;
    
    // Stability tracking
    std::vector<int> m_recentFaceCounts;
    int m_stableFrameCount = 0;
    int m_lastStableFaceCount = 0;
    
    // Performance tracking
    mutable FrameProcessingStats m_stats;
    
    // Private methods
    std::vector<cv::Rect> DetectFaces(const cv::Mat& frame);
    bool IsFrameStable(int faceCount);
    bool ShouldTriggerSwitch(int stableFaceCount);
    void UpdateSwitchingState(int faceCount);
    bool IsInSwitchCooldown() const;
    void ProcessDetectionResult(const std::vector<cv::Rect>& faces);
    cv::Mat PreprocessFrame(const cv::Mat& frame);
    std::string GetDefaultCascadePath();
};

