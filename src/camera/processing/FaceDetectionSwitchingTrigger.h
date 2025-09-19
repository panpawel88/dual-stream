#pragma once

#include "ICameraFrameListener.h"
#include "../CameraManager.h"
#include "../../video/triggers/ISwitchingTrigger.h"
#include "ui/Window.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>

#if defined(HAVE_OPENCV_DNN)
#include <opencv2/dnn.hpp>
#endif
#include <vector>
#include <chrono>
#include <atomic>
#include <queue>

/**
 * Face detection switching trigger that implements both ISwitchingTrigger and ICameraFrameListener.
 * This class bridges the camera system with the existing video switching system.
 * 
 * Switching Logic:
 * - No faces detected (0) → Switch to Video 1 (index 0)
 * - Single face detected (1) → Switch to Video 2 (index 1)
 * - Multiple faces detected (2+) → Switch to Video 2 (index 1)
 * 
 * Preview Feature:
 * When enablePreview is set to true, displays a real-time preview window showing:
 * - Live camera feed with face detection rectangles
 * - Face count and switching status overlay
 * - Algorithm information
 * 
 * Usage Example:
 * ```cpp
 * FaceDetectionSwitchingTrigger::FaceDetectionConfig config;
 * config.enablePreview = true;
 * config.algorithm = FaceDetectionAlgorithm::HAAR_CASCADE;
 * 
 * auto trigger = std::make_shared<FaceDetectionSwitchingTrigger>(config);
 * trigger->InitializeFaceDetection();
 * 
 * // To enable/disable preview at runtime:
 * trigger->SetPreviewEnabled(true);
 * ```
 */
/**
 * Face detection algorithm types
 */
enum class FaceDetectionAlgorithm {
    HAAR_CASCADE,  // Traditional Haar cascade classifier
#if defined(HAVE_OPENCV_DNN)
    YUNET         // Modern YuNet face detector using DNN
#endif
};

class FaceDetectionSwitchingTrigger : public ISwitchingTrigger, public ICameraFrameListener {
public:
    /**
     * Configuration for face detection behavior
     */
    struct FaceDetectionConfig {
        FaceDetectionAlgorithm algorithm = FaceDetectionAlgorithm::HAAR_CASCADE;
        
        // Common parameters
        int stabilityFrames = 5;            // Frames to wait before triggering switch
        double switchCooldownMs = 2000.0;   // Cooldown between switches (ms)
        bool enableVisualization = false;   // Draw face rectangles on frames
        bool enablePreview = false;         // Show face detection preview window
        
        // Video switching thresholds
        int video1FaceCount = 1;            // Face count that triggers switch to Video 1
        int video2FaceCount = 2;            // Face count that triggers switch to Video 2
        
        // Haar Cascade specific parameters
        int minFaceSize = 30;               // Minimum face size in pixels
        int maxFaceSize = 300;              // Maximum face size in pixels
        double scaleFactor = 1.1;           // Scale factor for detection
        int minNeighbors = 3;               // Minimum neighbors for detection
        
#if defined(HAVE_OPENCV_DNN)
        // YuNet specific parameters
        float scoreThreshold = 0.9f;        // Confidence score threshold
        float nmsThreshold = 0.3f;          // Non-maximum suppression threshold
        cv::Size inputSize = cv::Size(320, 320); // Input size for YuNet model
#endif
    };

    explicit FaceDetectionSwitchingTrigger(const FaceDetectionConfig& config = FaceDetectionConfig());

    /**
     * Create a FaceDetectionConfig populated from the global configuration system
     * @return FaceDetectionConfig with values from Config::GetInstance()
     */
    static FaceDetectionConfig CreateConfigFromGlobal();
    ~FaceDetectionSwitchingTrigger() override;
    
    // ISwitchingTrigger interface implementation
    void Update() override;
    std::optional<size_t> GetTargetVideoIndex() override;
    void Reset() override;
    std::string GetName() const override;

    // ICameraFrameListener interface implementation
    FrameProcessingResult ProcessFrame(std::shared_ptr<const CameraFrame> frame) override;
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
     * Initialize face detection with the configured algorithm.
     * 
     * @param modelPath Path to model file (empty for default)
     * @return true if initialization successful
     */
    bool InitializeFaceDetection(const std::string& modelPath = "");
    
    
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
    
    /**
     * Enable/disable face detection preview window.
     * 
     * @param enable Whether to show preview window
     * @return true if preview was successfully enabled/disabled
     */
    bool SetPreviewEnabled(bool enable);
    
    /**
     * Check if preview window is currently enabled.
     * 
     * @return true if preview is enabled
     */
    bool IsPreviewEnabled() const;
    
    /**
     * Update preview window from main thread.
     * This method safely processes queued preview frames using OpenCV GUI functions.
     * MUST be called from the main thread to avoid Win32 message pump conflicts.
     */
    void UpdatePreviewMainThread();

private:
    mutable std::mutex m_configMutex;
    mutable std::mutex m_detectionMutex;
    
    FaceDetectionConfig m_config;

    // Face detection components
    cv::CascadeClassifier m_faceClassifier;          // Haar cascade detector
#if defined(HAVE_OPENCV_DNN)
    cv::Ptr<cv::FaceDetectorYN> m_yunetDetector;     // YuNet detector
#endif
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
    
    // Preview window state
    bool m_previewEnabled = false;
    std::string m_previewWindowName;
    mutable std::mutex m_previewMutex;
    
    // Thread-safe preview frame queue for main thread processing
    std::queue<cv::Mat> m_previewFrameQueue;
    mutable std::mutex m_previewQueueMutex;
    static constexpr size_t MAX_PREVIEW_QUEUE_SIZE = 3;
    
    // Private methods
    std::vector<cv::Rect> DetectFaces(const cv::Mat& frame);
    std::vector<cv::Rect> DetectFacesHaar(const cv::Mat& frame);
#if defined(HAVE_OPENCV_DNN)
    std::vector<cv::Rect> DetectFacesYuNet(const cv::Mat& frame);
    bool InitializeYuNetDetection(const std::string& modelPath);
    cv::Mat PreprocessFrameYuNet(const cv::Mat& frame);
    std::string GetDefaultYuNetModelPath();
#endif
    bool InitializeHaarCascade(const std::string& cascadePath);
    bool IsFrameStable(int faceCount);
    bool ShouldTriggerSwitch(int stableFaceCount);
    void UpdateSwitchingState(int faceCount);
    bool IsInSwitchCooldown() const;
    void ProcessDetectionResult(const std::vector<cv::Rect>& faces);
    cv::Mat PreprocessFrame(const cv::Mat& frame);
    cv::Mat PreprocessFrameHaar(const cv::Mat& frame);
    std::string GetDefaultCascadePath();
    
    // Preview functionality
    void InitializePreview();
    void DestroyPreview();
    void UpdatePreview(const cv::Mat& frame, const std::vector<cv::Rect>& faces);
    cv::Mat CreatePreviewFrame(const cv::Mat& frame, const std::vector<cv::Rect>& faces);
    
    // Thread-safe preview frame queueing
    void QueuePreviewFrame(const cv::Mat& previewFrame);
};

