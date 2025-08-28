#include "FaceDetectionSwitchingTrigger.h"
#include "../../core/Logger.h"

FaceDetectionSwitchingTrigger::FaceDetectionSwitchingTrigger(const FaceDetectionConfig& config)
    : m_config(config) {
    m_stats.Reset();
    m_lastDetectionTime = std::chrono::steady_clock::now();
    m_lastSwitchTime = std::chrono::steady_clock::now();
    
    // Initialize recent face counts buffer
    m_recentFaceCounts.reserve(m_config.stabilityFrames);
}

FaceDetectionSwitchingTrigger::~FaceDetectionSwitchingTrigger() {
    // Cleanup will be handled by destructor
}

bool FaceDetectionSwitchingTrigger::InitializeFaceDetection(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    
    switch (m_config.algorithm) {
        case FaceDetectionAlgorithm::HAAR_CASCADE:
            return InitializeHaarCascade(modelPath);
#if defined(HAVE_OPENCV_DNN)
        case FaceDetectionAlgorithm::YUNET:
            return InitializeYuNetDetection(modelPath);
#endif
        default:
            LOG_ERROR("Unknown face detection algorithm");
            return false;
    }
}


void FaceDetectionSwitchingTrigger::Update() {
    // ISwitchingTrigger::Update() - called by main application loop
    // Face detection happens asynchronously through ProcessFrame()
    // This method just needs to maintain switching state
    
    // Check if we need to clear switching flags after they've been consumed
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastSwitch = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastSwitchTime);
    
    // Clear switching flags after a short delay to ensure they're consumed
    if (timeSinceLastSwitch.count() > 100) {
        if (m_shouldSwitchToVideo1.load() || m_shouldSwitchToVideo2.load()) {
            m_shouldSwitchToVideo1 = false;
            m_shouldSwitchToVideo2 = false;
        }
    }
}

bool FaceDetectionSwitchingTrigger::ShouldSwitchToVideo1() {
    return m_shouldSwitchToVideo1.load();
}

bool FaceDetectionSwitchingTrigger::ShouldSwitchToVideo2() {
    return m_shouldSwitchToVideo2.load();
}

FrameProcessingResult FaceDetectionSwitchingTrigger::ProcessFrame(const CameraFrame& frame) {
    if (!m_detectionInitialized) {
        UpdateStats(FrameProcessingResult::SKIPPED, 0.0);
        return FrameProcessingResult::SKIPPED;
    }
    
    auto startTime = std::chrono::steady_clock::now();
    
    try {
        // Convert frame to OpenCV Mat
        if (frame.cpu.mat.empty()) {
            UpdateStats(FrameProcessingResult::FAILED, 0.0);
            return FrameProcessingResult::FAILED;
        }
        
        // Preprocess frame for better detection
        cv::Mat processedFrame = PreprocessFrame(frame.cpu.mat);
        
        // Detect faces
        std::vector<cv::Rect> faces = DetectFaces(processedFrame);
        
        // Process detection results
        ProcessDetectionResult(faces);
        
        auto endTime = std::chrono::steady_clock::now();
        auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        double processingMs = processingTime.count() / 1000.0;
        
        UpdateStats(FrameProcessingResult::SUCCESS, processingMs);
        return FrameProcessingResult::SUCCESS;
        
    } catch (const std::exception& e) {
        auto endTime = std::chrono::steady_clock::now();
        auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        UpdateStats(FrameProcessingResult::FAILED, processingTime.count() / 1000.0);
        return FrameProcessingResult::FAILED;
    }
}

std::vector<cv::Rect> FaceDetectionSwitchingTrigger::DetectFaces(const cv::Mat& frame) {
    switch (m_config.algorithm) {
        case FaceDetectionAlgorithm::HAAR_CASCADE:
            return DetectFacesHaar(frame);
#if defined(HAVE_OPENCV_DNN)
        case FaceDetectionAlgorithm::YUNET:
            return DetectFacesYuNet(frame);
#endif
        default:
            return std::vector<cv::Rect>();
    }
}

cv::Mat FaceDetectionSwitchingTrigger::PreprocessFrame(const cv::Mat& frame) {
    switch (m_config.algorithm) {
        case FaceDetectionAlgorithm::HAAR_CASCADE:
            return PreprocessFrameHaar(frame);
#if defined(HAVE_OPENCV_DNN)
        case FaceDetectionAlgorithm::YUNET:
            return PreprocessFrameYuNet(frame);
#endif
        default:
            return frame.clone();
    }
}

cv::Mat FaceDetectionSwitchingTrigger::PreprocessFrameHaar(const cv::Mat& frame) {
    cv::Mat processed;
    
    // Convert to grayscale for Haar cascade detection
    if (frame.channels() == 3) {
        cv::cvtColor(frame, processed, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, processed, cv::COLOR_BGRA2GRAY);
    } else {
        processed = frame.clone();
    }
    
    // Apply histogram equalization for better detection
    cv::equalizeHist(processed, processed);
    
    return processed;
}

void FaceDetectionSwitchingTrigger::ProcessDetectionResult(const std::vector<cv::Rect>& faces) {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    
    int faceCount = static_cast<int>(faces.size());
    
    // Update detection results
    m_lastFaceRects = faces;
    m_lastFaceCount = faceCount;
    m_lastDetectionTime = std::chrono::steady_clock::now();
    
    // Check frame stability
    if (IsFrameStable(faceCount)) {
        UpdateSwitchingState(faceCount);
    }
    
    // Update recent face counts for stability tracking
    m_recentFaceCounts.push_back(faceCount);
    if (m_recentFaceCounts.size() > static_cast<size_t>(m_config.stabilityFrames)) {
        m_recentFaceCounts.erase(m_recentFaceCounts.begin());
    }
}

bool FaceDetectionSwitchingTrigger::IsFrameStable(int faceCount) {
    if (m_recentFaceCounts.size() < static_cast<size_t>(m_config.stabilityFrames)) {
        return false;
    }
    
    // Check if recent face counts are consistent
    int consistentCount = 0;
    for (int count : m_recentFaceCounts) {
        if (count == faceCount) {
            consistentCount++;
        }
    }
    
    // Require majority of frames to have same face count
    return consistentCount >= (m_config.stabilityFrames * 2 / 3);
}

void FaceDetectionSwitchingTrigger::UpdateSwitchingState(int faceCount) {
    if (IsInSwitchCooldown()) {
        return;
    }
    
    bool shouldTrigger = false;
    bool triggerVideo1 = false;
    
    // Determine switching action based on face count
    if (faceCount >= m_config.multipleFaceThreshold && m_lastStableFaceCount != faceCount) {
        // Multiple faces detected → switch to Video 2
        shouldTrigger = true;
        triggerVideo1 = false;
    } else if (faceCount == m_config.singleFaceThreshold && m_lastStableFaceCount != faceCount) {
        // Single face detected → switch to Video 1
        shouldTrigger = true;
        triggerVideo1 = true;
    }
    
    if (shouldTrigger) {
        if (triggerVideo1) {
            m_shouldSwitchToVideo1 = true;
            m_shouldSwitchToVideo2 = false;
        } else {
            m_shouldSwitchToVideo1 = false;
            m_shouldSwitchToVideo2 = true;
        }
        
        m_lastSwitchTime = std::chrono::steady_clock::now();
        m_lastStableFaceCount = faceCount;
    }
}

bool FaceDetectionSwitchingTrigger::IsInSwitchCooldown() const {
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastSwitch = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastSwitchTime);
    return timeSinceLastSwitch.count() < m_config.switchCooldownMs;
}

ListenerPriority FaceDetectionSwitchingTrigger::GetPriority() const {
    return ListenerPriority::CRITICAL; // High priority for switching triggers
}

std::string FaceDetectionSwitchingTrigger::GetListenerId() const {
    return "face_detection_switching_trigger";
}

std::string FaceDetectionSwitchingTrigger::GetListenerName() const {
    return "Face Detection Switching Trigger";
}

bool FaceDetectionSwitchingTrigger::CanProcessFormat(CameraFormat format) const {
    // Can process any RGB/BGR format
    return format == CameraFormat::BGR8 || format == CameraFormat::RGB8 ||
           format == CameraFormat::BGRA8 || format == CameraFormat::RGBA8 ||
           format == CameraFormat::GRAY8;
}

bool FaceDetectionSwitchingTrigger::RequiresDepthData() const {
    return false; // Face detection works with RGB data only
}

FrameProcessingStats FaceDetectionSwitchingTrigger::GetStats() const {
    return m_stats;
}

void FaceDetectionSwitchingTrigger::ResetStats() {
    m_stats.Reset();
}

void FaceDetectionSwitchingTrigger::OnRegistered() {
    // Called when registered with publisher
}

void FaceDetectionSwitchingTrigger::OnUnregistered() {
    // Called when unregistered from publisher
}

FaceDetectionSwitchingTrigger::FaceDetectionConfig FaceDetectionSwitchingTrigger::GetConfig() const {
    std::lock_guard<std::mutex> lock(m_configMutex);
    return m_config;
}

void FaceDetectionSwitchingTrigger::UpdateConfig(const FaceDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(m_configMutex);
    m_config = config;
    
    // Clear stability buffer when config changes
    m_recentFaceCounts.clear();
    m_stableFrameCount = 0;
}

int FaceDetectionSwitchingTrigger::GetLastFaceCount() const {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    return m_lastFaceCount;
}

std::vector<cv::Rect> FaceDetectionSwitchingTrigger::GetLastFaceRects() const {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    return m_lastFaceRects;
}

std::string FaceDetectionSwitchingTrigger::GetDefaultCascadePath() {
    // Return path to default Haar cascade file in data directory
    return "data/haarcascades/haarcascade_frontalface_alt.xml";
}

#if defined(HAVE_OPENCV_DNN)
std::string FaceDetectionSwitchingTrigger::GetDefaultYuNetModelPath() {
    // Return path to default YuNet model file in data directory
    return "data/yunet/face_detection_yunet_2023mar.onnx";
}
#endif

bool FaceDetectionSwitchingTrigger::InitializeHaarCascade(const std::string& cascadePath) {
    std::string actualPath = cascadePath.empty() ? GetDefaultCascadePath() : cascadePath;
    
    if (!m_faceClassifier.load(actualPath)) {
        LOG_DEBUG("Failed to load cascade from primary path: ", actualPath);
        
        // Try fallback cascade paths (in order of preference)
        std::vector<std::string> fallbackPaths = {
            "data/haarcascades/haarcascade_frontalface_alt.xml",
            "data/haarcascades/haarcascade_frontalface_default.xml",
            "data/haarcascades/haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt.xml",              // Legacy fallback
            "haarcascade_frontalface_default.xml"           // Legacy fallback
        };
        
        bool loaded = false;
        for (const auto& fallback : fallbackPaths) {
            LOG_DEBUG("Attempting to load cascade from: ", fallback);
            if (m_faceClassifier.load(fallback)) {
                LOG_INFO("Successfully loaded face detection cascade: ", fallback);
                loaded = true;
                break;
            }
        }
        
        if (!loaded) {
            LOG_ERROR("Failed to load any face detection cascade file. Tried paths:");
            LOG_ERROR("  Primary: ", actualPath);
            for (const auto& path : fallbackPaths) {
                LOG_ERROR("  Fallback: ", path);
            }
            LOG_ERROR("Ensure face detection models are downloaded by running CMake with DOWNLOAD_FACE_DETECTION_MODELS=ON");
            return false;
        }
    } else {
        LOG_INFO("Successfully loaded face detection cascade: ", actualPath);
    }
    
    m_detectionInitialized = true;
    return true;
}

#if defined(HAVE_OPENCV_DNN)
bool FaceDetectionSwitchingTrigger::InitializeYuNetDetection(const std::string& modelPath) {
    std::string actualPath = modelPath.empty() ? GetDefaultYuNetModelPath() : modelPath;
    
    try {
        // Create YuNet face detector
        m_yunetDetector = cv::FaceDetectorYN::create(
            actualPath,
            "",  // config (empty for default)
            m_config.inputSize,
            m_config.scoreThreshold,
            m_config.nmsThreshold
        );
        
        if (m_yunetDetector.empty()) {
            LOG_ERROR("Failed to create YuNet face detector from: ", actualPath);
            
            // Try fallback paths
            std::vector<std::string> fallbackPaths = {
                "data/yunet/face_detection_yunet_2023mar.onnx",
                "face_detection_yunet_2023mar.onnx"  // Legacy fallback
            };
            
            bool loaded = false;
            for (const auto& fallback : fallbackPaths) {
                LOG_DEBUG("Attempting to load YuNet model from: ", fallback);
                try {
                    m_yunetDetector = cv::FaceDetectorYN::create(
                        fallback,
                        "",
                        m_config.inputSize,
                        m_config.scoreThreshold,
                        m_config.nmsThreshold
                    );
                    
                    if (!m_yunetDetector.empty()) {
                        LOG_INFO("Successfully loaded YuNet face detector: ", fallback);
                        loaded = true;
                        break;
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG("Failed to load YuNet model from ", fallback, ": ", e.what());
                }
            }
            
            if (!loaded) {
                LOG_ERROR("Failed to load any YuNet model file. Tried paths:");
                LOG_ERROR("  Primary: ", actualPath);
                for (const auto& path : fallbackPaths) {
                    LOG_ERROR("  Fallback: ", path);
                }
                LOG_ERROR("Ensure YuNet model is downloaded by running CMake with DOWNLOAD_FACE_DETECTION_MODELS=ON");
                return false;
            }
        } else {
            LOG_INFO("Successfully loaded YuNet face detector: ", actualPath);
        }
        
        m_detectionInitialized = true;
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during YuNet initialization: ", e.what());
        return false;
    }
}

std::vector<cv::Rect> FaceDetectionSwitchingTrigger::DetectFacesHaar(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    
    std::vector<cv::Rect> faces;
    
    if (m_faceClassifier.empty()) {
        return faces;
    }
    
    m_faceClassifier.detectMultiScale(
        frame,
        faces,
        m_config.scaleFactor,
        m_config.minNeighbors,
        0,
        cv::Size(m_config.minFaceSize, m_config.minFaceSize),
        cv::Size(m_config.maxFaceSize, m_config.maxFaceSize)
    );
    
    return faces;
}

std::vector<cv::Rect> FaceDetectionSwitchingTrigger::DetectFacesYuNet(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    
    std::vector<cv::Rect> faces;
    
    if (m_yunetDetector.empty()) {
        return faces;
    }
    
    try {
        // Set input size for this frame
        m_yunetDetector->setInputSize(cv::Size(frame.cols, frame.rows));
        
        // Detect faces
        cv::Mat detectionResult;
        m_yunetDetector->detect(frame, detectionResult);
        
        // Convert detection results to cv::Rect format
        if (detectionResult.rows > 0) {
            for (int i = 0; i < detectionResult.rows; ++i) {
                // YuNet returns [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
                // We only need the bounding box: [x, y, w, h]
                float x = detectionResult.at<float>(i, 0);
                float y = detectionResult.at<float>(i, 1);
                float w = detectionResult.at<float>(i, 2);
                float h = detectionResult.at<float>(i, 3);
                
                // Convert to cv::Rect
                cv::Rect faceRect(
                    static_cast<int>(x),
                    static_cast<int>(y),
                    static_cast<int>(w),
                    static_cast<int>(h)
                );
                
                faces.push_back(faceRect);
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during YuNet face detection: ", e.what());
    }
    
    return faces;
}

cv::Mat FaceDetectionSwitchingTrigger::PreprocessFrameYuNet(const cv::Mat& frame) {
    // YuNet can work with color images directly, no need for grayscale conversion
    // Just ensure we have the right format (BGR)
    cv::Mat processed;
    
    if (frame.channels() == 4) {
        // Convert BGRA to BGR
        cv::cvtColor(frame, processed, cv::COLOR_BGRA2BGR);
    } else if (frame.channels() == 1) {
        // Convert grayscale to BGR
        cv::cvtColor(frame, processed, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 3) {
        // Already BGR, just clone
        processed = frame.clone();
    } else {
        // Fallback
        processed = frame.clone();
    }
    
    return processed;
}
#endif

void FaceDetectionSwitchingTrigger::Reset() {
    // Reset switching state flags after a switch has occurred
    m_shouldSwitchToVideo1.store(false);
    m_shouldSwitchToVideo2.store(false);
}

std::string FaceDetectionSwitchingTrigger::GetName() const {
    return "FaceDetection";
}