#include "FaceDetectionSwitchingTrigger.h"
#include "../../core/Logger.h"
#include "../../core/Config.h"
#include <fstream>

FaceDetectionSwitchingTrigger::FaceDetectionConfig FaceDetectionSwitchingTrigger::CreateConfigFromGlobal() {
    Config* config = Config::GetInstance();
    FaceDetectionConfig faceConfig;
    
    // Parse algorithm from string
    std::string algorithmStr = config->GetString("face_detection.algorithm", "haar_cascade");
    if (algorithmStr == "yunet") {
        faceConfig.algorithm = FaceDetectionAlgorithm::YUNET;
    } else if (algorithmStr == "centerface") {
        faceConfig.algorithm = FaceDetectionAlgorithm::CENTERFACE;
    } else {
        faceConfig.algorithm = FaceDetectionAlgorithm::HAAR_CASCADE;
    }
    
    // Common parameters
    faceConfig.stabilityFrames = config->GetInt("face_detection.stability_frames", 5);
    faceConfig.switchCooldownMs = config->GetDouble("face_detection.switch_cooldown_ms", 2000.0);
    faceConfig.enableVisualization = config->GetBool("face_detection.enable_visualization", false);
    faceConfig.enablePreview = config->GetBool("face_detection.enable_preview", false);
    
    // Video switching thresholds
    faceConfig.video1FaceCount = config->GetInt("face_detection.video1_face_count", 1);
    faceConfig.video2FaceCount = config->GetInt("face_detection.video2_face_count", 2);
    
    // Haar Cascade specific parameters
    faceConfig.minFaceSize = config->GetInt("face_detection.min_face_size", 30);
    faceConfig.maxFaceSize = config->GetInt("face_detection.max_face_size", 300);
    faceConfig.scaleFactor = config->GetDouble("face_detection.scale_factor", 1.1);
    faceConfig.minNeighbors = config->GetInt("face_detection.min_neighbors", 3);
    
#if defined(HAVE_OPENCV_DNN)
    // YuNet specific parameters
    faceConfig.scoreThreshold = config->GetFloat("face_detection.score_threshold", 0.9f);
    faceConfig.nmsThreshold = config->GetFloat("face_detection.nms_threshold", 0.3f);
    int inputWidth = config->GetInt("face_detection.input_width", 320);
    int inputHeight = config->GetInt("face_detection.input_height", 320);
    faceConfig.inputSize = cv::Size(inputWidth, inputHeight);

    // CenterFace specific parameters
    int centerfaceInputWidth = config->GetInt("face_detection.centerface_input_width", 640);
    int centerfaceInputHeight = config->GetInt("face_detection.centerface_input_height", 640);
    faceConfig.centerfaceInputSize = cv::Size(centerfaceInputWidth, centerfaceInputHeight);
    faceConfig.centerfaceScoreThreshold = config->GetFloat("face_detection.centerface_score_threshold", 0.5f);
    faceConfig.centerfaceNmsThreshold = config->GetFloat("face_detection.centerface_nms_threshold", 0.2f);
#endif
    
    return faceConfig;
}

FaceDetectionSwitchingTrigger::FaceDetectionSwitchingTrigger(const FaceDetectionConfig& config)
    : m_config(config) {
    m_stats.Reset();
    m_lastDetectionTime = std::chrono::steady_clock::now();
    m_lastSwitchTime = std::chrono::steady_clock::now();
    m_previewWindowName = "Face Detection Preview - " + GetListenerId();
    
    // Initialize recent face counts buffer
    m_recentFaceCounts.reserve(m_config.stabilityFrames);
    
    // Initialize preview if enabled in config
    if (m_config.enablePreview) {
        SetPreviewEnabled(true);
    }
}

FaceDetectionSwitchingTrigger::~FaceDetectionSwitchingTrigger() {
    // Cleanup preview window
    if (m_previewEnabled) {
        DestroyPreview();
    }
}

bool FaceDetectionSwitchingTrigger::InitializeFaceDetection(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(m_detectionMutex);
    
    switch (m_config.algorithm) {
        case FaceDetectionAlgorithm::HAAR_CASCADE:
            return InitializeHaarCascade(modelPath);
#if defined(HAVE_OPENCV_DNN)
        case FaceDetectionAlgorithm::YUNET:
            return InitializeYuNetDetection(modelPath);
        case FaceDetectionAlgorithm::CENTERFACE:
            return InitializeCenterFaceDetection(modelPath);
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
    
    // Update face detection preview from main thread (safe for OpenCV GUI operations)
    if (m_previewEnabled) {
        UpdatePreviewMainThread();
    }
    
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

std::optional<size_t> FaceDetectionSwitchingTrigger::GetTargetVideoIndex() {
    // As specified, face detection is limited to switching between videos 0 and 1
    if (m_shouldSwitchToVideo1.load()) {
        return size_t(0); // Switch to video index 0 (video 1)
    }
    if (m_shouldSwitchToVideo2.load()) {
        return size_t(1); // Switch to video index 1 (video 2)
    }
    return std::nullopt; // No switch requested
}

FrameProcessingResult FaceDetectionSwitchingTrigger::ProcessFrame(std::shared_ptr<const CameraFrame> frame) {
    if (!m_detectionInitialized || !frame) {
        UpdateStats(FrameProcessingResult::SKIPPED, 0.0);
        return FrameProcessingResult::SKIPPED;
    }

    auto startTime = std::chrono::steady_clock::now();

    try {
        // Convert frame to OpenCV Mat
        if (frame->mat.empty()) {
            UpdateStats(FrameProcessingResult::FAILED, 0.0);
            return FrameProcessingResult::FAILED;
        }
        
        // Preprocess frame for better detection
        cv::Mat processedFrame = PreprocessFrame(frame->mat);
        
        // Detect faces
        std::vector<cv::Rect> faces = DetectFaces(processedFrame);
        
        // Process detection results
        ProcessDetectionResult(faces);
        
        // Update preview window if enabled (use original frame for consistent brightness)
        if (m_previewEnabled) {
            UpdatePreview(frame->mat, faces);
        }
        
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
        case FaceDetectionAlgorithm::CENTERFACE:
            return DetectFacesCenterFace(frame);
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
        case FaceDetectionAlgorithm::CENTERFACE:
            return PreprocessFrameCenterFace(frame);
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
    if (faceCount == 0 && m_lastStableFaceCount != faceCount) {
        // No faces detected → switch to Video 1 (index 0)
        shouldTrigger = true;
        triggerVideo1 = true;
    } else if (faceCount == m_config.video1FaceCount && m_lastStableFaceCount != faceCount) {
        // Single face detected → switch to Video 2 (index 1)
        shouldTrigger = true;
        triggerVideo1 = false;
    } else if (faceCount >= m_config.video2FaceCount && m_lastStableFaceCount != faceCount) {
        // Multiple faces detected → switch to Video 2 (index 1)
        shouldTrigger = true;
        triggerVideo1 = false;
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
    bool previewChanged = (m_config.enablePreview != config.enablePreview);
    m_config = config;
    
    // Clear stability buffer when config changes
    m_recentFaceCounts.clear();
    m_stableFrameCount = 0;
    
    // Handle preview state change
    if (previewChanged) {
        SetPreviewEnabled(m_config.enablePreview);
    }
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

std::string FaceDetectionSwitchingTrigger::GetDefaultCenterFaceModelPath() {
    // Return path to default CenterFace model file in data directory
    return "data/centerface/centerface.onnx";
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
    
    // Validate that the model file exists and is not empty
    std::ifstream modelFile(actualPath, std::ios::binary | std::ios::ate);
    if (!modelFile.is_open()) {
        LOG_ERROR("YuNet model file does not exist: ", actualPath);
        return false;
    }
    
    std::streamsize fileSize = modelFile.tellg();
    modelFile.close();
    
    if (fileSize == 0) {
        LOG_ERROR("YuNet model file is empty (0 bytes): ", actualPath);
        LOG_ERROR("The model may not have been downloaded correctly. Try cleaning build directory and rebuilding.");
        return false;
    }
    
    if (fileSize < 1024) {  // Expect at least 1KB for a valid ONNX model
        LOG_ERROR("YuNet model file is too small (", fileSize, " bytes): ", actualPath);
        LOG_ERROR("The file appears to be corrupted. Try cleaning build directory and rebuilding.");
        return false;
    }
    
    LOG_DEBUG("Loading YuNet model file (", fileSize, " bytes): ", actualPath);
    
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
            LOG_ERROR("Ensure YuNet model is downloaded by running CMake with DOWNLOAD_FACE_DETECTION_MODELS=ON");
            return false;
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

bool FaceDetectionSwitchingTrigger::InitializeCenterFaceDetection(const std::string& modelPath) {
    std::string actualPath = modelPath.empty() ? GetDefaultCenterFaceModelPath() : modelPath;

    // Validate that the model file exists and is not empty
    std::ifstream modelFile(actualPath, std::ios::binary | std::ios::ate);
    if (!modelFile.is_open()) {
        LOG_ERROR("CenterFace model file does not exist: ", actualPath);
        return false;
    }

    std::streamsize fileSize = modelFile.tellg();
    modelFile.close();

    if (fileSize == 0) {
        LOG_ERROR("CenterFace model file is empty (0 bytes): ", actualPath);
        LOG_ERROR("The model may not have been downloaded correctly. Try downloading centerface.onnx model.");
        return false;
    }

    try {
        // Load the ONNX model using OpenCV DNN
        m_centerfaceNet = cv::dnn::readNetFromONNX(actualPath);

        if (m_centerfaceNet.empty()) {
            LOG_ERROR("Failed to load CenterFace model from: ", actualPath);
            return false;
        }

        // Set backend and target for inference
        m_centerfaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_centerfaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        LOG_INFO("Successfully loaded CenterFace model: ", actualPath);
        m_detectionInitialized = true;
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Exception during CenterFace initialization: ", e.what());
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

std::vector<cv::Rect> FaceDetectionSwitchingTrigger::DetectFacesCenterFace(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_detectionMutex);

    std::vector<cv::Rect> faces;

    if (m_centerfaceNet.empty()) {
        return faces;
    }

    try {
        // Preprocess frame for CenterFace
        cv::Mat preprocessed = PreprocessFrameCenterFace(frame);

        // Create blob from image
        cv::Mat blob;
        cv::dnn::blobFromImage(preprocessed, blob, 1.0, m_config.centerfaceInputSize, cv::Scalar(0, 0, 0), true, false);

        // Set input to the network
        m_centerfaceNet.setInput(blob);

        // Run forward pass
        std::vector<cv::Mat> outputs;
        m_centerfaceNet.forward(outputs, m_centerfaceNet.getUnconnectedOutLayersNames());

        // Post-process outputs to get bounding boxes
        faces = PostprocessCenterFaceOutput(outputs, frame.cols, frame.rows);

    } catch (const std::exception& e) {
        LOG_ERROR("Exception during CenterFace detection: ", e.what());
        return faces;
    }

    return faces;
}

cv::Mat FaceDetectionSwitchingTrigger::PreprocessFrameCenterFace(const cv::Mat& frame) {
    cv::Mat processed;

    // CenterFace expects BGR format
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

std::vector<cv::Rect> FaceDetectionSwitchingTrigger::PostprocessCenterFaceOutput(const std::vector<cv::Mat>& outputs, int originalWidth, int originalHeight) {
    std::vector<cv::Rect> faces;

    if (outputs.size() < 4) {
        LOG_ERROR("CenterFace model should output 4 tensors (heatmap, scale, offset, landmarks), got: ", outputs.size());
        return faces;
    }

    const cv::Mat& heatmap = outputs[0];  // (1, 1, H/4, W/4) - face center heatmap
    const cv::Mat& scale = outputs[1];    // (1, 2, H/4, W/4) - bbox scales
    const cv::Mat& offset = outputs[2];   // (1, 2, H/4, W/4) - center offsets
    // const cv::Mat& landmarks = outputs[3]; // (1, 10, H/4, W/4) - facial landmarks (not used for face count)

    float scoreThreshold = m_config.centerfaceScoreThreshold;
    float nmsThreshold = m_config.centerfaceNmsThreshold;

    // Calculate scaling factors
    float scaleX = static_cast<float>(originalWidth) / m_config.centerfaceInputSize.width;
    float scaleY = static_cast<float>(originalHeight) / m_config.centerfaceInputSize.height;

    // Extract face candidates from heatmap
    std::vector<cv::Rect> candidateBoxes;
    std::vector<float> confidences;

    // Access heatmap data (assuming NCHW format: N=1, C=1, H=height/4, W=width/4)
    const float* heatmapData = (const float*)heatmap.data;
    const float* scaleData = (const float*)scale.data;
    const float* offsetData = (const float*)offset.data;

    int heatmapHeight = heatmap.size[2];
    int heatmapWidth = heatmap.size[3];

    for (int y = 0; y < heatmapHeight; y++) {
        for (int x = 0; x < heatmapWidth; x++) {
            int index = y * heatmapWidth + x;
            float confidence = heatmapData[index];

            if (confidence > scoreThreshold) {
                // Get scale and offset for this position
                // Based on reference implementation: scale tensor has [height, width] channels
                float scaleH = std::exp(scaleData[index]) * 4.0f; // Scale height (first channel)
                float scaleW = std::exp(scaleData[heatmapHeight * heatmapWidth + index]) * 4.0f; // Scale width (second channel)
                // Offset tensor has [y_offset, x_offset] channels
                float offsetY = offsetData[index]; // Y offset (first channel)
                float offsetX = offsetData[heatmapHeight * heatmapWidth + index]; // X offset (second channel)

                // Calculate center position (downsampling factor is 4, +0.5 for pixel shift)
                float centerX = (x + offsetX + 0.5f) * 4.0f;
                float centerY = (y + offsetY + 0.5f) * 4.0f;

                // Calculate bounding box size (already transformed with exp)
                float width = scaleW;
                float height = scaleH;

                // Convert to bounding box coordinates
                float x1 = centerX - width / 2.0f;
                float y1 = centerY - height / 2.0f;
                float x2 = centerX + width / 2.0f;
                float y2 = centerY + height / 2.0f;

                // Scale back to original image coordinates
                x1 *= scaleX;
                y1 *= scaleY;
                x2 *= scaleX;
                y2 *= scaleY;

                // Clamp to image boundaries
                x1 = std::max(0.0f, std::min(x1, (float)originalWidth));
                y1 = std::max(0.0f, std::min(y1, (float)originalHeight));
                x2 = std::max(0.0f, std::min(x2, (float)originalWidth));
                y2 = std::max(0.0f, std::min(y2, (float)originalHeight));

                // Create rectangle
                cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1),
                             static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));

                // Validate bounding box
                if (bbox.width > 10 && bbox.height > 10) {
                    candidateBoxes.push_back(bbox);
                    confidences.push_back(confidence);
                }
            }
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(candidateBoxes, confidences, scoreThreshold, nmsThreshold, indices);

    // Extract final detections
    for (int idx : indices) {
        faces.push_back(candidateBoxes[idx]);
    }

    return faces;
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

bool FaceDetectionSwitchingTrigger::SetPreviewEnabled(bool enable) {
    std::lock_guard<std::mutex> lock(m_previewMutex);
    
    if (enable == m_previewEnabled) {
        return true; // Already in desired state
    }
    
    if (enable) {
        InitializePreview();
        m_previewEnabled = true;
        LOG_INFO("Face detection preview enabled for: ", GetListenerName());
    } else {
        DestroyPreview();
        m_previewEnabled = false;
        
        // Clear preview queue when disabling
        {
            std::lock_guard<std::mutex> lock(m_previewQueueMutex);
            while (!m_previewFrameQueue.empty()) {
                m_previewFrameQueue.pop();
            }
        }
        
        LOG_INFO("Face detection preview disabled for: ", GetListenerName());
    }
    
    return true;
}

bool FaceDetectionSwitchingTrigger::IsPreviewEnabled() const {
    std::lock_guard<std::mutex> lock(m_previewMutex);
    return m_previewEnabled;
}

void FaceDetectionSwitchingTrigger::InitializePreview() {
    try {
        // Create named window for preview
        cv::namedWindow(m_previewWindowName, cv::WINDOW_AUTOSIZE);
        
        // Set window properties
        cv::setWindowProperty(m_previewWindowName, cv::WND_PROP_TOPMOST, 1.0);
        
        LOG_DEBUG("Initialized face detection preview window: ", m_previewWindowName);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize preview window: ", e.what());
    }
}

void FaceDetectionSwitchingTrigger::DestroyPreview() {
    try {
        cv::destroyWindow(m_previewWindowName);
        LOG_DEBUG("Destroyed face detection preview window: ", m_previewWindowName);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error destroying preview window: ", e.what());
    }
}

void FaceDetectionSwitchingTrigger::UpdatePreview(const cv::Mat& frame, const std::vector<cv::Rect>& faces) {
    if (!m_previewEnabled) {
        return;
    }
    
    try {
        // Create preview frame with face rectangles
        cv::Mat previewFrame = CreatePreviewFrame(frame, faces);
        
        // Queue the frame for main thread processing (no OpenCV GUI calls here)
        QueuePreviewFrame(previewFrame);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error creating preview frame: ", e.what());
    }
}

cv::Mat FaceDetectionSwitchingTrigger::CreatePreviewFrame(const cv::Mat& frame, const std::vector<cv::Rect>& faces) {
    cv::Mat previewFrame;
    
    // Convert to color if grayscale for better visualization
    if (frame.channels() == 1) {
        cv::cvtColor(frame, previewFrame, cv::COLOR_GRAY2BGR);
    } else {
        previewFrame = frame.clone();
    }
    
    // Draw face rectangles only if visualization is enabled
    if (m_config.enableVisualization) {
        for (const auto& face : faces) {
            // Draw main face rectangle in green
            cv::rectangle(previewFrame, face, cv::Scalar(0, 255, 0), 2);

            // Add face size text
            std::string sizeText = std::to_string(face.width) + "x" + std::to_string(face.height);
            cv::putText(previewFrame, sizeText,
                       cv::Point(face.x, face.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
    
    // Add status information overlay
    std::string statusText = "Faces: " + std::to_string(faces.size());
    cv::putText(previewFrame, statusText, 
               cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    
    // Add switching state information
    std::string switchText;
    if (m_shouldSwitchToVideo1.load()) {
        switchText = "SWITCH TO VIDEO 1";
        cv::putText(previewFrame, switchText, 
                   cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    } else if (m_shouldSwitchToVideo2.load()) {
        switchText = "SWITCH TO VIDEO 2";
        cv::putText(previewFrame, switchText, 
                   cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }
    
    // Add algorithm information
    std::string algoText = "Algorithm: ";
    switch (m_config.algorithm) {
        case FaceDetectionAlgorithm::HAAR_CASCADE:
            algoText += "Haar Cascade";
            break;
#if defined(HAVE_OPENCV_DNN)
        case FaceDetectionAlgorithm::YUNET:
            algoText += "YuNet";
            break;
        case FaceDetectionAlgorithm::CENTERFACE:
            algoText += "CenterFace";
            break;
#endif
        default:
            algoText += "Unknown";
            break;
    }
    
    cv::putText(previewFrame, algoText, 
               cv::Point(10, previewFrame.rows - 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(128, 128, 128), 1);
    
    return previewFrame;
}

void FaceDetectionSwitchingTrigger::QueuePreviewFrame(const cv::Mat& previewFrame) {
    std::lock_guard<std::mutex> lock(m_previewQueueMutex);
    
    // Add frame to queue
    m_previewFrameQueue.push(previewFrame.clone());
    
    // Keep queue size limited to prevent memory buildup
    while (m_previewFrameQueue.size() > MAX_PREVIEW_QUEUE_SIZE) {
        m_previewFrameQueue.pop();
    }
}

void FaceDetectionSwitchingTrigger::UpdatePreviewMainThread() {
    if (!m_previewEnabled) {
        return;
    }
    
    cv::Mat frameToDisplay;
    
    // Get the most recent frame from the queue
    {
        std::lock_guard<std::mutex> lock(m_previewQueueMutex);
        if (m_previewFrameQueue.empty()) {
            return;
        }
        
        // Get the most recent frame and clear older ones
        while (!m_previewFrameQueue.empty()) {
            frameToDisplay = m_previewFrameQueue.front();
            m_previewFrameQueue.pop();
        }
    }
    
    if (!frameToDisplay.empty()) {
        try {
            // Safe to call OpenCV GUI functions from main thread
            cv::imshow(m_previewWindowName, frameToDisplay);
            cv::waitKey(1);  // Process window events (non-blocking)
            
        } catch (const std::exception& e) {
            LOG_ERROR("Error displaying preview window: ", e.what());
        }
    }
}