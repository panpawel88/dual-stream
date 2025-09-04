#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#endif

#include "ui/Window.h"
#include "core/CommandLineParser.h"
#include "video/VideoValidator.h"
#include "video/decode/HardwareDecoder.h"
#include "video/demux/VideoDemuxer.h"
#include "rendering/RendererFactory.h"
#include "rendering/TextureConverter.h"
#include "video/VideoManager.h"
#include "video/triggers/SwitchingTriggerFactory.h"
#include "camera/CameraManager.h"
#include "camera/processing/FaceDetectionSwitchingTrigger.h"
#include "core/PerformanceStatistics.h"
#include "rendering/D3D11Renderer.h"
#include "ui/GlobalInputHandler.h"
#include "ui/UIRegistry.h"
#include "core/Logger.h"

#include "core/FFmpegInitializer.h"
#include "core/Config.h"


int main(int argc, char* argv[]) {
    VideoPlayerArgs args = CommandLineParser::Parse(argc, argv);
    if (!args.valid) {
        LOG_ERROR("Error: ", args.errorMessage);
        return 1;
    }
    
    // Initialize configuration system early
    Config* config = Config::GetInstance();
    
    // Load configuration file from command line or default location
    std::string configPath = args.configPath.empty() ? "config/default.ini" : args.configPath;
    if (!config->LoadFromFile(configPath)) {
        // If specified config doesn't exist, use built-in defaults
        LOG_INFO("Configuration file not found at '", configPath, "', using built-in defaults");
    }
    
    // Apply command line overrides to logging level
    LogLevel logLevel = args.debugLogging ? LogLevel::Debug : 
                       (config->GetString("logging.default_level") == "debug" ? LogLevel::Debug :
                        config->GetString("logging.default_level") == "warning" ? LogLevel::Warning :
                        config->GetString("logging.default_level") == "error" ? LogLevel::Error :
                        LogLevel::Info);
    
    Logger::GetInstance().SetLogLevel(logLevel);
    
    // If algorithm not specified on command line, use config default
    SwitchingAlgorithm finalAlgorithm;
    if (!args.switchingAlgorithm.has_value()) {
        std::string configAlgorithm = config->GetString("video.default_algorithm", "keyframe-sync");
        SwitchingAlgorithm parsedAlgorithm = VideoSwitchingStrategyFactory::ParseAlgorithm(configAlgorithm);
        if (parsedAlgorithm != static_cast<SwitchingAlgorithm>(-1)) {
            finalAlgorithm = parsedAlgorithm;
        } else {
            finalAlgorithm = SwitchingAlgorithm::KEYFRAME_SYNC;  // Fallback to keyframe-sync
        }
    } else {
        finalAlgorithm = args.switchingAlgorithm.value();
        
        // Apply command line overrides to configuration
        std::string algorithmName = VideoSwitchingStrategyFactory::GetAlgorithmName(finalAlgorithm);
        config->SetString("video.default_algorithm", algorithmName);
    }
    
    if (args.playbackSpeed != 1.0) {
        config->SetDouble("video.default_speed", args.playbackSpeed);
    }
    
    LOG_INFO("MultiVideo Player v3.0.0");
    
    FFmpegInitializer ffmpegInit;
    if (!ffmpegInit.Initialize()) {
        LOG_ERROR("Failed to initialize FFmpeg");
        return 1;
    }
    
    // Log all video paths
    LOG_INFO("Loading ", args.videoPaths.size(), " video file(s):");
    for (size_t i = 0; i < args.videoPaths.size(); i++) {
        LOG_INFO("Video ", (i + 1), ": ", args.videoPaths[i]);
    }
    LOG_INFO("Switching Algorithm: ", VideoSwitchingStrategyFactory::GetAlgorithmName(finalAlgorithm));
    LOG_INFO("Switching Trigger: ", SwitchingTriggerFactory::GetTriggerTypeName(args.triggerType));
    LOG_INFO("Playback Speed: ", args.playbackSpeed, "x");
    
    // Validate all videos
    std::vector<VideoInfo> videoInfos;
    std::string compatibilityError;
    if (!VideoValidator::ValidateMultipleVideos(args.videoPaths, videoInfos, compatibilityError)) {
        LOG_ERROR("Error: ", compatibilityError);
        return 1;
    }
    
    // Find maximum video dimensions
    int maxVideoWidth = 0;
    int maxVideoHeight = 0;
    for (const auto& info : videoInfos) {
        maxVideoWidth = std::max(maxVideoWidth, info.width);
        maxVideoHeight = std::max(maxVideoHeight, info.height);
    }
    
    // Get default window size from configuration
    int defaultWidth = config->GetInt("window.default_width", 1280);
    int defaultHeight = config->GetInt("window.default_height", 720);
    
    // Use the larger of video size or default size, but respect config preference
    int windowWidth = std::max(maxVideoWidth, defaultWidth);
    int windowHeight = std::max(maxVideoHeight, defaultHeight);
    
    // Limit to display resolution if configured to do so
    if (config->GetBool("window.limit_to_display", true)) {
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        windowWidth = std::min(windowWidth, screenWidth);
        windowHeight = std::min(windowHeight, screenHeight);
    }
    
    LOG_INFO("Window size: ", windowWidth, "x", windowHeight, 
             " (video max: ", maxVideoWidth, "x", maxVideoHeight, 
             ", config default: ", defaultWidth, "x", defaultHeight, ")");
    
    Window window;
    std::string windowTitle = "MultiVideo Player";
    if (args.videoPaths.size() > 1) {
        windowTitle += " (1 of " + std::to_string(args.videoPaths.size()) + ")";
    }
    
    if (!window.Create(windowTitle, windowWidth, windowHeight)) {
        LOG_ERROR("Failed to create window");
        return 1;
    }
    
    window.Show();
    if (args.videoPaths.size() == 1) {
        LOG_INFO("Window created. Press F11 for fullscreen, ESC to exit (single video mode)");
    } else {
        LOG_INFO("Window created. Press 1-", args.videoPaths.size(), " to switch videos, F11 for fullscreen, ESC to exit");
    }
    
    // Parse preferred renderer backend from config
    std::string preferredBackendStr = config->GetString("rendering.preferred_backend", "auto");
    RendererBackend preferredBackend = RendererFactory::ParseBackendString(preferredBackendStr);
    
    LOG_INFO("Preferred renderer backend from config: ", preferredBackendStr);
    
    auto renderer = RendererFactory::CreateRenderer(preferredBackend);
    if (!renderer) {
        LOG_ERROR("Failed to create any renderer backend");
        return 1;
    }
    
    if (!renderer->Initialize(window.GetHandle(), windowWidth, windowHeight)) {
        LOG_ERROR("Failed to initialize ", RendererFactory::GetRendererName(preferredBackend), " renderer");
        return 1;
    }
    
    // Log which renderer was actually created
    const char* actualRendererName = (renderer->GetRendererType() == RendererType::OpenGL) ? "OpenGL" : "DirectX 11";
    LOG_INFO("Successfully initialized ", actualRendererName, " renderer");
    
    // Initialize global input handler and register overlay toggle (Insert key)
    GlobalInputHandler::GetInstance().RegisterOverlayToggle(VK_INSERT);
    LOG_INFO("Registered overlay toggle on Insert key");
    
    // Register fullscreen toggle (F11 key)
    GlobalInputHandler::GetInstance().RegisterFullscreenToggle(VK_F11, &window);
    LOG_INFO("Registered fullscreen toggle on F11 key");
    
    
    VideoManager videoManager;
    if (!videoManager.Initialize(args.videoPaths, renderer.get(), finalAlgorithm, args.playbackSpeed)) {
        LOG_ERROR("Failed to initialize video manager");
        return 1;
    }
    
    // Initialize camera manager if face detection is requested
    std::unique_ptr<CameraManager> cameraManager;
    if (args.triggerType == TriggerType::FACE_DETECTION) {
        cameraManager = std::make_unique<CameraManager>();
        CameraConfig cameraConfig = CameraManager::CreateCameraConfigFromGlobal();
        PublisherConfig publisherConfig = CameraManager::CreatePublisherConfigFromGlobal();
        if (!cameraManager->InitializeAuto(cameraConfig, publisherConfig)) {
            LOG_ERROR("Failed to initialize camera for face detection");
            return 1;
        }
        if (!cameraManager->StartCapture()) {
            LOG_ERROR("Failed to start camera capture");
            return 1;
        }
        LOG_INFO("Camera initialized for face detection");
    }
    
    // Create trigger configuration
    TriggerConfig triggerConfig;
    triggerConfig.window = &window;
    triggerConfig.videoCount = args.videoPaths.size();
    triggerConfig.faceDetectionConfig = FaceDetectionSwitchingTrigger::CreateConfigFromGlobal();
    
    auto switchingTrigger = SwitchingTriggerFactory::Create(args.triggerType, triggerConfig);
    if (!switchingTrigger) {
        LOG_ERROR("Failed to create switching trigger");
        return 1;
    }
    
    // Initialize face detection if needed and store reference for main loop
    std::shared_ptr<FaceDetectionSwitchingTrigger> faceDetectionTrigger;
    if (args.triggerType == TriggerType::FACE_DETECTION) {
        faceDetectionTrigger = std::dynamic_pointer_cast<FaceDetectionSwitchingTrigger>(switchingTrigger);
        if (faceDetectionTrigger && !faceDetectionTrigger->InitializeFaceDetection()) {
            LOG_ERROR("Failed to initialize face detection");
            return 1;
        }
        
        // Register the trigger as a frame listener with the camera manager
        if (cameraManager && faceDetectionTrigger) {
            cameraManager->RegisterFrameListener(faceDetectionTrigger);
        }
    }
    
    if (!videoManager.SetSwitchingTrigger(switchingTrigger)) {
        LOG_ERROR("Failed to set switching trigger");
        return 1;
    }
    
    if (!videoManager.Play()) {
        LOG_ERROR("Failed to start video playback");
        return 1;
    }
    
    LOG_INFO("Video playback started successfully");
    
    int lastWindowWidth = window.GetWidth();
    int lastWindowHeight = window.GetHeight();
    
    // Register performance statistics with UI registry
    PerformanceStatistics& stats = PerformanceStatistics::GetInstance();
    UIRegistry::GetInstance().RegisterDrawable(&stats);
    
    // Main loop timing variables
    auto loopStartTime = std::chrono::high_resolution_clock::now();
    auto lastFrameTime = loopStartTime;
    
    while (window.ProcessMessages()) {
        auto currentFrameTime = std::chrono::high_resolution_clock::now();
        auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(currentFrameTime - lastFrameTime);
        double frameTimeMs = frameDuration.count() / 1000.0;
        int currentWidth = window.GetWidth();
        int currentHeight = window.GetHeight();
        
        if (currentWidth != lastWindowWidth || currentHeight != lastWindowHeight) {
            LOG_INFO("Window size changed to ", currentWidth, "x", currentHeight);
            
            if (!renderer->Resize(currentWidth, currentHeight)) {
                LOG_ERROR("Failed to resize renderer to ", currentWidth, "x", currentHeight);
                break;
            }
            
            lastWindowWidth = currentWidth;
            lastWindowHeight = currentHeight;
        }
        videoManager.UpdateSwitchingTrigger();
        videoManager.ProcessSwitchingTriggers();
        
        // Update global input handler for overlay toggle and other global shortcuts
        GlobalInputHandler::GetInstance().Update();
        
        // Update video frames only when needed (based on video frame rate)
        if (videoManager.ShouldUpdateFrame()) {
            if (!videoManager.UpdateFrame()) {
                LOG_ERROR("Failed to update video frame");
                break;
            }
        }
        
        DecodedFrame* currentFrame = videoManager.GetCurrentFrame();
        
        RenderTexture renderTexture;
        if (currentFrame && currentFrame->valid) {
            renderTexture = TextureConverter::ConvertFrame(*currentFrame, renderer.get());
        } else {
            renderTexture = TextureConverter::CreateNullTexture();
        }
        
        // Track render time
        auto renderStartTime = std::chrono::high_resolution_clock::now();
        renderer->Present(renderTexture);
        auto renderEndTime = std::chrono::high_resolution_clock::now();
        
        auto renderDuration = std::chrono::duration_cast<std::chrono::microseconds>(renderEndTime - renderStartTime);
        double renderTimeMs = renderDuration.count() / 1000.0;
        
        // Update performance statistics
        stats.RecordApplicationFrameTime(frameTimeMs);
        stats.RecordRenderTime(renderTimeMs);
        
        // Record main loop time  
        auto loopEndTime = std::chrono::high_resolution_clock::now();
        auto loopDuration = std::chrono::duration_cast<std::chrono::microseconds>(loopEndTime - currentFrameTime);
        double loopTimeMs = loopDuration.count() / 1000.0;
        stats.RecordMainLoopTime(loopTimeMs);
        
        lastFrameTime = currentFrameTime;
        
        // Sleep for a short time to prevent busy waiting, but much shorter than frame interval
        Sleep(1); // 1ms sleep to prevent excessive CPU usage
    }
    
    LOG_INFO("Application exiting...");
    return 0;
}