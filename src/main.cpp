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
#include "core/Logger.h"
#include "core/FFmpegInitializer.h"

int main(int argc, char* argv[]) {
    VideoPlayerArgs args = CommandLineParser::Parse(argc, argv);
    if (!args.valid) {
        LOG_ERROR("Error: ", args.errorMessage);
        return 1;
    }
    
    Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
    
    LOG_INFO("DualStream Video Player v2.0.0");
    
    FFmpegInitializer ffmpegInit;
    if (!ffmpegInit.Initialize()) {
        LOG_ERROR("Failed to initialize FFmpeg");
        return 1;
    }
    
    LOG_INFO("Video 1: ", args.video1Path);
    LOG_INFO("Video 2: ", args.video2Path);
    LOG_INFO("Switching Algorithm: ", VideoSwitchingStrategyFactory::GetAlgorithmName(args.switchingAlgorithm));
    LOG_INFO("Switching Trigger: ", SwitchingTriggerFactory::GetTriggerTypeName(args.triggerType));
    LOG_INFO("Playback Speed: ", args.playbackSpeed, "x");
    
    VideoInfo video1Info = VideoValidator::GetVideoInfo(args.video1Path);
    VideoInfo video2Info = VideoValidator::GetVideoInfo(args.video2Path);
    
    std::string compatibilityError;
    if (!VideoValidator::ValidateCompatibility(video1Info, video2Info, compatibilityError)) {
        LOG_ERROR("Error: ", compatibilityError);
        return 1;
    }
    
    int maxVideoWidth = std::max(video1Info.width, video2Info.width);
    int maxVideoHeight = std::max(video1Info.height, video2Info.height);
    
    // Get display resolution to limit window size
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    int windowWidth = std::min(maxVideoWidth, screenWidth);
    int windowHeight = std::min(maxVideoHeight, screenHeight);
    
    if (windowWidth != maxVideoWidth || windowHeight != maxVideoHeight) {
        LOG_INFO("Window size limited to display resolution: ", windowWidth, "x", windowHeight, 
                " (video max: ", maxVideoWidth, "x", maxVideoHeight, ")");
    }
    
    Window window;
    if (!window.Create("DualStream Video Player", windowWidth, windowHeight)) {
        LOG_ERROR("Failed to create window");
        return 1;
    }
    
    window.Show();
    LOG_INFO("Window created. Press 1/2 to switch videos, F11 for fullscreen, ESC to exit");
    
    auto renderer = RendererFactory::CreateRenderer();
    if (!renderer->Initialize(window.GetHandle(), windowWidth, windowHeight)) {
        LOG_ERROR("Failed to initialize ", RendererFactory::GetRendererName(), " renderer");
        return 1;
    }
    
    LOG_INFO("Initialized ", RendererFactory::GetRendererName(), " renderer");
    
    VideoManager videoManager;
    if (!videoManager.Initialize(args.video1Path, args.video2Path, renderer.get(), args.switchingAlgorithm, args.playbackSpeed)) {
        LOG_ERROR("Failed to initialize video manager");
        return 1;
    }
    
    // Initialize camera manager if face detection is requested
    std::unique_ptr<CameraManager> cameraManager;
    if (args.triggerType == TriggerType::FACE_DETECTION) {
        cameraManager = std::make_unique<CameraManager>();
        if (!cameraManager->InitializeAuto()) {
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
    
    auto switchingTrigger = SwitchingTriggerFactory::Create(args.triggerType, triggerConfig);
    if (!switchingTrigger) {
        LOG_ERROR("Failed to create switching trigger");
        return 1;
    }
    
    // Initialize face detection if needed
    if (args.triggerType == TriggerType::FACE_DETECTION) {
        auto faceDetection = std::dynamic_pointer_cast<FaceDetectionSwitchingTrigger>(switchingTrigger);
        if (faceDetection && !faceDetection->InitializeFaceDetection()) {
            LOG_ERROR("Failed to initialize face detection");
            return 1;
        }
        
        // Register the trigger as a frame listener with the camera manager
        if (cameraManager && faceDetection) {
            cameraManager->RegisterFrameListener(faceDetection);
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
    
    while (window.ProcessMessages()) {
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
        
        renderer->Present(renderTexture);
        
        // Sleep for a short time to prevent busy waiting, but much shorter than frame interval
        Sleep(1); // 1ms sleep to prevent excessive CPU usage
    }
    
    LOG_INFO("Application exiting...");
    return 0;
}