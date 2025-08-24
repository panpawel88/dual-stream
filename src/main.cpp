#include <iostream>
#include <string>
#include <vector>
#include "ui/Window.h"
#include "core/CommandLineParser.h"
#include "video/VideoValidator.h"
#include "video/decode/HardwareDecoder.h"
#include "video/demux/VideoDemuxer.h"
#include "rendering/RendererFactory.h"
#include "rendering/TextureConverter.h"
#include "video/VideoManager.h"
#include "video/triggers/SwitchingTriggerFactory.h"
#include "core/Logger.h"
#include "core/FFmpegInitializer.h"

int main(int argc, char* argv[]) {
    // Parse command line arguments first to get debug flag
    VideoPlayerArgs args = CommandLineParser::Parse(argc, argv);
    if (!args.valid) {
        LOG_ERROR("Error: ", args.errorMessage);
        return 1;
    }
    
    // Initialize logger based on debug flag
    Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
    
    LOG_INFO("FFmpeg Video Player v1.0.0");
    
    // Initialize FFmpeg with RAII cleanup
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
    
    // Validate video files and get their properties
    VideoInfo video1Info = VideoValidator::GetVideoInfo(args.video1Path);
    VideoInfo video2Info = VideoValidator::GetVideoInfo(args.video2Path);
    
    std::string compatibilityError;
    if (!VideoValidator::ValidateCompatibility(video1Info, video2Info, compatibilityError)) {
        LOG_ERROR("Error: ", compatibilityError);
        return 1;
    }
    
    // Create window with video resolution
    Window window;
    if (!window.Create("FFmpeg Video Player", video1Info.width, video1Info.height)) {
        LOG_ERROR("Failed to create window");
        return 1;
    }
    
    window.Show();
    LOG_INFO("Window created. Press 1/2 to switch videos, ESC to exit");
    
    // Create renderer using factory
    auto renderer = RendererFactory::CreateRenderer();
    if (!renderer->Initialize(window.GetHandle(), video1Info.width, video1Info.height)) {
        LOG_ERROR("Failed to initialize ", RendererFactory::GetRendererName(), " renderer");
        return 1;
    }
    
    LOG_INFO("Initialized ", RendererFactory::GetRendererName(), " renderer");
    
    // Initialize video manager with renderer for hardware decoding support
    VideoManager videoManager;
    if (!videoManager.Initialize(args.video1Path, args.video2Path, renderer.get(), args.switchingAlgorithm, args.playbackSpeed)) {
        LOG_ERROR("Failed to initialize video manager");
        return 1;
    }
    
    // Create and set switching trigger
    auto switchingTrigger = SwitchingTriggerFactory::Create(args.triggerType, &window);
    if (!switchingTrigger) {
        LOG_ERROR("Failed to create switching trigger");
        return 1;
    }
    
    if (!videoManager.SetSwitchingTrigger(std::move(switchingTrigger))) {
        LOG_ERROR("Failed to set switching trigger");
        return 1;
    }
    
    // Start playback
    if (!videoManager.Play()) {
        LOG_ERROR("Failed to start video playback");
        return 1;
    }
    
    LOG_INFO("Video playback started successfully");
    
    // Main message loop
    while (window.ProcessMessages()) {
        // Update and process switching triggers
        videoManager.UpdateSwitchingTrigger();
        videoManager.ProcessSwitchingTriggers();
        
        // Update video frames only when needed (based on video frame rate)
        if (videoManager.ShouldUpdateFrame()) {
            if (!videoManager.UpdateFrame()) {
                LOG_ERROR("Failed to update video frame");
                break;
            }
        }
        
        // Get current frame and render it
        DecodedFrame* currentFrame = videoManager.GetCurrentFrame();
        
        // Convert frame to generic render texture
        RenderTexture renderTexture;
        if (currentFrame && currentFrame->valid) {
            renderTexture = TextureConverter::ConvertFrame(*currentFrame, renderer.get());
        } else {
            renderTexture = TextureConverter::CreateNullTexture();
        }
        
        // Present texture using unified renderer interface
        renderer->Present(renderTexture);
        
        // Sleep for a short time to prevent busy waiting, but much shorter than frame interval
        Sleep(1); // 1ms sleep to prevent excessive CPU usage
    }
    
    LOG_INFO("Application exiting...");
    return 0;
}