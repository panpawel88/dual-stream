#include <iostream>
#include <string>
#include <vector>
#include "Window.h"
#include "CommandLineParser.h"
#include "VideoValidator.h"
#include "HardwareDecoder.h"
#include "VideoDemuxer.h"
#include "D3D11Renderer.h"
#include "VideoManager.h"
#include "Logger.h"

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
    
    // Initialize FFmpeg
    if (!VideoValidator::Initialize()) {
        LOG_ERROR("Failed to initialize FFmpeg");
        return 1;
    }
    
    // Initialize hardware decoder detection
    if (!HardwareDecoder::Initialize()) {
        LOG_ERROR("Failed to initialize hardware decoder detection");
        VideoValidator::Cleanup();
        return 1;
    }
    
    LOG_INFO("Video 1: ", args.video1Path);
    LOG_INFO("Video 2: ", args.video2Path);
    
    // Validate video files and get their properties
    VideoInfo video1Info = VideoValidator::GetVideoInfo(args.video1Path);
    VideoInfo video2Info = VideoValidator::GetVideoInfo(args.video2Path);
    
    std::string compatibilityError;
    if (!VideoValidator::ValidateCompatibility(video1Info, video2Info, compatibilityError)) {
        LOG_ERROR("Error: ", compatibilityError);
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    // Create window with video resolution
    Window window;
    if (!window.Create("FFmpeg Video Player", video1Info.width, video1Info.height)) {
        LOG_ERROR("Failed to create window");
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    window.Show();
    LOG_INFO("Window created. Press 1/2 to switch videos, ESC to exit");
    
    // Initialize D3D11 renderer
    D3D11Renderer renderer;
    if (!renderer.Initialize(window.GetHandle(), video1Info.width, video1Info.height)) {
        LOG_ERROR("Failed to initialize D3D11 renderer");
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    // Initialize video manager
    VideoManager videoManager;
    if (!videoManager.Initialize(args.video1Path, args.video2Path, renderer.GetDevice())) {
        LOG_ERROR("Failed to initialize video manager");
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    // Start playback
    if (!videoManager.Play()) {
        LOG_ERROR("Failed to start video playback");
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    LOG_INFO("Video playback started successfully");
    
    // Main message loop
    while (window.ProcessMessages()) {
        // Handle keyboard input for video switching
        if (window.IsKeyPressed('1')) {
            LOG_INFO("Switching to video 1");
            videoManager.SwitchToVideo(ActiveVideo::VIDEO_1);
            window.ClearKeyPress('1');
        }
        
        if (window.IsKeyPressed('2')) {
            LOG_INFO("Switching to video 2");
            videoManager.SwitchToVideo(ActiveVideo::VIDEO_2);
            window.ClearKeyPress('2');
        }
        
        // Update video frames only when needed (based on video frame rate)
        if (videoManager.ShouldUpdateFrame()) {
            if (!videoManager.UpdateFrame()) {
                LOG_ERROR("Failed to update video frame");
                break;
            }
        }
        
        // Get current frame and render it
        DecodedFrame* currentFrame = videoManager.GetCurrentFrame();
        if (currentFrame && currentFrame->valid && currentFrame->texture) {
            renderer.Present(currentFrame->texture.Get(), currentFrame->isYUV, currentFrame->format);
        } else {
            renderer.Present(nullptr); // Render black screen if no frame available
        }
        
        // Sleep for a short time to prevent busy waiting, but much shorter than frame interval
        Sleep(1); // 1ms sleep to prevent excessive CPU usage
    }
    
    LOG_INFO("Application exiting...");
    HardwareDecoder::Cleanup();
    VideoValidator::Cleanup();
    return 0;
}