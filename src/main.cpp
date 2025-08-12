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
        std::cerr << "Error: " << args.errorMessage << "\n";
        return 1;
    }
    
    // Initialize logger based on debug flag
    Logger::GetInstance().SetLogLevel(args.debugLogging ? LogLevel::Debug : LogLevel::Info);
    
    LOG_INFO("FFmpeg Video Player v1.0.0");
    
    // Initialize FFmpeg
    if (!VideoValidator::Initialize()) {
        std::cerr << "Failed to initialize FFmpeg\n";
        return 1;
    }
    
    // Initialize hardware decoder detection
    if (!HardwareDecoder::Initialize()) {
        std::cerr << "Failed to initialize hardware decoder detection\n";
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
        std::cerr << "Error: " << compatibilityError << "\n";
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    // Create window with video resolution
    Window window;
    if (!window.Create("FFmpeg Video Player", video1Info.width, video1Info.height)) {
        std::cerr << "Failed to create window\n";
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    window.Show();
    LOG_INFO("Window created. Press 1/2 to switch videos, ESC to exit");
    
    // Initialize D3D11 renderer
    D3D11Renderer renderer;
    if (!renderer.Initialize(window.GetHandle(), video1Info.width, video1Info.height)) {
        std::cerr << "Failed to initialize D3D11 renderer\n";
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    // Initialize video manager
    VideoManager videoManager;
    if (!videoManager.Initialize(args.video1Path, args.video2Path, renderer.GetDevice())) {
        std::cerr << "Failed to initialize video manager\n";
        HardwareDecoder::Cleanup();
        VideoValidator::Cleanup();
        return 1;
    }
    
    // Start playback
    if (!videoManager.Play()) {
        std::cerr << "Failed to start video playback\n";
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
                std::cerr << "Failed to update video frame\n";
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