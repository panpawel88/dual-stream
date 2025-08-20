#include <iostream>
#include <string>
#include <vector>
#include "Window.h"
#include "CommandLineParser.h"
#include "VideoValidator.h"
#include "HardwareDecoder.h"
#include "VideoDemuxer.h"
#if USE_OPENGL_RENDERER
#include "OpenGLRenderer.h"
#else
#include "D3D11Renderer.h"
#endif
#include "VideoManager.h"
#include "Logger.h"
#include "FFmpegInitializer.h"

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
    
#if USE_OPENGL_RENDERER
    // Initialize OpenGL renderer
    OpenGLRenderer renderer;
    if (!renderer.Initialize(window.GetHandle(), video1Info.width, video1Info.height)) {
        LOG_ERROR("Failed to initialize OpenGL renderer");
        return 1;
    }
    
    // Initialize video manager with switching strategy and playback speed (no D3D device for OpenGL)
    VideoManager videoManager;
    bool cudaInteropAvailable = false;
#if USE_OPENGL_RENDERER && HAVE_CUDA
    cudaInteropAvailable = renderer.IsCudaInteropAvailable();
#endif
    if (!videoManager.Initialize(args.video1Path, args.video2Path, nullptr, args.switchingAlgorithm, args.playbackSpeed, cudaInteropAvailable)) {
        LOG_ERROR("Failed to initialize video manager");
        return 1;
    }
#else
    // Initialize D3D11 renderer
    D3D11Renderer renderer;
    if (!renderer.Initialize(window.GetHandle(), video1Info.width, video1Info.height)) {
        LOG_ERROR("Failed to initialize D3D11 renderer");
        return 1;
    }
    
    // Initialize video manager with switching strategy and playback speed
    VideoManager videoManager;
    if (!videoManager.Initialize(args.video1Path, args.video2Path, renderer.GetDevice(), args.switchingAlgorithm, args.playbackSpeed)) {
        LOG_ERROR("Failed to initialize video manager");
        return 1;
    }
#endif
    
    // Start playback
    if (!videoManager.Play()) {
        LOG_ERROR("Failed to start video playback");
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
        if (currentFrame && currentFrame->valid) {
#if USE_OPENGL_RENDERER
            // OpenGL renderer can use both hardware CUDA frames and software frame data
#if HAVE_CUDA
            if (currentFrame->isHardwareCuda) {
                // Use hardware presentation for CUDA frames (only when CUDA interop is available)
                renderer.PresentHardware(*currentFrame);
            } else
#endif
            if (currentFrame->data) {
                // Use software presentation for CPU frames
                renderer.Present(currentFrame->data, currentFrame->width, currentFrame->height, currentFrame->pitch);
            } else {
                renderer.Present(nullptr, 0, 0, 0); // Render black screen if no frame data available
            }
#else
            // D3D11 renderer uses texture
            if (currentFrame->texture) {
                renderer.Present(currentFrame->texture.Get(), currentFrame->isYUV, currentFrame->format);
            } else {
                renderer.Present(nullptr); // Render black screen if no frame available
            }
#endif
        } else {
#if USE_OPENGL_RENDERER
            renderer.Present(nullptr, 0, 0, 0); // Render black screen
#else
            renderer.Present(nullptr); // Render black screen if no frame available
#endif
        }
        
        // Sleep for a short time to prevent busy waiting, but much shorter than frame interval
        Sleep(1); // 1ms sleep to prevent excessive CPU usage
    }
    
    LOG_INFO("Application exiting...");
    return 0;
}