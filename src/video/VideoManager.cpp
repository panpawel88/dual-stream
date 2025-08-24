#include "VideoManager.h"
#include "core/Logger.h"
#include "rendering/IRenderer.h"
#if USE_OPENGL_RENDERER
#include "rendering/OpenGLRenderer.h"
#else
#include "rendering/D3D11Renderer.h"
#endif
#include <iostream>
#include <algorithm>
#include <cmath>

VideoManager::VideoManager()
    : m_initialized(false)
    , m_state(VideoState::STOPPED)
    , m_activeVideo(ActiveVideo::VIDEO_1)
    , m_pausedTime(0.0)
    , m_frameInterval(1.0 / 60.0) // Default to 60 FPS
    , m_playbackSpeed(1.0) { // Default to normal speed
}

VideoManager::~VideoManager() {
    Cleanup();
}

bool VideoManager::Initialize(const std::string& video1Path, const std::string& video2Path, IRenderer* renderer, SwitchingAlgorithm switchingAlgorithm, double playbackSpeed) {
    if (m_initialized) {
        Cleanup();
    }
    
    LOG_INFO("Initializing VideoManager...");
    
    // Set playback speed
    m_playbackSpeed = playbackSpeed;
    LOG_INFO("Playback speed set to: ", m_playbackSpeed, "x");
    
    // Extract platform-specific information from renderer
    ID3D11Device* d3dDevice = nullptr;
    bool cudaInteropAvailable = false;
    
    if (renderer) {
        switch (renderer->GetRendererType()) {
#if USE_OPENGL_RENDERER
            case RendererType::OpenGL: {
                OpenGLRenderer* glRenderer = static_cast<OpenGLRenderer*>(renderer);
                cudaInteropAvailable = glRenderer->IsCudaInteropAvailable();
                LOG_INFO("Using OpenGL renderer", cudaInteropAvailable ? " with CUDA interop" : " with software decoding");
                break;
            }
#else
            case RendererType::DirectX11: {
                D3D11Renderer* d3d11Renderer = static_cast<D3D11Renderer*>(renderer);
                d3dDevice = d3d11Renderer->GetDevice();
                LOG_INFO("Using D3D11 renderer for hardware decoding");
                break;
            }
#endif
            default:
                LOG_WARNING("Unknown renderer type");
                break;
        }
    }
    
    // Initialize both video streams
    if (!InitializeVideoStream(m_videos[0], video1Path, d3dDevice, cudaInteropAvailable)) {
        LOG_ERROR("Failed to initialize video stream 1");
        Cleanup();
        return false;
    }
    
    if (!InitializeVideoStream(m_videos[1], video2Path, d3dDevice, cudaInteropAvailable)) {
        LOG_ERROR("Failed to initialize video stream 2");
        Cleanup();
        return false;
    }
    
    // Validate stream compatibility
    if (!ValidateStreams()) {
        LOG_ERROR("Video streams are not compatible");
        Cleanup();
        return false;
    }
    
    // Set frame interval based on active video frame rate
    double frameRate = m_videos[0].demuxer.GetFrameRate();
    if (frameRate > 0) {
        m_frameInterval = 1.0 / frameRate;
    } else {
        // Fallback to 30 FPS if frame rate couldn't be determined
        m_frameInterval = 1.0 / 30.0;
        frameRate = 30.0;
    }
    
    // Create and initialize switching strategy
    m_switchingStrategy = VideoSwitchingStrategyFactory::Create(switchingAlgorithm);
    if (!m_switchingStrategy) {
        LOG_ERROR("Failed to create switching strategy");
        Cleanup();
        return false;
    }
    
    if (!m_switchingStrategy->Initialize(m_videos, this)) {
        LOG_ERROR("Failed to initialize switching strategy");
        Cleanup();
        return false;
    }
    
    LOG_INFO("VideoManager initialized successfully");
    LOG_INFO("Video 1 duration: ", m_videos[0].duration, " seconds");
    LOG_INFO("Video 2 duration: ", m_videos[1].duration, " seconds");
    LOG_INFO("Frame rate: ", frameRate, " FPS");
    LOG_INFO("Using switching strategy: ", m_switchingStrategy->GetName());
    
    m_initialized = true;
    return true;
}

void VideoManager::Cleanup() {
    if (m_initialized) {
        Stop();
        
        // Cleanup switching strategy
        if (m_switchingStrategy) {
            m_switchingStrategy->Cleanup();
            m_switchingStrategy.reset();
        }
        
        for (int i = 0; i < 2; i++) {
            m_videos[i].decoder.Cleanup();
            m_videos[i].demuxer.Close();
            m_videos[i].initialized = false;
        }
        
        m_initialized = false;
    }
}

bool VideoManager::Play() {
    if (!m_initialized) {
        return false;
    }
    
    if (m_state == VideoState::PLAYING) {
        return true; // Already playing
    }
    
    if (m_state == VideoState::PAUSED) {
        // Resume from pause
        ResetPlaybackTiming();
    } else {
        // Start from beginning or current position
        m_pausedTime = 0.0;
        ResetPlaybackTiming();
    }
    
    m_state = VideoState::PLAYING;
    LOG_INFO("Playback started");
    return true;
}

bool VideoManager::Pause() {
    if (!m_initialized || m_state != VideoState::PLAYING) {
        return false;
    }
    
    m_pausedTime = GetCurrentTime();
    m_state = VideoState::PAUSED;
    LOG_INFO("Playback paused at ", m_pausedTime, " seconds");
    return true;
}

bool VideoManager::Stop() {
    if (!m_initialized) {
        return false;
    }
    
    m_state = VideoState::STOPPED;
    m_pausedTime = 0.0;
    
    // Reset both video streams
    for (int i = 0; i < 2; i++) {
        m_videos[i].state = VideoState::STOPPED;
        m_videos[i].currentTime = 0.0;
        m_videos[i].decoder.Flush();
    }
    
    LOG_INFO("Playback stopped");
    return true;
}

bool VideoManager::SwitchToVideo(ActiveVideo video) {
    if (!m_initialized || !m_switchingStrategy) {
        return false;
    }
    
    double currentTime = GetCurrentTime();
    
    // Delegate switching to the strategy
    bool result = m_switchingStrategy->SwitchToVideo(video, currentTime);
    if (result) {
        m_activeVideo = video;
    }
    
    return result;
}

bool VideoManager::UpdateFrame() {
    if (!m_initialized || m_state != VideoState::PLAYING || !m_switchingStrategy) {
        if (!m_initialized) {
            LOG_DEBUG("UpdateFrame skipped - not initialized");
        } else if (m_state != VideoState::PLAYING) {
            LOG_DEBUG("UpdateFrame skipped - state is not PLAYING (state=", static_cast<int>(m_state), ")");
        }
        return true;
    }
    
    LOG_DEBUG("UpdateFrame called - processing...");

    // Check if it's time to present a new frame
    if (!ShouldPresentFrame()) {
        LOG_DEBUG("No new frame needed yet");
        return true; // No new frame needed yet
    }
    
    LOG_DEBUG("Time to present new frame");

    // Delegate frame updating to the strategy
    if (!m_switchingStrategy->UpdateFrame()) {
        LOG_ERROR("Strategy failed to update frame");
        return false;
    }
    
    m_lastFrameTime = std::chrono::steady_clock::now();
    LOG_DEBUG("UpdateFrame completed successfully");
    return true;
}

DecodedFrame* VideoManager::GetCurrentFrame() {
    if (!m_initialized || !m_switchingStrategy) {
        return nullptr;
    }
    
    // Delegate to the strategy
    return m_switchingStrategy->GetCurrentFrame();
}

double VideoManager::GetCurrentTime() const {
    if (!m_initialized) {
        return 0.0;
    }
    
    if (m_state == VideoState::PAUSED || m_state == VideoState::STOPPED) {
        return m_pausedTime;
    }
    
    if (m_state == VideoState::PLAYING) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - m_playbackStartTime);
        return m_pausedTime + (elapsed.count() / 1000000.0) * m_playbackSpeed;
    }
    
    return 0.0;
}

double VideoManager::GetDuration() const {
    if (!m_initialized) {
        return 0.0;
    }
    
    return m_videos[static_cast<int>(m_activeVideo)].duration;
}

bool VideoManager::InitializeVideoStream(VideoStream& stream, const std::string& filePath, ID3D11Device* d3dDevice, bool cudaInteropAvailable) {
    // Open demuxer
    if (!stream.demuxer.Open(filePath)) {
        LOG_ERROR("Failed to open video file: ", filePath);
        return false;
    }
    
    // Get decoder info
    DecoderInfo decoderInfo = HardwareDecoder::GetBestDecoder(stream.demuxer.GetCodecID());
    
    // Initialize decoder
    if (!stream.decoder.Initialize(stream.demuxer.GetCodecParameters(), decoderInfo, d3dDevice, stream.demuxer.GetTimeBase(), cudaInteropAvailable)) {
        LOG_ERROR("Failed to initialize decoder for: ", filePath);
        return false;
    }
    
    // Set stream properties
    stream.duration = stream.demuxer.GetDuration();
    stream.currentTime = 0.0;
    stream.state = VideoState::STOPPED;
    stream.initialized = true;
    
    LOG_INFO("Video stream initialized: ", filePath);
    LOG_INFO("  Using decoder: ", decoderInfo.name);
    
    return true;
}

bool VideoManager::ValidateStreams() {
    if (!m_videos[0].initialized || !m_videos[1].initialized) {
        return false;
    }
    
    // Check if resolutions match (should already be validated, but double-check)
    int width1 = m_videos[0].demuxer.GetWidth();
    int height1 = m_videos[0].demuxer.GetHeight();
    int width2 = m_videos[1].demuxer.GetWidth();
    int height2 = m_videos[1].demuxer.GetHeight();
    
    if (width1 != width2 || height1 != height2) {
        LOG_ERROR("Video resolution mismatch: ", width1, "x", height1, " vs ", width2, "x", height2);
        return false;
    }
    
    return true;
}

bool VideoManager::ShouldPresentFrame() const {
    if (m_state != VideoState::PLAYING) {
        return false;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastFrameTime);
    double elapsedSeconds = timeSinceLastFrame.count() / 1000000.0;
    
    // Adjust frame interval based on playback speed
    // Lower speed means longer interval between frames
    double adjustedFrameInterval = m_frameInterval / m_playbackSpeed;
    
    return elapsedSeconds >= adjustedFrameInterval;
}

bool VideoManager::ShouldUpdateFrame() const {
    return ShouldPresentFrame();
}

void VideoManager::ResetPlaybackTiming() {
    m_playbackStartTime = std::chrono::steady_clock::now();
    m_lastFrameTime = m_playbackStartTime;
}