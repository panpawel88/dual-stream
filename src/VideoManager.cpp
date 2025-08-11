#include "VideoManager.h"
#include <iostream>
#include <algorithm>

VideoManager::VideoManager()
    : m_initialized(false)
    , m_state(VideoState::STOPPED)
    , m_activeVideo(ActiveVideo::VIDEO_1)
    , m_pausedTime(0.0)
    , m_needsSeek(false)
    , m_targetSeekTime(0.0)
    , m_frameInterval(1.0 / 60.0) { // Default to 60 FPS
}

VideoManager::~VideoManager() {
    Cleanup();
}

bool VideoManager::Initialize(const std::string& video1Path, const std::string& video2Path, ID3D11Device* d3dDevice) {
    if (m_initialized) {
        Cleanup();
    }
    
    std::cout << "Initializing VideoManager...\n";
    
    // Initialize both video streams
    if (!InitializeVideoStream(m_videos[0], video1Path, d3dDevice)) {
        std::cerr << "Failed to initialize video stream 1\n";
        Cleanup();
        return false;
    }
    
    if (!InitializeVideoStream(m_videos[1], video2Path, d3dDevice)) {
        std::cerr << "Failed to initialize video stream 2\n";
        Cleanup();
        return false;
    }
    
    // Validate stream compatibility
    if (!ValidateStreams()) {
        std::cerr << "Video streams are not compatible\n";
        Cleanup();
        return false;
    }
    
    // Set frame interval based on active video frame rate
    double frameRate = m_videos[0].demuxer.GetFrameRate();
    if (frameRate > 0) {
        m_frameInterval = 1.0 / frameRate;
    }
    
    std::cout << "VideoManager initialized successfully\n";
    std::cout << "Video 1 duration: " << m_videos[0].duration << " seconds\n";
    std::cout << "Video 2 duration: " << m_videos[1].duration << " seconds\n";
    std::cout << "Frame rate: " << frameRate << " FPS\n";
    
    m_initialized = true;
    return true;
}

void VideoManager::Cleanup() {
    if (m_initialized) {
        Stop();
        
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
    std::cout << "Playback started\n";
    return true;
}

bool VideoManager::Pause() {
    if (!m_initialized || m_state != VideoState::PLAYING) {
        return false;
    }
    
    m_pausedTime = GetCurrentTime();
    m_state = VideoState::PAUSED;
    std::cout << "Playback paused at " << m_pausedTime << " seconds\n";
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
    
    std::cout << "Playback stopped\n";
    return true;
}

bool VideoManager::SwitchToVideo(ActiveVideo video) {
    if (!m_initialized || video == m_activeVideo) {
        return true;
    }
    
    ActiveVideo previousVideo = m_activeVideo;
    m_activeVideo = video;
    
    std::cout << "Switching to video " << (video == ActiveVideo::VIDEO_1 ? "1" : "2") << "\n";
    
    // If playing, synchronize the new active video to current time
    if (m_state == VideoState::PLAYING) {
        double currentTime = GetCurrentTime();
        if (!SynchronizeStreams()) {
            std::cerr << "Failed to synchronize streams after switch\n";
            m_activeVideo = previousVideo; // Revert
            return false;
        }
    }
    
    return true;
}

bool VideoManager::UpdateFrame() {
    if (!m_initialized || m_state != VideoState::PLAYING) {
        return true;
    }
    
    // Handle seeking if needed
    if (m_needsSeek) {
        if (!SeekToTime(m_targetSeekTime)) {
            std::cerr << "Failed to seek to time: " << m_targetSeekTime << "\n";
            return false;
        }
        m_needsSeek = false;
    }
    
    // Check if it's time to present a new frame
    if (!ShouldPresentFrame()) {
        return true; // No new frame needed yet
    }
    
    // Update playback time
    UpdatePlaybackTime();
    
    // Process frame for active video
    VideoStream& activeStream = m_videos[static_cast<int>(m_activeVideo)];
    
    if (!ProcessVideoFrame(activeStream)) {
        // Check if we reached end of stream
        if (activeStream.state == VideoState::END_OF_STREAM) {
            if (!HandleEndOfStream(activeStream)) {
                return false;
            }
        } else {
            std::cerr << "Failed to process video frame\n";
            return false;
        }
    }
    
    m_lastFrameTime = std::chrono::steady_clock::now();
    return true;
}

DecodedFrame* VideoManager::GetCurrentFrame() {
    if (!m_initialized) {
        return nullptr;
    }
    
    VideoStream& activeStream = m_videos[static_cast<int>(m_activeVideo)];
    if (activeStream.currentFrame.valid) {
        return &activeStream.currentFrame;
    }
    
    return nullptr;
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
        return m_pausedTime + (elapsed.count() / 1000000.0);
    }
    
    return 0.0;
}

double VideoManager::GetDuration() const {
    if (!m_initialized) {
        return 0.0;
    }
    
    return m_videos[static_cast<int>(m_activeVideo)].duration;
}

bool VideoManager::SeekToTime(double timeInSeconds) {
    if (!m_initialized) {
        return false;
    }
    
    // Clamp time to valid range
    double duration = GetDuration();
    timeInSeconds = (std::max)(0.0, (std::min)(timeInSeconds, duration));
    
    std::cout << "Seeking to " << timeInSeconds << " seconds\n";
    
    // Seek both streams to maintain synchronization
    for (int i = 0; i < 2; i++) {
        if (!SeekVideoStream(m_videos[i], timeInSeconds)) {
            std::cerr << "Failed to seek video stream " << (i + 1) << "\n";
            return false;
        }
    }
    
    m_pausedTime = timeInSeconds;
    ResetPlaybackTiming();
    
    return true;
}

bool VideoManager::InitializeVideoStream(VideoStream& stream, const std::string& filePath, ID3D11Device* d3dDevice) {
    // Open demuxer
    if (!stream.demuxer.Open(filePath)) {
        std::cerr << "Failed to open video file: " << filePath << "\n";
        return false;
    }
    
    // Get decoder info
    DecoderInfo decoderInfo = HardwareDecoder::GetBestDecoder(stream.demuxer.GetCodecID());
    
    // Initialize decoder
    if (!stream.decoder.Initialize(stream.demuxer.GetCodecParameters(), decoderInfo, d3dDevice)) {
        std::cerr << "Failed to initialize decoder for: " << filePath << "\n";
        return false;
    }
    
    // Set stream properties
    stream.duration = stream.demuxer.GetDuration();
    stream.currentTime = 0.0;
    stream.state = VideoState::STOPPED;
    stream.initialized = true;
    
    std::cout << "Video stream initialized: " << filePath << "\n";
    std::cout << "  Using decoder: " << decoderInfo.name << "\n";
    
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
        std::cerr << "Video resolution mismatch: " << width1 << "x" << height1 
                  << " vs " << width2 << "x" << height2 << "\n";
        return false;
    }
    
    return true;
}

bool VideoManager::ProcessVideoFrame(VideoStream& stream) {
    // Try to decode next frame
    if (!DecodeNextFrame(stream)) {
        return false;
    }
    
    return true;
}

bool VideoManager::DecodeNextFrame(VideoStream& stream) {
    AVPacket packet;
    av_init_packet(&packet);
    
    bool frameDecoded = false;
    
    // Keep trying to decode frames until we get one or reach end of stream
    while (!frameDecoded) {
        // Read packet from demuxer
        if (!stream.demuxer.ReadFrame(&packet)) {
            // End of stream
            stream.state = VideoState::END_OF_STREAM;
            av_packet_unref(&packet);
            return false;
        }
        
        // Send packet to decoder
        if (!stream.decoder.SendPacket(&packet)) {
            std::cerr << "Failed to send packet to decoder\n";
            av_packet_unref(&packet);
            return false;
        }
        
        // Try to receive decoded frame
        DecodedFrame newFrame;
        if (stream.decoder.ReceiveFrame(newFrame)) {
            if (newFrame.valid) {
                stream.currentFrame = newFrame;
                stream.currentTime = newFrame.presentationTime;
                frameDecoded = true;
            }
        }
        
        av_packet_unref(&packet);
    }
    
    return frameDecoded;
}

bool VideoManager::ShouldPresentFrame() const {
    if (m_state != VideoState::PLAYING) {
        return false;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastFrameTime);
    double elapsedSeconds = timeSinceLastFrame.count() / 1000000.0;
    
    return elapsedSeconds >= m_frameInterval;
}

void VideoManager::UpdatePlaybackTime() {
    // Playback time is managed by GetCurrentTime() method
    // This method can be used for additional time-related updates if needed
}

bool VideoManager::SynchronizeStreams() {
    double currentTime = GetCurrentTime();
    
    // Seek both streams to current time
    for (int i = 0; i < 2; i++) {
        if (!SeekVideoStream(m_videos[i], currentTime)) {
            return false;
        }
    }
    
    return true;
}

bool VideoManager::SeekVideoStream(VideoStream& stream, double timeInSeconds) {
    if (!stream.demuxer.SeekToTime(timeInSeconds)) {
        return false;
    }
    
    stream.decoder.Flush();
    stream.currentTime = timeInSeconds;
    stream.currentFrame.valid = false;
    
    return true;
}

void VideoManager::ResetPlaybackTiming() {
    m_playbackStartTime = std::chrono::steady_clock::now();
    m_lastFrameTime = m_playbackStartTime;
}

bool VideoManager::HandleEndOfStream(VideoStream& stream) {
    std::cout << "End of stream reached, restarting video\n";
    return RestartVideo(stream);
}

bool VideoManager::RestartVideo(VideoStream& stream) {
    // Seek back to beginning
    if (!SeekVideoStream(stream, 0.0)) {
        std::cerr << "Failed to restart video\n";
        return false;
    }
    
    stream.state = VideoState::PLAYING;
    return true;
}