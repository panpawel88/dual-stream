#include "VideoManager.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <cmath>

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
    
    LOG_INFO("Initializing VideoManager...");
    
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
    } else {
        // Fallback to 30 FPS if frame rate couldn't be determined
        m_frameInterval = 1.0 / 30.0;
        frameRate = 30.0;
    }
    
    LOG_INFO("VideoManager initialized successfully");
    LOG_INFO("Video 1 duration: ", m_videos[0].duration, " seconds");
    LOG_INFO("Video 2 duration: ", m_videos[1].duration, " seconds");
    LOG_INFO("Frame rate: ", frameRate, " FPS");
    
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
    if (!m_initialized || video == m_activeVideo) {
        return true;
    }
    
    ActiveVideo previousVideo = m_activeVideo;
    double currentTime = GetCurrentTime();
    
    LOG_INFO("Switching to video ", (video == ActiveVideo::VIDEO_1 ? "1" : "2"), " at time ", currentTime);
    
    // Switch active video first
    m_activeVideo = video;
    
    // If playing, synchronize the new active video to current playback time
    if (m_state == VideoState::PLAYING) {
        VideoStream& newActiveStream = m_videos[static_cast<int>(m_activeVideo)];
        
        // Handle looping if current time exceeds new video's duration
        double targetTime = currentTime;
        if (targetTime >= newActiveStream.duration && newActiveStream.duration > 0.0) {
            // Loop to the appropriate position within the video
            targetTime = fmod(targetTime, newActiveStream.duration);
            LOG_INFO("Seeking to looped position: ", targetTime);
        }
        
        // Seek the new active video to the synchronized time
        if (!SeekVideoStream(newActiveStream, targetTime)) {
            std::cerr << "Failed to synchronize new active stream to time " <<  targetTime << "\n";
            m_activeVideo = previousVideo; // Revert
            return false;
        }
        
        // Update the stream's current time
        newActiveStream.currentTime = targetTime;
        newActiveStream.state = VideoState::PLAYING;
        
        LOG_INFO("Successfully synchronized video ", (video == ActiveVideo::VIDEO_1 ? "1" : "2"), " to time ", targetTime);
    }
    
    return true;
}

bool VideoManager::UpdateFrame() {
    if (!m_initialized || m_state != VideoState::PLAYING) {
        if (!m_initialized) {
            LOG_DEBUG("UpdateFrame skipped - not initialized");
        } else if (m_state != VideoState::PLAYING) {
            LOG_DEBUG("UpdateFrame skipped - state is not PLAYING (state=", static_cast<int>(m_state), ")");
        }
        return true;
    }
    
    LOG_DEBUG("UpdateFrame called - processing...");
    
    // Handle seeking if needed
    if (m_needsSeek) {
        LOG_DEBUG("Handling seek to time: ", m_targetSeekTime);
        if (!SeekToTime(m_targetSeekTime)) {
            std::cerr << "Failed to seek to time: " << m_targetSeekTime << "\n";
            return false;
        }
        m_needsSeek = false;
    }
    
    // Check if it's time to present a new frame
    if (!ShouldPresentFrame()) {
        LOG_DEBUG("No new frame needed yet");
        return true; // No new frame needed yet
    }
    
    LOG_DEBUG("Time to present new frame");
    
    // Update playback time
    UpdatePlaybackTime();
    
    // Process frame for active video
    VideoStream& activeStream = m_videos[static_cast<int>(m_activeVideo)];
    
    if (!ProcessVideoFrame(activeStream)) {
        // Check if we reached end of stream
        if (activeStream.state == VideoState::END_OF_STREAM) {
            LOG_DEBUG("End of stream reached, handling...");
            if (!HandleEndOfStream(activeStream)) {
                return false;
            }
        } else {
            std::cerr << "Failed to process video frame\n";
            return false;
        }
    }
    
    m_lastFrameTime = std::chrono::steady_clock::now();
    LOG_DEBUG("UpdateFrame completed successfully");
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
    
    LOG_INFO("Seeking to ", timeInSeconds, " seconds");
    
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
    if (!stream.decoder.Initialize(stream.demuxer.GetCodecParameters(), decoderInfo, d3dDevice, stream.demuxer.GetTimeBase())) {
        std::cerr << "Failed to initialize decoder for: " << filePath << "\n";
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
        std::cerr << "Video resolution mismatch: " << width1 << "x" << height1 
                  << " vs " << width2 << "x" << height2 << "\n";
        return false;
    }
    
    return true;
}

bool VideoManager::ProcessVideoFrame(VideoStream& stream) {
    LOG_DEBUG("Processing video frame for ", (m_activeVideo == ActiveVideo::VIDEO_1 ? "video 1" : "video 2"));
    
    // Try to decode next frame
    if (!DecodeNextFrame(stream)) {
        LOG_DEBUG("Failed to decode next frame");
        return false;
    }
    
    LOG_DEBUG("Video frame processed successfully");
    return true;
}

bool VideoManager::DecodeNextFrame(VideoStream& stream) {
    LOG_DEBUG("Attempting to decode next frame");
    
    AVPacket packet;
    av_init_packet(&packet);
    
    bool frameDecoded = false;
    int attempts = 0;
    
    // Keep trying to decode frames until we get one or reach end of stream
    while (!frameDecoded) {
        attempts++;
        LOG_DEBUG("Decode attempt #", attempts);
        
        // Read packet from demuxer
        if (!stream.demuxer.ReadFrame(&packet)) {
            // End of stream
            LOG_DEBUG("End of stream reached");
            stream.state = VideoState::END_OF_STREAM;
            av_packet_unref(&packet);
            return false;
        }
        
        // Send packet to decoder
        if (!stream.decoder.SendPacket(&packet)) {
            LOG_DEBUG("Failed to send packet to decoder");
            av_packet_unref(&packet);
            return false;
        }
        
        // Try to receive decoded frame
        DecodedFrame newFrame;
        if (stream.decoder.ReceiveFrame(newFrame)) {
            if (newFrame.valid) {
                LOG_DEBUG("Valid frame decoded! PTS: ", newFrame.presentationTime);
                stream.currentFrame = newFrame;
                stream.currentTime = newFrame.presentationTime;
                frameDecoded = true;
            } else {
                LOG_DEBUG("Frame received but not valid");
            }
        } else {
            LOG_DEBUG("No frame received from decoder");
        }
        
        av_packet_unref(&packet);
        
        // Prevent infinite loops
        if (attempts > 100) {
            LOG_DEBUG("Too many decode attempts, giving up");
            return false;
        }
    }
    
    LOG_DEBUG("Frame decoded successfully after ", attempts, " attempts");
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

bool VideoManager::ShouldUpdateFrame() const {
    return ShouldPresentFrame();
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
    LOG_DEBUG("SeekVideoStream to ", timeInSeconds, " seconds");
    
    if (!stream.demuxer.SeekToTime(timeInSeconds)) {
        LOG_DEBUG("Demuxer seek failed");
        return false;
    }
    
    // Flush decoder to reset internal state
    stream.decoder.Flush();
    stream.currentFrame.valid = false;
    
    // Decode frames until we reach the target time or get close to it
    const double SEEK_TOLERANCE = 0.5; // 500ms tolerance for better seeking
    bool foundFrame = false;
    int attempts = 0;
    const int MAX_SEEK_ATTEMPTS = 300; // Increased for longer GOP sizes
    
    LOG_DEBUG("Looking for frame at target time ", timeInSeconds);
    
    while (!foundFrame && attempts < MAX_SEEK_ATTEMPTS) {
        attempts++;
        
        AVPacket packet;
        av_init_packet(&packet);
        
        // Read packet from demuxer
        if (!stream.demuxer.ReadFrame(&packet)) {
            LOG_DEBUG("No more packets during seek");
            av_packet_unref(&packet);
            break;
        }
        
        // Send packet to decoder
        if (!stream.decoder.SendPacket(&packet)) {
            LOG_DEBUG("Failed to send packet during seek");
            av_packet_unref(&packet);
            continue;
        }
        
        // Try to receive decoded frame
        DecodedFrame tempFrame;
        if (stream.decoder.ReceiveFrame(tempFrame)) {
            if (tempFrame.valid) {
                LOG_DEBUG("Decoded frame at time ", tempFrame.presentationTime, " (target: ", timeInSeconds, ")");
                
                // Check if this frame is close enough to our target time
                if (tempFrame.presentationTime >= timeInSeconds - SEEK_TOLERANCE) {
                    stream.currentFrame = tempFrame;
                    stream.currentTime = tempFrame.presentationTime;
                    foundFrame = true;
                    LOG_DEBUG("Found suitable frame at ", tempFrame.presentationTime, " seconds");
                }
            }
        }
        
        av_packet_unref(&packet);
    }
    
    if (!foundFrame) {
        LOG_DEBUG("Could not find frame near target time after ", attempts, " attempts");
        // Set current time anyway - we'll get the next available frame
        stream.currentTime = timeInSeconds;
    }
    
    return true;
}

void VideoManager::ResetPlaybackTiming() {
    m_playbackStartTime = std::chrono::steady_clock::now();
    m_lastFrameTime = m_playbackStartTime;
}

bool VideoManager::HandleEndOfStream(VideoStream& stream) {
    LOG_INFO("End of stream reached for ", (&stream == &m_videos[0] ? "video 1" : "video 2"));
    
    // Calculate how much we've overshot the video duration
    double overshoot = stream.currentTime - stream.duration;
    
    // Restart the video and seek to the overshoot position to maintain timing continuity
    if (!RestartVideo(stream)) {
        return false;
    }
    
    // If we overshot, seek to the overshoot position for seamless looping
    if (overshoot > 0.0 && overshoot < stream.duration) {
        LOG_INFO("Seeking to overshoot position: ", overshoot);
        if (!SeekVideoStream(stream, overshoot)) {
            std::cerr << "Failed to seek to overshoot position\n";
            // Continue anyway - not critical
        } else {
            stream.currentTime = overshoot;
        }
    }
    
    return true;
}

bool VideoManager::RestartVideo(VideoStream& stream) {
    LOG_INFO("Restarting video ", (&stream == &m_videos[0] ? "1" : "2"));
    
    // Seek back to beginning
    if (!SeekVideoStream(stream, 0.0)) {
        std::cerr << "Failed to restart video\n";
        return false;
    }
    
    stream.state = VideoState::PLAYING;
    stream.currentTime = 0.0;
    
    LOG_INFO("Video restarted successfully");
    return true;
}