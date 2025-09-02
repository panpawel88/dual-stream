#include "PredecodedSwitchStrategy.h"
#include "core/Logger.h"
#include <algorithm>
#include <cmath>
#include <chrono>

extern "C" {
#include <libavcodec/avcodec.h>
}

PredecodedSwitchStrategy::PredecodedSwitchStrategy()
    : m_streamsSynchronized(false)
    , m_lastSyncTime(0.0)
    , m_lastUpdateTime(0.0) {
    m_activeVideoIndex = 0;
}

PredecodedSwitchStrategy::~PredecodedSwitchStrategy() {
    Cleanup();
}

bool PredecodedSwitchStrategy::Initialize(std::vector<VideoStream>* streams, VideoManager* manager) {
    m_streams = streams;
    m_manager = manager;
    m_activeVideoIndex = 0;
    m_streamsSynchronized = false;
    m_lastSyncTime = 0.0;
    m_lastUpdateTime = 0.0;
    
    LOG_INFO("PredecodedSwitchStrategy initialized - will decode both streams simultaneously");
    return true;
}

bool PredecodedSwitchStrategy::SwitchToVideo(size_t targetVideoIndex, double currentTime) {
    if (targetVideoIndex >= m_streams->size() || targetVideoIndex == m_activeVideoIndex) {
        return true;
    }
    
    LOG_INFO("PredecodedSwitchStrategy: Instant switch to video ", (targetVideoIndex + 1), " at time ", currentTime);
    
    // Instant switch - no seeking required since streams are predecoded
    m_activeVideoIndex = targetVideoIndex;
    
    // Ensure the target stream has a valid frame at or near the current time
    VideoStream& targetStream = (*m_streams)[m_activeVideoIndex];
    
    // If the frame is too far from current time, we might need to synchronize
    if (!targetStream.currentFrame.valid || 
        std::abs(targetStream.currentFrame.presentationTime - currentTime) > 1.0) {
        LOG_INFO("Target stream frame is out of sync, synchronizing...");
        if (!SynchronizeStreams(currentTime)) {
            LOG_ERROR("Failed to synchronize streams for instant switch");
            return false;
        }
    }
    
    LOG_INFO("Instant switch completed successfully");
    return true;
}

bool PredecodedSwitchStrategy::UpdateFrame() {
    // Update both streams to keep them synchronized
    if (!PredecodeBothStreams()) {
        LOG_ERROR("Failed to predecode both streams");
        return false;
    }
    
    return true;
}

DecodedFrame* PredecodedSwitchStrategy::GetCurrentFrame() {
    VideoStream& activeStream = (*m_streams)[m_activeVideoIndex];
    if (activeStream.currentFrame.valid) {
        return &activeStream.currentFrame;
    }
    
    return nullptr;
}

void PredecodedSwitchStrategy::Cleanup() {
    // Nothing special to cleanup for predecoded strategy
    m_streamsSynchronized = false;
}

std::string PredecodedSwitchStrategy::GetName() const {
    return "Predecoded Switch Strategy";
}

bool PredecodedSwitchStrategy::PredecodeBothStreams() {
    bool stream1Success = true;
    bool stream2Success = true;
    
    // Try to decode next frame for all streams (for now, limit to first 2 for predecoded)
    size_t numStreams = std::min(m_streams->size(), size_t(2));
    
    if (numStreams > 0) {
        if (!DecodeNextFrame((*m_streams)[0])) {
            if ((*m_streams)[0].state == VideoState::END_OF_STREAM) {
                if (!HandleEndOfStream((*m_streams)[0])) {
                    stream1Success = false;
                }
            } else {
                stream1Success = false;
            }
        }
    }
    
    if (numStreams > 1) {
        if (!DecodeNextFrame((*m_streams)[1])) {
            if ((*m_streams)[1].state == VideoState::END_OF_STREAM) {
                if (!HandleEndOfStream((*m_streams)[1])) {
                    stream2Success = false;
                }
            } else {
                stream2Success = false;
            }
        }
    }
    
    // Both streams should succeed for proper predecoded operation
    bool success = stream1Success && stream2Success;
    
    if (success) {
        LOG_DEBUG("Both streams predecoded successfully");
        if (numStreams > 0) {
            LOG_DEBUG("Stream 1 time: ", (*m_streams)[0].currentTime);
        }
        if (numStreams > 1) {
            LOG_DEBUG("Stream 2 time: ", (*m_streams)[1].currentTime);
        }
    } else {
        LOG_DEBUG("Predecoding failed - Stream 1: ", (stream1Success ? "OK" : "FAIL"), ", Stream 2: ", (stream2Success ? "OK" : "FAIL"));
    }
    
    return success;
}

bool PredecodedSwitchStrategy::DecodeNextFrame(VideoStream& stream) {
    LOG_DEBUG("Attempting to decode next frame for stream");
    
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

bool PredecodedSwitchStrategy::HandleEndOfStream(VideoStream& stream) {
    // Find which video this stream is
    size_t streamIndex = 0;
    for (size_t i = 0; i < m_streams->size(); i++) {
        if (&stream == &(*m_streams)[i]) {
            streamIndex = i;
            break;
        }
    }
    LOG_INFO("End of stream reached for video ", (streamIndex + 1));
    
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
            LOG_ERROR("Failed to seek to overshoot position");
            // Continue anyway - not critical
        } else {
            stream.currentTime = overshoot;
        }
    }
    
    // After restarting one stream, we need to resynchronize
    m_streamsSynchronized = false;
    
    return true;
}

bool PredecodedSwitchStrategy::RestartVideo(VideoStream& stream) {
    // Find which video this stream is
    size_t streamIndex = 0;
    for (size_t i = 0; i < m_streams->size(); i++) {
        if (&stream == &(*m_streams)[i]) {
            streamIndex = i;
            break;
        }
    }
    LOG_INFO("Restarting video ", (streamIndex + 1));
    
    // Seek back to beginning
    if (!SeekVideoStream(stream, 0.0)) {
        LOG_ERROR("Failed to restart video");
        return false;
    }
    
    stream.state = VideoState::PLAYING;
    stream.currentTime = 0.0;
    
    LOG_INFO("Video restarted successfully");
    return true;
}

bool PredecodedSwitchStrategy::SeekVideoStream(VideoStream& stream, double timeInSeconds) {
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

bool PredecodedSwitchStrategy::SynchronizeStreams(double targetTime) {
    LOG_INFO("Synchronizing both streams to time: ", targetTime);
    
    // Seek streams to the target time (limit to first 2 for predecoded)
    size_t numStreams = std::min(m_streams->size(), size_t(2));
    bool stream1Success = true;
    bool stream2Success = true;
    
    if (numStreams > 0) {
        stream1Success = SeekVideoStream((*m_streams)[0], targetTime);
    }
    if (numStreams > 1) {
        stream2Success = SeekVideoStream((*m_streams)[1], targetTime);
    }
    
    if (stream1Success && stream2Success) {
        m_streamsSynchronized = true;
        m_lastSyncTime = targetTime;
        LOG_INFO("Both streams synchronized successfully");
        return true;
    } else {
        m_streamsSynchronized = false;
        LOG_ERROR("Failed to synchronize streams - Stream 1: ", (stream1Success ? "OK" : "FAIL"), ", Stream 2: ", (stream2Success ? "OK" : "FAIL"));
        return false;
    }
}