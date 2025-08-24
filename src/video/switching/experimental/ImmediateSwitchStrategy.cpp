#include "ImmediateSwitchStrategy.h"
#include "core/Logger.h"
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
}

ImmediateSwitchStrategy::ImmediateSwitchStrategy() {
    m_activeVideo = ActiveVideo::VIDEO_1;
}

ImmediateSwitchStrategy::~ImmediateSwitchStrategy() {
    Cleanup();
}

bool ImmediateSwitchStrategy::Initialize(VideoStream* streams, VideoManager* manager) {
    m_streams = streams;
    m_manager = manager;
    m_activeVideo = ActiveVideo::VIDEO_1;
    
    LOG_INFO("ImmediateSwitchStrategy initialized");
    return true;
}

bool ImmediateSwitchStrategy::SwitchToVideo(ActiveVideo targetVideo, double currentTime) {
    if (targetVideo == m_activeVideo) {
        return true;
    }
    
    ActiveVideo previousVideo = m_activeVideo;
    
    LOG_INFO("ImmediateSwitchStrategy: Switching to video ", (targetVideo == ActiveVideo::VIDEO_1 ? "1" : "2"), " at time ", currentTime);
    
    // Switch active video first
    m_activeVideo = targetVideo;
    
    VideoStream& newActiveStream = m_streams[static_cast<int>(m_activeVideo)];
    
    // Handle looping if current time exceeds new video's duration
    double targetTime = currentTime;
    if (targetTime >= newActiveStream.duration && newActiveStream.duration > 0.0) {
        // Loop to the appropriate position within the video
        targetTime = fmod(targetTime, newActiveStream.duration);
        LOG_INFO("Seeking to looped position: ", targetTime);
    }
    
    // Seek the new active video to the synchronized time
    if (!SeekVideoStream(newActiveStream, targetTime)) {
        LOG_ERROR("Failed to synchronize new active stream to time ", targetTime);
        m_activeVideo = previousVideo; // Revert
        return false;
    }
    
    // Update the stream's current time
    newActiveStream.currentTime = targetTime;
    newActiveStream.state = VideoState::PLAYING;
    
    LOG_INFO("Successfully synchronized video ", (targetVideo == ActiveVideo::VIDEO_1 ? "1" : "2"), " to time ", targetTime);
    
    return true;
}

bool ImmediateSwitchStrategy::UpdateFrame() {
    VideoStream& activeStream = m_streams[static_cast<int>(m_activeVideo)];
    
    if (!ProcessVideoFrame(activeStream)) {
        // Check if we reached end of stream
        if (activeStream.state == VideoState::END_OF_STREAM) {
            LOG_DEBUG("End of stream reached, handling...");
            if (!HandleEndOfStream(activeStream)) {
                return false;
            }
        } else {
            LOG_ERROR("Failed to process video frame");
            return false;
        }
    }
    
    return true;
}

DecodedFrame* ImmediateSwitchStrategy::GetCurrentFrame() {
    VideoStream& activeStream = m_streams[static_cast<int>(m_activeVideo)];
    if (activeStream.currentFrame.valid) {
        return &activeStream.currentFrame;
    }
    
    return nullptr;
}

void ImmediateSwitchStrategy::Cleanup() {
    // Nothing special to cleanup for immediate strategy
}

std::string ImmediateSwitchStrategy::GetName() const {
    return "Immediate Switch Strategy";
}

bool ImmediateSwitchStrategy::ProcessVideoFrame(VideoStream& stream) {
    LOG_DEBUG("Processing video frame for ", (m_activeVideo == ActiveVideo::VIDEO_1 ? "video 1" : "video 2"));
    
    // Try to decode next frame
    if (!DecodeNextFrame(stream)) {
        LOG_DEBUG("Failed to decode next frame");
        return false;
    }
    
    LOG_DEBUG("Video frame processed successfully");
    return true;
}

bool ImmediateSwitchStrategy::DecodeNextFrame(VideoStream& stream) {
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

bool ImmediateSwitchStrategy::SeekVideoStream(VideoStream& stream, double timeInSeconds) {
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

bool ImmediateSwitchStrategy::HandleEndOfStream(VideoStream& stream) {
    LOG_INFO("End of stream reached for ", (&stream == &m_streams[0] ? "video 1" : "video 2"));
    
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
    
    return true;
}

bool ImmediateSwitchStrategy::RestartVideo(VideoStream& stream) {
    LOG_INFO("Restarting video ", (&stream == &m_streams[0] ? "1" : "2"));
    
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