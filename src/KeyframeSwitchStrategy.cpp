#include "KeyframeSwitchStrategy.h"
#include "Logger.h"
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
}

KeyframeSwitchStrategy::KeyframeSwitchStrategy()
    : m_lastKeyframeTime(0.0)
    , m_keyframeDetected(false) {
    m_activeVideo = ActiveVideo::VIDEO_1;
}

KeyframeSwitchStrategy::~KeyframeSwitchStrategy() {
    Cleanup();
}

bool KeyframeSwitchStrategy::Initialize(VideoStream* streams, VideoManager* manager) {
    m_streams = streams;
    m_manager = manager;
    m_activeVideo = ActiveVideo::VIDEO_1;
    m_pendingSwitch.pending = false;
    m_lastKeyframeTime = 0.0;
    m_keyframeDetected = false;
    
    LOG_INFO("KeyframeSwitchStrategy initialized - switches will occur at synchronized keyframes");
    return true;
}

bool KeyframeSwitchStrategy::SwitchToVideo(ActiveVideo targetVideo, double currentTime) {
    if (targetVideo == m_activeVideo) {
        return true;
    }
    
    LOG_INFO("KeyframeSwitchStrategy: Queueing switch to video ", (targetVideo == ActiveVideo::VIDEO_1 ? "1" : "2"), " at time ", currentTime);
    
    // Queue the switch request - it will be executed at the next keyframe
    m_pendingSwitch.targetVideo = targetVideo;
    m_pendingSwitch.requestTime = currentTime;
    m_pendingSwitch.pending = true;
    
    LOG_INFO("Switch request queued, waiting for next synchronized keyframe");
    return true;
}

bool KeyframeSwitchStrategy::UpdateFrame() {
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
    
    // Check if we can execute pending switch at this frame
    if (m_pendingSwitch.pending) {
        if (!CheckAndExecutePendingSwitch(activeStream)) {
            LOG_DEBUG("Pending switch not executed yet, waiting for keyframe");
        }
    }
    
    return true;
}

DecodedFrame* KeyframeSwitchStrategy::GetCurrentFrame() {
    VideoStream& activeStream = m_streams[static_cast<int>(m_activeVideo)];
    if (activeStream.currentFrame.valid) {
        return &activeStream.currentFrame;
    }
    
    return nullptr;
}

void KeyframeSwitchStrategy::Cleanup() {
    m_pendingSwitch.pending = false;
    m_keyframeDetected = false;
}

std::string KeyframeSwitchStrategy::GetName() const {
    return "Keyframe Sync Switch Strategy";
}

bool KeyframeSwitchStrategy::HasPendingOperations() const {
    return m_pendingSwitch.pending;
}

bool KeyframeSwitchStrategy::ProcessVideoFrame(VideoStream& stream) {
    LOG_DEBUG("Processing video frame for ", (m_activeVideo == ActiveVideo::VIDEO_1 ? "video 1" : "video 2"));
    
    // Try to decode next frame
    if (!DecodeNextFrame(stream)) {
        LOG_DEBUG("Failed to decode next frame");
        return false;
    }
    
    // Check if this is a keyframe
    if (IsKeyframe(stream.currentFrame)) {
        m_keyframeDetected = true;
        m_lastKeyframeTime = stream.currentFrame.presentationTime;
        LOG_DEBUG("Keyframe detected at time: ", m_lastKeyframeTime);
    }
    
    LOG_DEBUG("Video frame processed successfully");
    return true;
}

bool KeyframeSwitchStrategy::DecodeNextFrame(VideoStream& stream) {
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

bool KeyframeSwitchStrategy::IsKeyframe(const DecodedFrame& frame) {
    // In this simplified implementation, we'll consider every frame that's
    // reasonably close to a regular keyframe interval (e.g., every 2 seconds)
    // to be a potential switch point. In a more sophisticated implementation,
    // you would check the actual frame flags or packet flags for keyframes.
    
    if (!frame.valid) {
        return false;
    }
    
    // Assume keyframes occur every 2 seconds for H264/H265 content
    const double KEYFRAME_INTERVAL = 2.0;
    double frameTime = frame.presentationTime;
    double modTime = fmod(frameTime, KEYFRAME_INTERVAL);
    
    // Consider frames within 100ms of the expected keyframe time as keyframes
    bool isKeyframe = (modTime < 0.1) || (modTime > KEYFRAME_INTERVAL - 0.1);
    
    if (isKeyframe) {
        LOG_DEBUG("Detected keyframe at time: ", frameTime);
    }
    
    return isKeyframe;
}

bool KeyframeSwitchStrategy::CheckAndExecutePendingSwitch(VideoStream& stream) {
    if (!m_pendingSwitch.pending) {
        return false;
    }
    
    // Execute switch if we detected a keyframe
    if (m_keyframeDetected) {
        LOG_INFO("Executing pending switch at keyframe (time: ", m_lastKeyframeTime, ")");
        
        if (ExecuteSwitch(m_pendingSwitch.targetVideo, m_lastKeyframeTime)) {
            m_pendingSwitch.pending = false;
            m_keyframeDetected = false;
            LOG_INFO("Keyframe-synchronized switch completed successfully");
            return true;
        } else {
            LOG_ERROR("Failed to execute pending switch");
            return false;
        }
    }
    
    return false;
}

bool KeyframeSwitchStrategy::ExecuteSwitch(ActiveVideo targetVideo, double currentTime) {
    ActiveVideo previousVideo = m_activeVideo;
    
    LOG_INFO("Executing switch to video ", (targetVideo == ActiveVideo::VIDEO_1 ? "1" : "2"), " at keyframe time ", currentTime);
    
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
    
    LOG_INFO("Successfully synchronized video ", (targetVideo == ActiveVideo::VIDEO_1 ? "1" : "2"), " to keyframe time ", targetTime);
    
    return true;
}

bool KeyframeSwitchStrategy::HandleEndOfStream(VideoStream& stream) {
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

bool KeyframeSwitchStrategy::RestartVideo(VideoStream& stream) {
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

bool KeyframeSwitchStrategy::SeekVideoStream(VideoStream& stream, double timeInSeconds) {
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