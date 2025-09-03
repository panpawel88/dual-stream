#include "KeyframeSwitchStrategy.h"
#include "core/Logger.h"
#include "ui/NotificationManager.h"
#include <algorithm>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
}

KeyframeSwitchStrategy::KeyframeSwitchStrategy()
    : m_lastKeyframeTime(0.0)
    , m_keyframeDetected(false)
    , m_switchInProgress(false) {
    m_activeVideoIndex = 0;
}

KeyframeSwitchStrategy::~KeyframeSwitchStrategy() {
    Cleanup();
}

bool KeyframeSwitchStrategy::Initialize(std::vector<VideoStream>* streams, VideoManager* manager) {
    m_streams = streams;
    m_manager = manager;
    m_activeVideoIndex = 0;
    m_pendingSwitch.pending = false;
    m_lastKeyframeTime = 0.0;
    m_keyframeDetected = false;
    m_switchInProgress = false;
    
    
    LOG_INFO("KeyframeSwitchStrategy initialized - switches will occur at next keyframe for seamless transitions");
    return true;
}

bool KeyframeSwitchStrategy::SwitchToVideo(size_t targetVideoIndex, double currentTime) {
    if (targetVideoIndex >= m_streams->size() || targetVideoIndex == m_activeVideoIndex) {
        return true;
    }
    
    LOG_INFO("KeyframeSwitchStrategy: Queueing switch to video ", (targetVideoIndex + 1), " at time ", currentTime);
    
    // Queue the switch request - it will be executed at the next keyframe
    m_pendingSwitch.targetVideoIndex = targetVideoIndex;
    m_pendingSwitch.requestTime = currentTime;
    m_pendingSwitch.pending = true;
    
    LOG_INFO("Switch request queued, waiting for next synchronized keyframe");
    return true;
}

bool KeyframeSwitchStrategy::UpdateFrame() {
    VideoStream& activeStream = (*m_streams)[m_activeVideoIndex];
    
    // Reset keyframe detection at the beginning of each frame update
    m_keyframeDetected = false;
    
    if (!ProcessVideoFrame(activeStream)) {
        // Check if we reached end of stream
        if (activeStream.state == VideoState::END_OF_STREAM) {
            LOG_DEBUG("End of stream reached, handling...");
            if (!HandleEndOfStream(activeStream)) {
                return false;
            }
            // After handling end of stream, we already have a valid frame from restart
            // Skip the rest of processing to avoid decoding another frame
            // Check if this restarted frame is a keyframe for pending switches
            if (IsKeyframe(activeStream.currentFrame)) {
                m_keyframeDetected = true;
                m_lastKeyframeTime = activeStream.currentFrame.presentationTime;
                LOG_DEBUG("Keyframe detected at restart time: ", m_lastKeyframeTime);
            }
        } else {
            LOG_ERROR("Failed to process video frame");
            return false;
        }
    }
    
    // Check if we can execute pending switch at this frame
    // This will only execute if ProcessVideoFrame detected a keyframe in the current frame
    if (m_pendingSwitch.pending) {
        if (!CheckAndExecutePendingSwitch(activeStream)) {
            LOG_DEBUG("Pending switch not executed yet, waiting for keyframe");
        }
    }
    
    return true;
}

DecodedFrame* KeyframeSwitchStrategy::GetCurrentFrame() {
    // Safety check: Don't return any frame if a switch is in progress
    // This prevents rendering stale or corrupted frames during transitions
    if (m_switchInProgress) {
        LOG_DEBUG("Switch in progress, not returning any frame to prevent artifacts");
        return nullptr;
    }
    
    VideoStream& activeStream = (*m_streams)[m_activeVideoIndex];
    if (activeStream.currentFrame.valid) {
        return &activeStream.currentFrame;
    }
    
    return nullptr;
}

void KeyframeSwitchStrategy::Cleanup() {
    m_pendingSwitch.pending = false;
    m_keyframeDetected = false;
    m_switchInProgress = false;
}

std::string KeyframeSwitchStrategy::GetName() const {
    return "Keyframe Sync Switch Strategy";
}

bool KeyframeSwitchStrategy::HasPendingOperations() const {
    return m_pendingSwitch.pending;
}

bool KeyframeSwitchStrategy::ProcessVideoFrame(VideoStream& stream) {
    LOG_DEBUG("Processing video frame for video ", (m_activeVideoIndex + 1));
    
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
    if (!frame.valid) {
        return false;
    }
    
    // Use FFmpeg's keyframe flag - this is accurate and reliable
    bool isKeyframe = frame.keyframe;
    
    if (isKeyframe) {
        LOG_DEBUG("Detected keyframe at time: ", frame.presentationTime);
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
        
        if (ExecuteSwitch(m_pendingSwitch.targetVideoIndex, m_lastKeyframeTime)) {
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

bool KeyframeSwitchStrategy::ExecuteSwitch(size_t targetVideoIndex, double currentTime) {
    size_t previousVideoIndex = m_activeVideoIndex;
    
    LOG_INFO("Executing switch to video ", (targetVideoIndex + 1), " at keyframe time ", currentTime);
    
    // Mark switch as in progress to prevent stale frame rendering
    m_switchInProgress = true;
    
    // Immediately invalidate current frame from previous video to prevent artifacts
    VideoStream& previousStream = (*m_streams)[previousVideoIndex];
    previousStream.currentFrame.valid = false;
    
    // Switch active video
    m_activeVideoIndex = targetVideoIndex;
    
    VideoStream& newActiveStream = (*m_streams)[m_activeVideoIndex];
    
    // Handle looping if current time exceeds video duration
    double targetTime = currentTime;
    if (newActiveStream.duration > 0.0 && targetTime >= newActiveStream.duration) {
        // Loop to the appropriate position within the video
        targetTime = fmod(targetTime, newActiveStream.duration);
        LOG_INFO("Looping target time to: ", targetTime, " (from: ", currentTime, ", duration: ", newActiveStream.duration, ")");
    }
    
    // Seek the new active video to the synchronized keyframe time
    // Since we're switching at a keyframe, the target video should have a keyframe at the same timestamp
    if (!SeekVideoStream(newActiveStream, targetTime)) {
        LOG_ERROR("Failed to synchronize new active stream to keyframe time ", targetTime);
        m_activeVideoIndex = previousVideoIndex; // Revert
        previousStream.currentFrame.valid = true; // Restore previous frame validity
        m_switchInProgress = false;
        return false;
    }
    
    // Decode at least one frame to ensure we have a valid frame ready before completing the switch
    if (!DecodeNextFrame(newActiveStream) || !newActiveStream.currentFrame.valid) {
        LOG_ERROR("Failed to decode valid frame after seek for new active video");
        m_activeVideoIndex = previousVideoIndex; // Revert
        previousStream.currentFrame.valid = true; // Restore previous frame validity
        m_switchInProgress = false;
        return false;
    }
    
    // Update the stream's state
    newActiveStream.state = VideoState::PLAYING;
    
    // Switch completed successfully - clear switch in progress flag
    m_switchInProgress = false;

    LOG_INFO("Successfully synchronized video ", (targetVideoIndex + 1), " to keyframe time ", newActiveStream.currentTime);
    
    // Show notification for successful switch
    NotificationManager::GetInstance().ShowSuccess(
        "Video Switch", 
        "Switched to Video " + std::to_string(targetVideoIndex + 1)
    );
    
    return true;
}

bool KeyframeSwitchStrategy::HandleEndOfStream(VideoStream& stream) {
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
            // Decode a frame at the overshoot position to prevent artifacts
            if (!DecodeNextFrame(stream) || !stream.currentFrame.valid) {
                LOG_ERROR("Failed to decode frame at overshoot position");
                // Continue anyway - not critical, we have the frame from restart
            } else {
                stream.currentTime = overshoot;
            }
        }
    }
    
    return true;
}

bool KeyframeSwitchStrategy::RestartVideo(VideoStream& stream) {
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
    
    // Decode the first frame immediately to prevent stale texture artifacts
    if (!DecodeNextFrame(stream) || !stream.currentFrame.valid) {
        LOG_ERROR("Failed to decode first frame after video restart");
        return false;
    }
    
    stream.state = VideoState::PLAYING;
    stream.currentTime = 0.0;
    
    LOG_INFO("Video restarted successfully with first frame decoded");
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
    
    // For keyframe switching, we assume both videos have keyframes at the same timestamps
    // So we can rely on the demuxer seek to position us at the right keyframe
    // The next DecodeNextFrame call should give us the frame at the target time
    stream.currentTime = timeInSeconds;
    
    LOG_DEBUG("Stream positioned at time: ", timeInSeconds, " seconds");
    return true;
}
