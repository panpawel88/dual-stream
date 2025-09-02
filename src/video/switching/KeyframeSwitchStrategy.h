#pragma once

#include "VideoSwitchingStrategy.h"
#include "video/VideoManager.h"

struct PendingSwitchRequest {
    size_t targetVideoIndex;
    double requestTime;
    bool pending;
    
    PendingSwitchRequest() : targetVideoIndex(0), requestTime(0.0), pending(false) {}
};

class KeyframeSwitchStrategy : public VideoSwitchingStrategy {
public:
    KeyframeSwitchStrategy();
    ~KeyframeSwitchStrategy() override;
    
    bool Initialize(std::vector<VideoStream>* streams, VideoManager* manager) override;
    bool SwitchToVideo(size_t targetVideoIndex, double currentTime) override;
    bool UpdateFrame() override;
    DecodedFrame* GetCurrentFrame() override;
    void Cleanup() override;
    std::string GetName() const override;
    bool HasPendingOperations() const override;
    
private:
    bool ProcessVideoFrame(VideoStream& stream);
    bool DecodeNextFrame(VideoStream& stream);
    bool HandleEndOfStream(VideoStream& stream);
    bool RestartVideo(VideoStream& stream);
    bool SeekVideoStream(VideoStream& stream, double timeInSeconds);
    
    // Keyframe detection and switching logic
    bool IsKeyframe(const DecodedFrame& frame);
    bool CheckAndExecutePendingSwitch(VideoStream& stream);
    bool ExecuteSwitch(size_t targetVideoIndex, double currentTime);
    
    // Pending switch management
    PendingSwitchRequest m_pendingSwitch;
    
    // Keyframe synchronization
    double m_lastKeyframeTime;
    bool m_keyframeDetected;
    
    // Switch state tracking to prevent stale frame rendering
    bool m_switchInProgress;
};