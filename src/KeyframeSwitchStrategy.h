#pragma once

#include "VideoSwitchingStrategy.h"
#include "VideoManager.h"

struct PendingSwitchRequest {
    ActiveVideo targetVideo;
    double requestTime;
    bool pending;
    
    PendingSwitchRequest() : targetVideo(ActiveVideo::VIDEO_1), requestTime(0.0), pending(false) {}
};

class KeyframeSwitchStrategy : public VideoSwitchingStrategy {
public:
    KeyframeSwitchStrategy();
    ~KeyframeSwitchStrategy() override;
    
    bool Initialize(VideoStream* streams, VideoManager* manager) override;
    bool SwitchToVideo(ActiveVideo targetVideo, double currentTime) override;
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
    bool ExecuteSwitch(ActiveVideo targetVideo, double currentTime);
    
    // Pending switch management
    PendingSwitchRequest m_pendingSwitch;
    
    // Keyframe synchronization
    double m_lastKeyframeTime;
    bool m_keyframeDetected;
};