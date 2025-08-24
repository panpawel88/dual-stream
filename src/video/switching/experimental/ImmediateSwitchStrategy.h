#pragma once

#include "video/switching/VideoSwitchingStrategy.h"
#include "video/VideoManager.h"

class ImmediateSwitchStrategy : public VideoSwitchingStrategy {
public:
    ImmediateSwitchStrategy();
    ~ImmediateSwitchStrategy() override;
    
    bool Initialize(VideoStream* streams, VideoManager* manager) override;
    bool SwitchToVideo(ActiveVideo targetVideo, double currentTime) override;
    bool UpdateFrame() override;
    DecodedFrame* GetCurrentFrame() override;
    void Cleanup() override;
    std::string GetName() const override;
    
private:
    // Helper methods that mirror the current VideoManager implementation
    bool SeekVideoStream(VideoStream& stream, double timeInSeconds);
    bool DecodeNextFrame(VideoStream& stream);
    bool ProcessVideoFrame(VideoStream& stream);
    bool HandleEndOfStream(VideoStream& stream);
    bool RestartVideo(VideoStream& stream);
};