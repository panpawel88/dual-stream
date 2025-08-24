#pragma once

#include "video/switching/VideoSwitchingStrategy.h"
#include "video/VideoManager.h"

class PredecodedSwitchStrategy : public VideoSwitchingStrategy {
public:
    PredecodedSwitchStrategy();
    ~PredecodedSwitchStrategy() override;
    
    bool Initialize(VideoStream* streams, VideoManager* manager) override;
    bool SwitchToVideo(ActiveVideo targetVideo, double currentTime) override;
    bool UpdateFrame() override;
    DecodedFrame* GetCurrentFrame() override;
    void Cleanup() override;
    std::string GetName() const override;
    
private:
    // Predecode both streams simultaneously
    bool PredecodeBothStreams();
    bool DecodeNextFrame(VideoStream& stream);
    bool HandleEndOfStream(VideoStream& stream);
    bool RestartVideo(VideoStream& stream);
    bool SeekVideoStream(VideoStream& stream, double timeInSeconds);
    bool SynchronizeStreams(double targetTime);
    
    // Track if streams are synchronized
    bool m_streamsSynchronized;
    double m_lastSyncTime;
    
    // Frame timing
    double m_lastUpdateTime;
};