#pragma once

#include "VideoSwitchingStrategy.h"
#include "../VideoManager.h"

class SingleVideoStrategy : public VideoSwitchingStrategy {
public:
    SingleVideoStrategy();
    ~SingleVideoStrategy();

    bool Initialize(std::vector<VideoStream>* streams, VideoManager* manager) override;
    bool SwitchToVideo(size_t targetVideoIndex, double currentTime) override;
    bool UpdateFrame() override;
    DecodedFrame* GetCurrentFrame() override;
    void Cleanup() override;
    std::string GetName() const override;

private:
    bool ProcessVideoFrame(VideoStream& stream);
    bool HandleEndOfStream(VideoStream& stream);
    bool SeekVideoStream(VideoStream& stream, double targetTime);
    bool DecodeNextFrame(VideoStream& stream);
};