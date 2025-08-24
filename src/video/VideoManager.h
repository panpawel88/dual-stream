#pragma once

#include <memory>
#include <string>
#include <chrono>
#include "demux/VideoDemuxer.h"
#include "decode/VideoDecoder.h"
#include "decode/HardwareDecoder.h"
#include "switching/VideoSwitchingStrategy.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

// Forward declarations to avoid including renderer headers
class IRenderer;
struct ID3D11Device;

enum class VideoState {
    STOPPED,
    PLAYING,
    PAUSED,
    END_OF_STREAM
};

enum class ActiveVideo {
    VIDEO_1 = 0,
    VIDEO_2 = 1
};

struct VideoStream {
    VideoDemuxer demuxer;
    VideoDecoder decoder;
    DecodedFrame currentFrame;
    VideoState state;
    double currentTime;
    double duration;
    bool initialized;
    
    VideoStream() : state(VideoState::STOPPED), currentTime(0.0), duration(0.0), initialized(false) {}
};

class VideoManager {
public:
    VideoManager();
    ~VideoManager();
    
    bool Initialize(const std::string& video1Path, const std::string& video2Path, IRenderer* renderer, SwitchingAlgorithm switchingAlgorithm = SwitchingAlgorithm::IMMEDIATE, double playbackSpeed = 1.0);
    void Cleanup();
    
    // Playback control
    bool Play();
    bool Pause();
    bool Stop();
    bool SwitchToVideo(ActiveVideo video);
    
    // Frame processing
    bool UpdateFrame();
    DecodedFrame* GetCurrentFrame();
    
    // Timing and synchronization
    double GetCurrentTime() const;
    double GetDuration() const;
    bool SeekToTime(double timeInSeconds);
    
    // Status
    VideoState GetState() const { return m_state; }
    ActiveVideo GetActiveVideo() const { return m_activeVideo; }
    bool IsInitialized() const { return m_initialized; }
    
    // Frame timing
    double GetFrameInterval() const { return m_frameInterval; }
    bool ShouldUpdateFrame() const;
    
private:
    bool m_initialized;
    VideoState m_state;
    ActiveVideo m_activeVideo;
    
    VideoStream m_videos[2];
    
    // Switching strategy
    std::unique_ptr<VideoSwitchingStrategy> m_switchingStrategy;
    
    // Timing management
    std::chrono::steady_clock::time_point m_playbackStartTime;
    double m_pausedTime;

    // Frame timing
    std::chrono::steady_clock::time_point m_lastFrameTime;
    double m_frameInterval;
    double m_playbackSpeed;
    
    // Initialization helpers
    bool InitializeVideoStream(VideoStream& stream, const std::string& filePath, ID3D11Device* d3dDevice, bool cudaInteropAvailable = false);
    bool ValidateStreams();
    
    // Playback helpers
    bool ShouldPresentFrame() const;

    void ResetPlaybackTiming();
};