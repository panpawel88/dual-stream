#pragma once

#include <memory>
#include <string>
#include <chrono>
#include "VideoDemuxer.h"
#include "VideoDecoder.h"
#include "HardwareDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <d3d11.h>

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
    
    bool Initialize(const std::string& video1Path, const std::string& video2Path, ID3D11Device* d3dDevice);
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
    
private:
    bool m_initialized;
    VideoState m_state;
    ActiveVideo m_activeVideo;
    
    VideoStream m_videos[2];
    
    // Timing management
    std::chrono::steady_clock::time_point m_playbackStartTime;
    double m_pausedTime;
    bool m_needsSeek;
    double m_targetSeekTime;
    
    // Frame timing
    std::chrono::steady_clock::time_point m_lastFrameTime;
    double m_frameInterval;
    
    // Initialization helpers
    bool InitializeVideoStream(VideoStream& stream, const std::string& filePath, ID3D11Device* d3dDevice);
    bool ValidateStreams();
    
    // Playback helpers
    bool ProcessVideoFrame(VideoStream& stream);
    bool DecodeNextFrame(VideoStream& stream);
    bool ShouldPresentFrame() const;
    void UpdatePlaybackTime();
    
    // Synchronization helpers
    bool SynchronizeStreams();
    bool SeekVideoStream(VideoStream& stream, double timeInSeconds);
    void ResetPlaybackTiming();
    
    // Loop handling
    bool HandleEndOfStream(VideoStream& stream);
    bool RestartVideo(VideoStream& stream);
};