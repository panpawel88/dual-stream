#pragma once

#include "VideoDecoder.h"
#include <string>

enum class SwitchingAlgorithm {
    IMMEDIATE,      // Current default: seek new video to current time and resume
    PREDECODED,     // Decode both streams simultaneously, switch frames instantly
    KEYFRAME_SYNC   // Queue switch requests until next synchronized keyframe
};

// Forward declarations
class VideoManager;
struct VideoStream;
enum class ActiveVideo;

class VideoSwitchingStrategy {
public:
    virtual ~VideoSwitchingStrategy() = default;
    
    // Initialize the strategy with video streams
    virtual bool Initialize(VideoStream* streams, VideoManager* manager) = 0;
    
    // Handle video switching request
    virtual bool SwitchToVideo(ActiveVideo targetVideo, double currentTime) = 0;
    
    // Update frame processing for the strategy
    virtual bool UpdateFrame() = 0;
    
    // Get current frame based on strategy
    virtual DecodedFrame* GetCurrentFrame() = 0;
    
    // Cleanup strategy resources
    virtual void Cleanup() = 0;
    
    // Get strategy name for debugging
    virtual std::string GetName() const = 0;
    
    // Check if strategy has pending operations
    virtual bool HasPendingOperations() const { return false; }
    
protected:
    VideoStream* m_streams = nullptr;
    VideoManager* m_manager = nullptr;
    ActiveVideo m_activeVideo;
};

// Factory function to create strategies
class VideoSwitchingStrategyFactory {
public:
    static std::unique_ptr<VideoSwitchingStrategy> Create(SwitchingAlgorithm algorithm);
    static SwitchingAlgorithm ParseAlgorithm(const std::string& algorithmName);
    static std::string GetAlgorithmName(SwitchingAlgorithm algorithm);
};