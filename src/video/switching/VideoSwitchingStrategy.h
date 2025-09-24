#pragma once

#include "video/decode/VideoDecoder.h"
#include <string>
#include <vector>

enum class SwitchingAlgorithm {
    IMMEDIATE,      // Current default: seek new video to current time and resume
    PREDECODED,     // Decode both streams simultaneously, switch frames instantly
    KEYFRAME_SYNC   // Queue switch requests until next synchronized keyframe
};

// Forward declarations
class VideoManager;
struct VideoStream;

class VideoSwitchingStrategy {
public:
    virtual ~VideoSwitchingStrategy() = default;
    
    // Initialize the strategy with video streams
    virtual bool Initialize(std::vector<VideoStream>* streams, VideoManager* manager) = 0;
    
    // Handle video switching request
    virtual bool SwitchToVideo(size_t targetVideoIndex, double currentTime) = 0;
    
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
    std::vector<VideoStream>* m_streams = nullptr;
    VideoManager* m_manager = nullptr;
    size_t m_activeVideoIndex = 0;
};

// Factory function to create strategies
class VideoSwitchingStrategyFactory {
public:
    static std::unique_ptr<VideoSwitchingStrategy> Create(SwitchingAlgorithm algorithm);
    static std::unique_ptr<VideoSwitchingStrategy> Create(SwitchingAlgorithm algorithm, size_t videoCount);
    static SwitchingAlgorithm ParseAlgorithm(const std::string& algorithmName);
    static std::string GetAlgorithmName(SwitchingAlgorithm algorithm);
};