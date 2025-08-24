# Video Switching System

This directory implements the core video switching architecture using the Strategy pattern, enabling seamless transitions between multiple video streams.

## Architecture Overview

The switching system is built around three main components:
1. **VideoSwitchingStrategy** - Abstract base class defining switching behavior
2. **VideoSwitchingStrategyFactory** - Factory for creating strategy instances
3. **Strategy Implementations** - Concrete switching algorithms

## Core Components

### VideoSwitchingStrategy (Abstract Base)
**File:** `VideoSwitchingStrategy.h/cpp`
**Purpose:** Defines the interface for all switching strategies

**Key Interface Methods:**
```cpp
virtual bool Initialize(VideoStream* streams, VideoManager* manager) = 0;
virtual bool SwitchToVideo(ActiveVideo targetVideo, double currentTime) = 0;
virtual bool UpdateFrame() = 0;
virtual DecodedFrame* GetCurrentFrame() = 0;
virtual void Cleanup() = 0;
virtual std::string GetName() const = 0;
```

**Strategy Types:**
```cpp
enum class SwitchingAlgorithm {
    IMMEDIATE,      // Default: seek new video to current time and resume
    PREDECODED,     // Decode both streams simultaneously, switch frames instantly
    KEYFRAME_SYNC   // Queue switch requests until next synchronized keyframe
};
```

### VideoSwitchingStrategyFactory
**File:** `VideoSwitchingStrategy.cpp`
**Purpose:** Creates appropriate strategy instances

**Factory Methods:**
- `Create(SwitchingAlgorithm)` - Creates strategy instance
- `ParseAlgorithm(string)` - Parses algorithm from string name
- `GetAlgorithmName(SwitchingAlgorithm)` - Gets human-readable algorithm name

## Strategy Implementations

### KeyframeSwitchStrategy
**File:** `KeyframeSwitchStrategy.h/cpp`
**Purpose:** Quality-focused switching that waits for synchronized keyframes
**Approach:** Queues switch requests and executes them at the next keyframe boundary

**Key Features:**
- **Artifact Prevention:** Switches only at keyframes to avoid visual corruption
- **Pending Queue System:** Switch requests are queued until suitable keyframe
- **Frame Validity Control:** Prevents rendering stale frames during transitions
- **Synchronized Timing:** Ensures both videos have keyframes at same timestamps

**Trade-offs:**
- ✅ Perfect visual quality (no artifacts)
- ✅ Synchronized keyframe timing
- ✅ Prevents frame corruption
- ⚠️ Variable switching latency (depends on keyframe interval)
- ⚠️ Requires videos with synchronized keyframe placement

**Core Logic:**
```cpp
struct PendingSwitchRequest {
    ActiveVideo targetVideo;
    double requestTime;
    bool pending;
};
```

**Keyframe Detection:**
- Uses FFmpeg's `AV_FRAME_FLAG_KEY` and `AV_PICTURE_TYPE_I` flags
- Maintains `m_lastKeyframeTime` for timing synchronization
- Implements `m_switchInProgress` state to prevent frame artifacts

### Experimental Strategies
**Directory:** `experimental/`
**Documentation:** [experimental/CLAUDE.md](experimental/CLAUDE.md)

Contains alternative switching approaches:
- **ImmediateSwitchStrategy** - Default immediate switching
- **PredecodedSwitchStrategy** - Zero-latency switching via parallel decoding

## Usage Patterns

### Strategy Selection
```cpp
// Create strategy via factory
auto strategy = VideoSwitchingStrategyFactory::Create(SwitchingAlgorithm::KEYFRAME_SYNC);

// Initialize with video streams
strategy->Initialize(videoStreams, videoManager);

// Perform switching
strategy->SwitchToVideo(ActiveVideo::VIDEO_2, currentTime);

// Update frames
strategy->UpdateFrame();
DecodedFrame* frame = strategy->GetCurrentFrame();
```

### Integration with VideoManager
The VideoManager delegates all switching operations to the active strategy:
```cpp
bool VideoManager::SwitchToVideo(ActiveVideo video) {
    return m_switchingStrategy->SwitchToVideo(video, GetCurrentTime());
}

bool VideoManager::UpdateFrame() {
    return m_switchingStrategy->UpdateFrame();
}

DecodedFrame* VideoManager::GetCurrentFrame() {
    return m_switchingStrategy->GetCurrentFrame();
}
```

## Error Handling and Recovery

### Switch Failure Recovery
- Failed switches revert to previous video state
- Frame validity flags prevent rendering corrupted frames
- Comprehensive logging for debugging switch issues

### Stream Error Handling
- Automatic stream restart on end-of-stream
- Overshoot calculation for seamless looping
- Graceful degradation on decode failures

## Performance Considerations

### Memory Usage
- **Immediate/Keyframe:** Single active stream (~1x memory)
- **Predecoded:** Both streams active (~2x memory)

### CPU/GPU Load  
- **Immediate/Keyframe:** Standard decoding load
- **Predecoded:** Double decoding load for zero-latency switching

### Switching Latency
- **Immediate:** ~1-5ms (seeking overhead)
- **Keyframe:** Variable (0-2000ms depending on GOP size)
- **Predecoded:** ~0ms (instant frame switching)

## Extensibility

The Strategy pattern enables easy addition of new switching algorithms:
1. Extend `VideoSwitchingStrategy` base class
2. Implement required virtual methods
3. Add to `SwitchingAlgorithm` enum
4. Update factory creation logic

**Potential Future Strategies:**
- Buffer-based switching with configurable lookahead
- Content-aware switching based on scene detection
- Network-adaptive switching for streaming scenarios
- Multi-video switching (>2 videos) with priority queues