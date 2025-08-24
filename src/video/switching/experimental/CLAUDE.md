# Video Switching Experimental Strategies

This directory contains experimental video switching strategies that provide alternatives to the default keyframe-based switching.

## Architecture

These strategies extend the `VideoSwitchingStrategy` base class and implement different approaches to video switching, each with distinct trade-offs:

## Strategy Implementations

### ImmediateSwitchStrategy
**File:** `ImmediateSwitchStrategy.h/cpp`
**Purpose:** Default switching strategy providing immediate transitions
**Approach:** Seeks target video to current timestamp and resumes playback

**Key Features:**
- Immediate switching without waiting for keyframes
- Handles video looping with time overflow calculation
- Maintains timing continuity across switches
- Uses seeking with tolerance for frame accuracy

**Trade-offs:**
- ✅ Fast switching response
- ✅ Simple implementation
- ⚠️ May cause visual artifacts during transitions
- ⚠️ Seeking accuracy depends on video keyframe distribution

**Core Methods:**
- `SwitchToVideo()` - Immediately seeks and switches to target video
- `SeekVideoStream()` - Positions video stream at specific timestamp
- `HandleEndOfStream()` - Manages video looping with overshoot handling

### PredecodedSwitchStrategy
**File:** `PredecodedSwitchStrategy.h/cpp`
**Purpose:** Zero-latency switching through simultaneous decoding
**Approach:** Decodes both video streams simultaneously, enabling instant frame switching

**Key Features:**
- Both video streams decoded in parallel
- Instant switching with no seeking delay
- Stream synchronization management
- Automatic resynchronization after stream restarts

**Trade-offs:**
- ✅ True zero-latency switching
- ✅ Perfect frame synchronization
- ❌ ~2x memory usage (both streams active)
- ❌ ~2x decoding CPU/GPU load
- ❌ Higher power consumption

**Core Methods:**
- `PredecodeBothStreams()` - Advances both video streams each frame
- `SynchronizeStreams()` - Ensures both streams are at same timestamp
- `UpdateFrame()` - Processes both streams simultaneously

## Implementation Details

### Frame Processing Pattern
All strategies follow this pattern:
1. **DecodeNextFrame()** - Reads packets and decodes video frames
2. **ProcessVideoFrame()** - Validates and processes decoded frames  
3. **HandleEndOfStream()** - Manages video looping and restart logic
4. **SeekVideoStream()** - Positions stream at target timestamp

### Timing and Synchronization
- Uses FFmpeg's presentation timestamp (PTS) for accurate timing
- Handles frame rate variations between videos
- Supports seamless looping with overshoot calculation
- Maintains playback continuity across strategy switches

### Error Handling
- Graceful degradation on decode failures
- Automatic fallback to previous video on switch failures
- Comprehensive logging for debugging
- Recovery mechanisms for stream errors

## Usage Notes

These experimental strategies are created via `VideoSwitchingStrategyFactory`:
```cpp
auto strategy = VideoSwitchingStrategyFactory::Create(SwitchingAlgorithm::IMMEDIATE);
// or
auto strategy = VideoSwitchingStrategyFactory::Create(SwitchingAlgorithm::PREDECODED);
```

The choice between strategies depends on application requirements:
- **Immediate:** Best for standard playback with acceptable transition artifacts
- **Predecoded:** Best for applications requiring perfect switching (e.g., live production)

## Future Extensions

This experimental framework enables additional strategies:
- **Keyframe-aligned switching** (implemented in parent directory)
- **Buffer-based switching** with configurable lookahead
- **Quality-adaptive switching** based on content analysis
- **Network-aware switching** for streaming applications