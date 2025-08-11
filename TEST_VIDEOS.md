# Test Videos for FFmpeg Video Player

This document describes how to generate and use test videos for the FFmpeg Video Player application.

## Quick Start

### Windows
```bash
generate_test_videos.bat
```

### Linux/macOS
```bash
./generate_test_videos.sh
```

## Generated Test Videos

All videos are created with:
- **Resolution**: 1280x720 (HD)
- **Container**: MP4
- **Codecs**: H.264 and H.265 (HEVC)
- **Pixel Format**: YUV420P (compatible with hardware decoders)

### Main Test Videos

1. **video1_red_square.mp4** (H264, 10s, 30fps)
   - Red background with animated white square
   - **Frame counter**: "Video 1 - Frame XXX" in top-left
   - Tests basic H264 decoding and animation

2. **video2_blue_circle.mp4** (H264, 10s, 30fps)  
   - Blue background with animated yellow box
   - **Frame counter**: "Video 2 - Frame XXX" in yellow text
   - Tests H264 with different motion patterns

3. **video3_gradient.mp4** (H265, 8s, 30fps)
   - Animated color gradient patterns
   - **Frame counter**: "Video 3 H265 - Frame XXX - Time: HH:MM:SS.ms"
   - Tests H265/HEVC hardware decoding

4. **video4_bouncing_ball.mp4** (H264, 12s, 30fps)
   - Green background with bouncing red ball
   - **Frame counter**: "Video 4 - Frame XXX - FPS: 30"
   - Tests longer duration playback

5. **video5_text.mp4** (H265, 15s, 30fps)
   - Animated text "FFmpeg Video Player"
   - **Frame counter**: "Video 5 H265 - Frame XXX" in yellow text
   - Tests text rendering and H265

### Quick Test Videos (3 seconds each)

6. **short_a_red_fade.mp4** (H264, 3s, 60fps)
   - Red screen with fade in/out
   - **Frame counter**: "SHORT A - Frame XXX (60fps)" at 60fps
   - Fast testing for development

7. **short_b_blue_pulse.mp4** (H264, 3s, 60fps)
   - Blue screen with pulsing white box
   - **Frame counter**: "SHORT B - Frame XXX (60fps)" in cyan text
   - Quick switching test

## Frame Number Verification

Each test video includes **embedded frame counters** that help verify:

### ✅ Video Playback Status
- **Static frame**: Frame number doesn't change → Video is paused/frozen
- **Advancing frames**: Frame numbers increment → Video is playing correctly
- **Frame rate verification**: Count frame advances over time to verify FPS

### ✅ Video Switching
- **Immediate visual change**: Different video identifiers ("Video 1" vs "Video 2")
- **Frame continuity**: Frame numbers should continue from current playback position
- **Synchronization**: Both videos should show similar time positions when switching

### ✅ Hardware vs Software Decoding
- **Performance difference**: Hardware decoding should maintain smooth frame advancement
- **Frame drops**: Missing frame numbers indicate decode performance issues

### Frame Counter Examples:
```
Video 1 - Frame 150     ← 5 seconds at 30fps
Video 2 - Frame 75      ← 2.5 seconds at 30fps  
SHORT A - Frame 120 (60fps) ← 2 seconds at 60fps
Video 3 H265 - Frame 45 - Time: 00:00:01.50 ← With timestamp
```

## Usage Examples

### Basic Testing
```bash
# Test H264 videos
ffmpeg_player.exe test_videos/video1_red_square.mp4 test_videos/video2_blue_circle.mp4

# Test mixed H264/H265
ffmpeg_player.exe test_videos/video1_red_square.mp4 test_videos/video3_gradient.mp4

# Quick testing
ffmpeg_player.exe test_videos/short_a_red_fade.mp4 test_videos/short_b_blue_pulse.mp4
```

### Controls
- **Key 1**: Switch to first video
- **Key 2**: Switch to second video  
- **ESC**: Exit application

## Hardware Decoding Test

The application will automatically:
1. Detect NVIDIA NVDEC capability
2. Use hardware decoding for H264/H265 when available
3. Fall back to software decoding if hardware unavailable
4. Display decoder type in console output

## Expected Console Output

```
FFmpeg Video Player v1.0.0
FFmpeg version: ...
Initializing hardware decoder detection...
H264 NVDEC decoder found
H265 NVDEC decoder found
NVDEC hardware decoding available
Available decoders:
  - NVIDIA NVDEC (Available)
  - Software (Available)

Video info for test_videos/video1_red_square.mp4:
  Resolution: 1280x720
  Frame rate: 30 FPS
  Codec: h264
  Duration: 10 seconds

Initializing video decoder with NVIDIA NVDEC
Hardware decoding enabled
```

## Troubleshooting

### FFmpeg Not Found
- **Windows**: Download from https://ffmpeg.org/download.html
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### Hardware Decoding Issues
- Ensure NVIDIA GPU with NVDEC support
- Update graphics drivers
- Check CUDA toolkit installation

### Video Generation Errors
- Verify FFmpeg installation
- Check available disk space
- Ensure write permissions in directory

## File Specifications

All generated videos meet the application requirements:
- ✅ MP4 container format
- ✅ H264 or H265 video codec  
- ✅ Identical resolutions (1280x720)
- ✅ Compatible pixel format (YUV420P)
- ✅ Reasonable duration for testing
- ✅ Proper keyframe structure for seeking