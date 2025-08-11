# FFmpeg Video Player Application

## Project Requirements

- C++ application for Windows 10/11
- Build system: CMake with FFmpeg 7.1.1 download from gyan.dev during generation
- Input: Paths to 2 video files via command line arguments (MP4 container, H264/H265 codec)
- Video files must have identical resolutions (error and exit if different)
- Hardware decoding using NVIDIA RTX GPU (NVDEC) with software fallback
- Console notification when software decoding is used instead of hardware
- DirectX 11 rendering with decoded frames as DX11 textures
- Fixed window size matching video resolution (non-resizable)
- Seamless video switching with keyboard keys (1 and 2)
- Auto-play with individual video looping (each video restarts itself at end)
- Synchronized timestamp when switching between videos
- Minimal resource usage - decode on-demand, not in advance
- No audio support currently, but architecture flexible for future separate audio file

## Application Architecture

### Core Components
1. **CMake Build System** - Downloads FFmpeg binaries and sets up dependencies
2. **Video Decoder Manager** - Handles hardware decoding with NVDEC to DX11 textures
3. **Dual Stream Manager** - Manages two video streams with seamless switching
4. **DirectX 11 Renderer** - Renders decoded frames to window
5. **Input Handler** - Processes keyboard input for video switching
6. **Frame Synchronizer** - Maintains proper timing and seeks to keyframes

### Key Technical Features
- Hardware decoding using NVIDIA's NVDEC API through FFmpeg
- Direct DX11 texture output from decoder (avoiding CPU-GPU memory transfers)
- Intelligent buffering strategy for seamless switching
- Keyframe-aware seeking for proper decode state
- Minimal resource usage with on-demand decoding

### Memory Management Strategy
- Decode frames just-in-time, not in advance
- Maintain small buffer around current playback position
- Pre-seek inactive video to current timestamp when switching
- Release unused decoder contexts promptly

### Error Handling Strategy
- Validate identical video resolutions at startup (exit with error if different)
- Exit with error message if video files are corrupted, missing, or incompatible
- Graceful fallback from hardware to software decoding with console notification
- Clear error messages for all failure scenarios

## Todo List

1. ✅ Set up CMake build system with FFmpeg 7.1.1 download from gyan.dev
2. ✅ Create basic C++ project structure and main window using Win32 API
3. ✅ Implement command line argument parsing for two video file paths
4. ✅ Add video resolution validation (ensure both videos have same dimensions)
5. ✅ Implement FFmpeg initialization and hardware decoder detection (NVDEC with software fallback)
6. ✅ Create video demuxer class for MP4 containers with H264/H265 support
7. ✅ Implement hardware decoder wrapper with DX11 texture output
8. ✅ Set up DirectX 11 rendering pipeline for video textures
9. ✅ Create dual video manager for handling two synchronized video streams
10. ✅ Implement seamless video switching with keyframe-aware seeking
11. ✅ Add keyboard input handling (keys 1 and 2 for video switching)
12. 🔄 Implement frame timing, synchronization, and individual video looping
13. ✅ Add error handling and resource management
14. 🔄 Test and optimize for minimal resource usage during playback

## Implementation Status

**Completed (13/14):**
- CMake build system with automatic FFmpeg 7.1.1 download from gyan.dev and optional CUDA support
- Win32 window creation with fixed size and keyboard input handling (keys 1/2/ESC)
- Command line argument parsing with MP4 file validation and existence checks
- Video resolution validation using FFmpeg (identical dimensions, H264/H265 codec support)
- Hardware decoder detection (NVDEC) with software fallback implementation
- Video demuxer class for MP4 containers with H264/H265 codec support
- Hardware decoder wrapper with DirectX 11 texture output capability
- DirectX 11 rendering pipeline for video textures (D3D11Renderer component)
- Dual video manager for handling two synchronized video streams (VideoManager component)
- Seamless video switching with keyframe-aware seeking
- Keyboard input handling (keys 1 and 2 for video switching)
- Error handling and resource management
- Test video generation scripts and sample videos

**Currently Working On:**
- Frame timing, synchronization, and individual video looping
- Final testing and optimization for minimal resource usage

**Project Structure:**
```
src/
├── main.cpp                 - Application entry point and orchestration
├── Window.h/cpp            - Win32 window management and input handling
├── CommandLineParser.h/cpp - Command line argument validation
├── VideoValidator.h/cpp    - FFmpeg-based video file validation
├── HardwareDecoder.h/cpp   - Hardware decoder detection (NVDEC/software)
├── VideoDemuxer.h/cpp      - MP4 demuxing and stream management
├── VideoDecoder.h/cpp      - Hardware/software decoder with DX11 texture output
├── D3D11Renderer.h/cpp     - DirectX 11 rendering pipeline for video textures
└── VideoManager.h/cpp      - Dual video stream management with seamless switching

Additional Files:
├── CMakeLists.txt          - Build configuration with FFmpeg integration
├── TEST_VIDEOS.md          - Test video documentation
├── generate_test_videos.*  - Scripts for generating test videos
└── test_videos/            - Sample MP4 videos for testing
```

## Implementation Notes

The application will:
- Accept two video file paths via command line arguments
- Validate that both videos have identical resolutions at startup
- Create a fixed-size window matching video resolution (non-resizable)
- Start with the first video in auto-play mode
- Allow instant switching between videos (keys 1 and 2) while maintaining synchronized timestamps
- Loop each video individually when it reaches the end
- Use hardware decoding (NVDEC) when available, with console notification for software fallback
- Leverage GPU hardware acceleration with direct DX11 texture output for optimal performance
- Maintain architecture flexibility for future separate audio file support