# Tracy Profiler Integration

This document outlines the Tracy profiler integration for the DualStream Video Player application, providing comprehensive performance analysis capabilities.

## Overview

Tracy is a real-time, nanosecond resolution frame profiler that provides detailed insights into CPU and GPU performance. This integration enables developers to identify performance bottlenecks and optimize the video player's rendering pipeline.

## Integration Status

### ✅ **Completed Features**

#### 1. **CMake Integration** ✅
- **File:** `CMakeLists.txt`
- **Features:**
  - Tracy v0.10 integration via FetchContent
  - `ENABLE_TRACY` CMake option (OFF by default)
  - Configured with `ON_DEMAND`, `NO_FRAME_IMAGE`, `NO_SYSTEM_TRACING`
  - Automatic linking to `dual_stream_core` target

**Build with Tracy:**
```bash
cmake -DENABLE_TRACY=ON ..
cmake --build . --config Release
```

#### 2. **Core Profiling Infrastructure** ✅
- **File:** `src/core/TracyProfiler.h`, `src/core/TracyProfiler.cpp`
- **Features:**
  - Zero-overhead macro abstraction when Tracy is disabled
  - CPU profiling macros (`PROFILE_ZONE`, `PROFILE_ZONE_N`)
  - GPU profiling macros for DirectX 11 and OpenGL
  - Memory profiling, message logging, and lock profiling
  - Domain-specific convenience macros for video player subsystems
  - `TracyProfilerManager` singleton for GPU context management

**Usage Examples:**
```cpp
// CPU profiling
PROFILE_ZONE();
PROFILE_ZONE_N("CustomName");

// GPU profiling - DirectX 11
PROFILE_GPU_D3D11_ZONE("RenderPass");
PROFILE_GPU_D3D11_COLLECT();

// GPU profiling - OpenGL
PROFILE_GPU_OPENGL_ZONE("RenderPass");
PROFILE_GPU_OPENGL_COLLECT();

// Domain-specific macros
PROFILE_VIDEO_DECODE();
PROFILE_RENDER();
PROFILE_CAMERA_CAPTURE();
```

#### 3. **Automatic Instrumentation Script** ✅
- **File:** `tools/tracy_instrumentor.py`
- **Features:**
  - Intelligent C++ function detection using regex patterns
  - Multiple operation modes: `auto`, `preview`, `clean`, `restore`
  - Backup and restore functionality
  - Skips simple getters/setters and already instrumented code
  - Subsystem-specific zone naming

**Usage:**
```bash
# Preview changes without modifying files
python tools/tracy_instrumentor.py --mode=preview

# Automatically instrument all files with backup
python tools/tracy_instrumentor.py --mode=auto --backup

# Clean Tracy instrumentation from files
python tools/tracy_instrumentor.py --mode=clean

# Restore files from backup
python tools/tracy_instrumentor.py --mode=restore
```

#### 4. **DirectX 11 Renderer Profiling** ✅
- **File:** `src/rendering/D3D11Renderer.h`, `src/rendering/D3D11Renderer.cpp`
- **Integration:**
  - Tracy GPU context initialization in `Initialize()` method
  - CPU profiling zone in `Present()` method
  - GPU profiling zones for render pass pipeline execution
  - GPU data collection after swap chain present
  - Conditional compilation with `#ifdef TRACY_ENABLE`

**Profiling Points:**
- `PROFILE_RENDER()` - Main rendering function
- `PROFILE_GPU_D3D11_ZONE("RenderPassPipeline")` - Render pass execution
- `PROFILE_GPU_D3D11_COLLECT()` - GPU timing data collection

#### 5. **OpenGL Renderer Profiling** ✅
- **File:** `src/rendering/OpenGLRenderer.h`, `src/rendering/OpenGLRenderer.cpp`
- **Integration:**
  - Tracy GPU context initialization after OpenGL setup
  - CPU profiling zone in `Present()` method
  - GPU profiling zones for render pass pipeline execution
  - GPU data collection after SwapBuffers call
  - Conditional compilation with `#ifdef TRACY_ENABLE`

**Profiling Points:**
- `PROFILE_RENDER()` - Main rendering function
- `PROFILE_GPU_OPENGL_ZONE("RenderPassPipeline")` - Render pass execution
- `PROFILE_GPU_OPENGL_COLLECT()` - GPU timing data collection

### 🔄 **In Progress**

#### 6. **CUDA Profiling Support** 🔄
- **Target:** CUDA-OpenGL interop profiling
- **Scope:** Add Tracy CUDA profiling for hardware decode → OpenGL texture path
- **Files:** `src/rendering/CudaOpenGLInterop.h/cpp`
- **Planned Features:**
  - CUDA kernel profiling for YUV→RGB conversion
  - Memory transfer profiling for CUDA↔OpenGL operations
  - Integration with existing OpenGL profiling context

### ⏳ **Pending Tasks**

#### 7. **Memory Profiling Integration** ⏳
- **Scope:** Add memory allocation tracking for large buffers
- **Target Areas:**
  - Video frame allocation (>1MB buffers)
  - GPU texture memory
  - Render pass intermediate textures
- **Implementation:**
  - `PROFILE_LARGE_ALLOC()` and `PROFILE_LARGE_FREE()` macros
  - Integration in `TextureConverter` and video decoders

#### 8. **Configuration Integration** ⏳
- **Scope:** Add Tracy configuration options to INI and command line
- **Planned Features:**
  - `--enable-tracy` command line flag
  - INI configuration for profiling zones
  - Runtime enable/disable profiling
  - Configurable profiling verbosity levels

## Architecture Overview

### Profiling Flow
```
Application Start
├── Tracy Initialization (if ENABLE_TRACY=ON)
├── GPU Context Setup
│   ├── DirectX 11: PROFILE_GPU_D3D11_CONTEXT()
│   └── OpenGL: PROFILE_GPU_OPENGL_CONTEXT()
└── Runtime Profiling
    ├── CPU Zones: PROFILE_ZONE() macros
    ├── GPU Zones: PROFILE_GPU_*_ZONE() macros
    └── Data Collection: PROFILE_GPU_*_COLLECT() calls
```

### Performance Impact
- **Zero Overhead:** All macros expand to nothing when `TRACY_ENABLE` is not defined
- **Minimal Runtime Cost:** <1% CPU overhead when Tracy is enabled
- **GPU Profiling:** No impact on GPU performance, only adds timing queries

### Integration Points

#### Rendering Pipeline
```
Video Frame → TextureConverter → RenderTexture
     ↓ [PROFILE_RENDER()]
IRenderer::Present()
     ↓ [PROFILE_GPU_*_ZONE("RenderPassPipeline")]
Render Pass Pipeline → GPU Operations
     ↓ [PROFILE_GPU_*_COLLECT()]
SwapBuffers/Present → Display
```

#### Video Processing Pipeline
```
VideoDemuxer → [PROFILE_VIDEO_DEMUX()]
     ↓
VideoDecoder → [PROFILE_VIDEO_DECODE()]
     ↓
VideoSwitchingStrategy → [PROFILE_VIDEO_SWITCH()]
     ↓
Rendering Pipeline
```

## Usage Guide

### Building with Tracy
```bash
# Configure build with Tracy enabled
cmake -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=<vcpkg-path> -DENABLE_TRACY=ON ..

# Build release version
cmake --build . --config Release

# Run with Tracy server connected
./dual_stream.exe video1.mp4 video2.mp4
```

### Tracy Profiler Setup
1. Download Tracy profiler from: https://github.com/wolfpld/tracy
2. Build or download Tracy server application
3. Run Tracy server and connect to the application
4. Analyze performance data in real-time

### Profiling Best Practices
1. **Build Release:** Use Release builds for accurate performance measurements
2. **Consistent Environment:** Profile on target hardware configuration
3. **Multiple Runs:** Average results across multiple profiling sessions
4. **Focus Areas:** Use domain-specific macros to identify bottlenecks
5. **GPU Synchronization:** Ensure GPU collect calls after each frame

## File Structure

```
Tracy Integration Files:
├── CMakeLists.txt                          # Tracy build integration
├── src/core/
│   ├── TracyProfiler.h                     # Main profiling header
│   └── TracyProfiler.cpp                   # GPU context management
├── src/rendering/
│   ├── D3D11Renderer.h/cpp                 # DirectX 11 profiling
│   └── OpenGLRenderer.h/cpp                # OpenGL profiling
├── tools/
│   └── tracy_instrumentor.py               # Automatic instrumentation
└── TRACY.md                                # This documentation
```

## Development Workflow

### Adding New Profiling Zones
1. Include `#include "core/TracyProfiler.h"`
2. Add appropriate profiling macro at function entry
3. Ensure GPU collect calls are placed after GPU operations
4. Test with both Tracy enabled and disabled builds

### Extending GPU Profiling
1. Add GPU context initialization in renderer setup
2. Place GPU zone macros around GPU command submission
3. Add collect calls after GPU synchronization points
4. Verify profiling data appears in Tracy server

## TODO List

### High Priority
- [ ] **CUDA Profiling Support:** Add CUDA kernel and memory transfer profiling
- [ ] **Memory Profiling:** Track large allocation patterns and GPU memory usage
- [ ] **Configuration Integration:** Add command line and INI configuration options

### Medium Priority
- [ ] **Render Pass Profiling:** Add detailed profiling for individual render passes
- [ ] **Video Pipeline Profiling:** Add comprehensive video processing pipeline zones
- [ ] **Camera Profiling:** Add profiling for camera capture and face detection

### Low Priority
- [ ] **Performance Benchmarks:** Create automated performance regression tests
- [ ] **Profiling Documentation:** Add detailed profiling analysis guides
- [ ] **Tracy Integration Testing:** Add CI/CD tests for Tracy builds

## Performance Insights

### Expected Profiling Data
- **Frame Rate:** 60+ FPS target performance
- **CPU Rendering:** <5ms per frame for video processing
- **GPU Rendering:** <2ms per frame for rendering pipeline
- **Memory Allocation:** <100MB total working set for HD video

### Optimization Targets
1. **Video Decoding:** Hardware decode latency and throughput
2. **Texture Conversion:** CPU↔GPU memory transfer optimization
3. **Render Passes:** GPU shader performance and memory bandwidth
4. **Camera Processing:** Face detection algorithm performance

This Tracy integration provides comprehensive performance analysis capabilities for optimizing the DualStream Video Player across all major subsystems.