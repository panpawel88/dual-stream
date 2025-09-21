#pragma once

// Tracy Profiler Integration Wrapper
// This header provides a clean abstraction for Tracy profiling that allows
// zero-overhead when Tracy is disabled and consistent profiling API usage.

#ifdef TRACY_ENABLE
    #include <tracy/Tracy.hpp>
    #include <tracy/TracyD3D11.hpp>

    // Include OpenGL headers before Tracy's OpenGL header
    #include "../rendering/OpenGLHeaders.h"
    #include <tracy/TracyOpenGL.hpp>

    // Basic CPU profiling macros
    #define PROFILE_ZONE() ZoneScoped
    #define PROFILE_ZONE_N(name) ZoneScopedN(name)
    #define PROFILE_ZONE_C(color) ZoneScopedC(color)
    #define PROFILE_ZONE_NC(name, color) ZoneScopedNC(name, color)

    // Frame marking for main loop
    #define PROFILE_FRAME_MARK() FrameMark

    // GPU profiling macros - DirectX 11
    #define PROFILE_GPU_D3D11_CONTEXT(device, context) TracyD3D11Context(device, context)
    #define PROFILE_GPU_D3D11_ZONE(name) TracyD3D11Zone(name)
    #define PROFILE_GPU_D3D11_ZONE_C(name, color) TracyD3D11ZoneC(name, color)
    #define PROFILE_GPU_D3D11_COLLECT() TracyD3D11Collect()

    // GPU profiling macros - OpenGL
    #define PROFILE_GPU_OPENGL_CONTEXT() TracyGpuContext
    #define PROFILE_GPU_OPENGL_ZONE(name) TracyGpuZone(name)
    #define PROFILE_GPU_OPENGL_ZONE_C(name, color) TracyGpuZoneC(name, color)
    #define PROFILE_GPU_OPENGL_COLLECT() TracyGpuCollect

    // Memory profiling macros
    #define PROFILE_ALLOC(ptr, size) TracyAlloc(ptr, size)
    #define PROFILE_FREE(ptr) TracyFree(ptr)
    #define PROFILE_SECURE_ALLOC(ptr, size) TracySecureAlloc(ptr, size)
    #define PROFILE_SECURE_FREE(ptr) TracySecureFree(ptr)

    // Message and plot macros
    #define PROFILE_MESSAGE(text, size) TracyMessage(text, size)
    #define PROFILE_MESSAGE_L(text) TracyMessageL(text)
    #define PROFILE_PLOT(name, value) TracyPlot(name, value)
    #define PROFILE_PLOT_CONFIG(name, format) TracyPlotConfig(name, format)

    // Lock profiling macros
    #define PROFILE_LOCKABLE(type, varname) TracyLockable(type, varname)
    #define PROFILE_SHARED_LOCKABLE(type, varname) TracySharedLockable(type, varname)
    #define PROFILE_LOCK_GUARD(mutex) TracyLockGuard(mutex)
    #define PROFILE_SHARED_LOCK_GUARD(mutex) TracySharedLockGuard(mutex)

#else
    // When Tracy is disabled, all macros expand to nothing
    #define PROFILE_ZONE()
    #define PROFILE_ZONE_N(name)
    #define PROFILE_ZONE_C(color)
    #define PROFILE_ZONE_NC(name, color)

    #define PROFILE_FRAME_MARK()

    #define PROFILE_GPU_D3D11_CONTEXT(device, context)
    #define PROFILE_GPU_D3D11_ZONE(name)
    #define PROFILE_GPU_D3D11_ZONE_C(name, color)
    #define PROFILE_GPU_D3D11_COLLECT()

    #define PROFILE_GPU_OPENGL_CONTEXT()
    #define PROFILE_GPU_OPENGL_ZONE(name)
    #define PROFILE_GPU_OPENGL_ZONE_C(name, color)
    #define PROFILE_GPU_OPENGL_COLLECT()

    #define PROFILE_ALLOC(ptr, size)
    #define PROFILE_FREE(ptr)
    #define PROFILE_SECURE_ALLOC(ptr, size)
    #define PROFILE_SECURE_FREE(ptr)

    #define PROFILE_MESSAGE(text, size)
    #define PROFILE_MESSAGE_L(text)
    #define PROFILE_PLOT(name, value)
    #define PROFILE_PLOT_CONFIG(name, format)

    #define PROFILE_LOCKABLE(type, varname) type varname
    #define PROFILE_SHARED_LOCKABLE(type, varname) type varname
    #define PROFILE_LOCK_GUARD(mutex) std::lock_guard<decltype(mutex)> _tracy_lock_guard(mutex)
    #define PROFILE_SHARED_LOCK_GUARD(mutex) std::shared_lock<decltype(mutex)> _tracy_shared_lock_guard(mutex)

    // Need to include these for the disabled lock guards
    #include <mutex>
    #include <shared_mutex>
#endif

// Convenience macros for common profiling patterns in the video player

// Main subsystem profiling
#define PROFILE_VIDEO_DECODE() PROFILE_ZONE_N("Video Decode")
#define PROFILE_VIDEO_DEMUX() PROFILE_ZONE_N("Video Demux")
#define PROFILE_VIDEO_SWITCH() PROFILE_ZONE_N("Video Switch")
#define PROFILE_TEXTURE_CONVERT() PROFILE_ZONE_N("Texture Convert")
#define PROFILE_RENDER() PROFILE_ZONE_N("Render")
#define PROFILE_PRESENT() PROFILE_ZONE_N("Present")
#define PROFILE_UI_DRAW() PROFILE_ZONE_N("UI Draw")
#define PROFILE_CAMERA_CAPTURE() PROFILE_ZONE_N("Camera Capture")
#define PROFILE_FACE_DETECTION() PROFILE_ZONE_N("Face Detection")

// Render pass profiling
#define PROFILE_RENDER_PASS(name) PROFILE_ZONE_N(name)
#define PROFILE_GPU_RENDER_PASS_D3D11(name) PROFILE_GPU_D3D11_ZONE(name)
#define PROFILE_GPU_RENDER_PASS_OPENGL(name) PROFILE_GPU_OPENGL_ZONE(name)

// Memory profiling for large allocations
#define PROFILE_LARGE_ALLOC(ptr, size) \
    do { \
        if (size > 1024 * 1024) { /* Only track allocations > 1MB */ \
            PROFILE_ALLOC(ptr, size); \
        } \
    } while(0)

#define PROFILE_LARGE_FREE(ptr, size) \
    do { \
        if (size > 1024 * 1024) { /* Only track allocations > 1MB */ \
            PROFILE_FREE(ptr); \
        } \
    } while(0)

// Frame timing plots
#define PROFILE_FPS(fps) PROFILE_PLOT("FPS", fps)
#define PROFILE_FRAME_TIME(ms) PROFILE_PLOT("Frame Time (ms)", ms)
#define PROFILE_DECODE_TIME(ms) PROFILE_PLOT("Decode Time (ms)", ms)
#define PROFILE_RENDER_TIME(ms) PROFILE_PLOT("Render Time (ms)", ms)

// Configuration helper class for managing Tracy state
#ifdef TRACY_ENABLE
class TracyProfilerManager {
public:
    static TracyProfilerManager& GetInstance() {
        static TracyProfilerManager instance;
        return instance;
    }

    // Initialize Tracy GPU contexts based on renderer type
    void InitializeGPUContext(const char* rendererType);
    void ShutdownGPUContext();

    // Enable/disable profiling at runtime
    void SetProfilingEnabled(bool enabled) { m_enabled = enabled; }
    bool IsProfilingEnabled() const { return m_enabled; }

private:
    TracyProfilerManager() = default;
    bool m_enabled = true;
    bool m_gpuContextInitialized = false;
};
#else
// Stub implementation when Tracy is disabled
class TracyProfilerManager {
public:
    static TracyProfilerManager& GetInstance() {
        static TracyProfilerManager instance;
        return instance;
    }

    void InitializeGPUContext(const char* rendererType) {}
    void ShutdownGPUContext() {}
    void SetProfilingEnabled(bool enabled) {}
    bool IsProfilingEnabled() const { return false; }
};
#endif