#pragma once

#include "../../ui/IUIDrawable.h"
#include "../processing/ICameraFrameListener.h"
#include "../CameraManager.h"
#include "CameraFrameTexture.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>

/**
 * ImGui UI component for camera control and live preview.
 * Implements IUIDrawable for integration with UIRegistry and ICameraFrameListener for frame preview.
 */
class CameraControlUI : public IUIDrawable, public ICameraFrameListener {
public:
    CameraControlUI();
    ~CameraControlUI();

    /**
     * Initialize with camera manager and renderer.
     * @param cameraManager Camera manager instance
     * @param renderer Current renderer for texture creation
     * @return true if initialization successful
     */
    bool Initialize(CameraManager* cameraManager, IRenderer* renderer);

    /**
     * Clean up resources.
     */
    void Cleanup();

    // IUIDrawable interface
    void DrawUI() override;
    std::string GetUIName() const override { return "Camera Control"; }
    std::string GetUICategory() const override { return "Camera"; }
    bool IsUIVisibleByDefault() const override { return false; }

    // ICameraFrameListener interface
    FrameProcessingResult ProcessFrame(std::shared_ptr<const CameraFrame> frame) override;
    ListenerPriority GetPriority() const override { return ListenerPriority::LOW; }
    std::string GetListenerId() const override { return "camera_control_ui"; }
    std::string GetListenerName() const override { return "Camera Control UI"; }
    bool CanProcessFormat(CameraFormat format) const override;
    FrameProcessingStats GetStats() const override;
    void ResetStats() override;

private:
    // Core components
    CameraManager* m_cameraManager;
    IRenderer* m_renderer;
    std::unique_ptr<CameraFrameTexture> m_frameTexture;

    // UI state
    bool m_previewEnabled;
    int m_brightness;
    int m_contrast;
    int m_saturation;
    int m_gain;

    // Frame preview
    std::shared_ptr<const CameraFrame> m_currentFrame;
    std::atomic<bool> m_hasNewFrame;
    std::chrono::steady_clock::time_point m_lastPreviewUpdate;
    double m_previewFPS;
    double m_maxPreviewFPS;

    // Thread safety
    mutable std::mutex m_frameMutex;
    mutable std::mutex m_uiStateMutex;

    // Internal state
    bool m_initialized;
    bool m_cameraAvailable;
    int m_frameWidth;
    int m_frameHeight;
    double m_currentFPS;

    // Stats tracking
    FrameProcessingStats m_stats;

    // UI drawing methods
    void DrawCameraProperties();
    void DrawFramePreview();
    void DrawCameraInfo();

    // Property handling
    void UpdateCameraProperty(CameraPropertyType property, int value);
    void ResetPropertiesToDefaults();
    void SyncUIWithCameraProperties();

    // Frame processing
    bool ShouldUpdatePreview() const;
    void UpdateFrameTexture();

    // Helper methods
    void LoadConfigurationSettings();
    void SaveConfigurationSettings();
    const char* FormatToString(CameraFormat format) const;
};