#pragma once
#include "IRenderPass.h"
#include <memory>
#include <atomic>

/**
 * Abstract base class for overlay render passes.
 * Provides common overlay functionality while leaving rendering implementation
 * to derived classes for specific graphics APIs.
 */
class OverlayRenderPass : public IRenderPass {
public:
    OverlayRenderPass();
    ~OverlayRenderPass() override;
    
    // Common IRenderPass interface - derived classes must implement these
    void Cleanup() override;
    
    // Overlay-specific methods
    void SetVisible(bool visible) { m_visible = visible; }
    bool IsVisible() const { return m_visible; }
    void ToggleVisibility() { m_visible = !m_visible; }
    
protected:
    // Common initialization logic (called by derived classes)
    bool InitializeCommon(int width, int height, void* hwnd = nullptr);
    
    // ImGui rendering logic (called by derived classes)
    void RenderImGuiContent();
    
    // Abstract methods that derived classes must implement
    virtual bool InitializeImGuiBackend() = 0;
    virtual void CleanupImGuiBackend() = 0;
    virtual void BeginImGuiFrame() = 0;
    virtual void EndImGuiFrame() = 0;
    
    // Common state
    std::atomic<bool> m_visible{false};
    bool m_initialized = false;
    int m_width = 0;
    int m_height = 0;
};