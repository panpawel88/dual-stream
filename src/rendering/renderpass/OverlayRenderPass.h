#pragma once
#include "IRenderPass.h"
#include "../../ui/IUIDrawable.h"
#include <memory>
#include <atomic>

/**
 * Abstract base class for overlay render passes.
 * Provides common overlay functionality while leaving rendering implementation
 * to derived classes for specific graphics APIs.
 */
class OverlayRenderPass : public IRenderPass, public IUIDrawable {
public:
    OverlayRenderPass();
    ~OverlayRenderPass() override;
    
    // Common IRenderPass interface - derived classes must implement these
    void Cleanup() override;
    
    // Individual component visibility controls
    void SetUIRegistryVisible(bool visible) { m_uiRegistryVisible = visible; }
    bool IsUIRegistryVisible() const { return m_uiRegistryVisible; }
    void ToggleUIRegistryVisibility() { m_uiRegistryVisible = !m_uiRegistryVisible; }
    
    void SetNotificationsVisible(bool visible) { m_notificationsVisible = visible; }
    bool IsNotificationsVisible() const { return m_notificationsVisible; }
    void ToggleNotificationsVisibility() { m_notificationsVisible = !m_notificationsVisible; }
    
    // IUIDrawable interface
    void DrawUI() override;
    std::string GetUIName() const override { return "Overlay Controls"; }
    std::string GetUICategory() const override { return "Render Passes"; }
    
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
    std::atomic<bool> m_uiRegistryVisible{false};     // UI panels disabled by default
    std::atomic<bool> m_notificationsVisible{true};   // Notifications enabled by default
    bool m_initialized = false;
    int m_width = 0;
    int m_height = 0;
};