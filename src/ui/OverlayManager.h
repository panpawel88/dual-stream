#pragma once
#include <memory>

class OverlayRenderPass;

class OverlayManager {
public:
    static OverlayManager& GetInstance();
    
    void SetOverlayRenderPass(OverlayRenderPass* overlayPass);
    void ToggleUIRegistry();
    void SetUIRegistryVisible(bool visible);
    bool IsUIRegistryVisible() const;
    
    void ToggleNotifications();
    void SetNotificationsVisible(bool visible);
    bool IsNotificationsVisible() const;
    
private:
    OverlayManager() = default;
    ~OverlayManager() = default;
    
    OverlayRenderPass* m_overlayPass = nullptr;
};