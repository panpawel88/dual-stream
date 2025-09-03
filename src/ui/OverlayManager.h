#pragma once
#include <memory>

class OverlayRenderPass;

class OverlayManager {
public:
    static OverlayManager& GetInstance();
    
    void SetOverlayRenderPass(OverlayRenderPass* overlayPass);
    void ToggleOverlay();
    void SetOverlayVisible(bool visible);
    bool IsOverlayVisible() const;
    
private:
    OverlayManager() = default;
    ~OverlayManager() = default;
    
    OverlayRenderPass* m_overlayPass = nullptr;
};