#pragma once
#include <memory>

class OverlayRenderPass;

class OverlayManager {
public:
    static OverlayManager& GetInstance();
    
    void SetOverlayRenderPass(std::shared_ptr<OverlayRenderPass> overlayPass);
    void ToggleOverlay();
    void SetOverlayVisible(bool visible);
    bool IsOverlayVisible() const;
    
private:
    OverlayManager() = default;
    ~OverlayManager() = default;
    
    std::shared_ptr<OverlayRenderPass> m_overlayPass;
};