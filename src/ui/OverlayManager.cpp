#include "OverlayManager.h"
#include "../rendering/OverlayRenderPass.h"

OverlayManager& OverlayManager::GetInstance() {
    static OverlayManager instance;
    return instance;
}

void OverlayManager::SetOverlayRenderPass(std::shared_ptr<OverlayRenderPass> overlayPass) {
    m_overlayPass = overlayPass;
}

void OverlayManager::ToggleOverlay() {
    if (m_overlayPass) {
        m_overlayPass->ToggleVisibility();
    }
}

void OverlayManager::SetOverlayVisible(bool visible) {
    if (m_overlayPass) {
        m_overlayPass->SetVisible(visible);
    }
}

bool OverlayManager::IsOverlayVisible() const {
    return m_overlayPass ? m_overlayPass->IsVisible() : false;
}