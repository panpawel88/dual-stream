#include "OverlayManager.h"
#include "../rendering/renderpass/OverlayRenderPass.h"

OverlayManager& OverlayManager::GetInstance() {
    static OverlayManager instance;
    return instance;
}

void OverlayManager::SetOverlayRenderPass(OverlayRenderPass* overlayPass) {
    m_overlayPass = overlayPass;
}

void OverlayManager::ToggleUIRegistry() {
    if (m_overlayPass) {
        m_overlayPass->ToggleUIRegistryVisibility();
    }
}

void OverlayManager::SetUIRegistryVisible(bool visible) {
    if (m_overlayPass) {
        m_overlayPass->SetUIRegistryVisible(visible);
    }
}

bool OverlayManager::IsUIRegistryVisible() const {
    return m_overlayPass ? m_overlayPass->IsUIRegistryVisible() : false;
}

void OverlayManager::ToggleNotifications() {
    if (m_overlayPass) {
        m_overlayPass->ToggleNotificationsVisibility();
    }
}

void OverlayManager::SetNotificationsVisible(bool visible) {
    if (m_overlayPass) {
        m_overlayPass->SetNotificationsVisible(visible);
    }
}

bool OverlayManager::IsNotificationsVisible() const {
    return m_overlayPass ? m_overlayPass->IsNotificationsVisible() : false;
}